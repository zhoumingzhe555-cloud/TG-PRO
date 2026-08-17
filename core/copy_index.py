from __future__ import annotations

"""SQLite-backed SSCD descriptor storage and candidate retrieval."""

import logging
from collections import defaultdict

import numpy as np

from ai.embedding import simhash64
from config import SSCD_FULL_SCAN_LIMIT, SSCD_TOP_K, SSCD_LSH_CANDIDATE_LIMIT
from core.database import now_iso
from core.image_features import bands64

log = logging.getLogger(__name__)
MODEL_NAME = "sscd_disc_mixup"


def pack_copy_feature(vec) -> tuple[bytes | None, int]:
    if vec is None:
        return None, 0
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    if not arr.size:
        return None, 0
    n = float(np.linalg.norm(arr))
    if n > 1e-8:
        arr = arr / n
    # float16 halves SQLite/Volume size; exact scoring converts back to float32.
    return arr.astype(np.float16).tobytes(), int(arr.size)


def unpack_copy_feature(blob, dim: int):
    if not blob or not dim:
        return None
    arr = np.frombuffer(blob, dtype=np.float16, count=int(dim)).astype(np.float32)
    n = float(np.linalg.norm(arr))
    if n > 1e-8:
        arr /= n
    return arr


def save_copy_features(conn, image_id: int, views: list[dict] | None) -> int:
    if not views:
        return 0
    saved = 0
    for item in views:
        kind = str(item.get("kind") or "full")[:40]
        vec = item.get("feature")
        blob, dim = pack_copy_feature(vec)
        if not blob:
            continue
        sh = simhash64(np.asarray(vec, dtype=np.float32))
        conn.execute(
            """INSERT INTO image_copy_features(image_id,kind,feature,dim,model_name,simhash,created_time)
               VALUES(?,?,?,?,?,?,?)
               ON CONFLICT(image_id,kind) DO UPDATE SET
                 feature=excluded.feature,dim=excluded.dim,model_name=excluded.model_name,
                 simhash=excluded.simhash,created_time=excluded.created_time""",
            (int(image_id), kind, blob, dim, MODEL_NAME, sh, now_iso()),
        )
        conn.execute("DELETE FROM image_copy_lsh WHERE image_id=? AND kind=?", (int(image_id), kind))
        for band, value in enumerate(bands64(sh)):
            if value:
                conn.execute(
                    "INSERT OR REPLACE INTO image_copy_lsh(image_id,kind,band,value) VALUES(?,?,?,?)",
                    (int(image_id), kind, int(band), value),
                )
        saved += 1
    return saved


def load_copy_features(conn, image_ids: list[int] | set[int]) -> dict[int, list[dict]]:
    ids = list({int(x) for x in image_ids if x is not None})
    if not ids:
        return {}
    out: dict[int, list[dict]] = defaultdict(list)
    step = 800
    for start in range(0, len(ids), step):
        chunk = ids[start:start + step]
        ph = ",".join("?" * len(chunk))
        rows = conn.execute(
            f"SELECT c.image_id,c.kind,c.feature,c.dim FROM image_copy_features c JOIN images i ON i.id=c.image_id WHERE COALESCE(i.trusted,1)=1 AND c.image_id IN ({ph})",
            chunk,
        ).fetchall()
        for r in rows:
            vec = unpack_copy_feature(r["feature"], r["dim"])
            if vec is not None:
                out[int(r["image_id"])].append({"kind": r["kind"], "feature": vec})
    return dict(out)


def _query_matrix(query_views: list[dict]) -> np.ndarray | None:
    arrs = []
    for x in query_views or []:
        try:
            v = np.asarray(x["feature"], dtype=np.float32).reshape(-1)
            n = float(np.linalg.norm(v))
            if n > 1e-8:
                arrs.append(v / n)
        except Exception:
            continue
    if not arrs:
        return None
    return np.stack(arrs, axis=0)


def _score_rows(rows, qmat: np.ndarray, image_best: dict[int, tuple[float, str]], limit: int):
    # Batch decoding avoids a large one-shot memory spike.
    feats = []
    meta = []
    for r in rows:
        vec = unpack_copy_feature(r["feature"], r["dim"])
        if vec is None or vec.size != qmat.shape[1]:
            continue
        feats.append(vec)
        meta.append((int(r["image_id"]), str(r["kind"] or "")))
        if len(feats) >= 512:
            _apply_batch(feats, meta, qmat, image_best)
            feats, meta = [], []
    if feats:
        _apply_batch(feats, meta, qmat, image_best)


def _apply_batch(feats, meta, qmat, image_best):
    mat = np.stack(feats, axis=0).astype(np.float32, copy=False)
    sims = mat @ qmat.T
    best = sims.max(axis=1)
    for (image_id, kind), score in zip(meta, best):
        pct = max(0.0, min(100.0, float(score) * 100.0))
        old = image_best.get(image_id)
        if old is None or pct > old[0]:
            image_best[image_id] = (pct, kind)


def _lsh_candidate_ids(conn, query_views: list[dict], limit: int) -> list[int]:
    pairs = []
    for item in query_views or []:
        try:
            sh = simhash64(item["feature"])
        except Exception:
            sh = ""
        for band, value in enumerate(bands64(sh)):
            if value:
                pairs.append((band, value))
    pairs = list(dict.fromkeys(pairs))
    if not pairs:
        return []
    clauses = []
    params = []
    for band, value in pairs:
        clauses.append("(band=? AND value=?)")
        params.extend([int(band), value])
    # Count matched bands across every query view. Exact cosine is computed next.
    sql = (
        "SELECT image_id, COUNT(*) AS hits FROM image_copy_lsh WHERE "
        + " OR ".join(clauses)
        + " GROUP BY image_id ORDER BY hits DESC LIMIT ?"
    )
    rows = conn.execute(sql, params + [int(limit)]).fetchall()
    return [int(r["image_id"]) for r in rows]


def copy_candidate_scores(conn, query_views: list[dict], top_k: int | None = None) -> list[dict]:
    """Return top image IDs by SSCD cosine similarity.

    For the current-size customer library we scan compressed descriptors exactly,
    which avoids hash-recall misses. At larger scale we switch to LSH candidate
    generation and still perform exact cosine scoring on the candidate set.
    """
    qmat = _query_matrix(query_views)
    if qmat is None:
        return []
    top_k = max(1, int(top_k or SSCD_TOP_K))
    count = int(conn.execute("SELECT COUNT(DISTINCT c.image_id) FROM image_copy_features c JOIN images i ON i.id=c.image_id WHERE COALESCE(i.trusted,1)=1").fetchone()[0] or 0)
    if count <= 0:
        return []

    image_best: dict[int, tuple[float, str]] = {}
    if count <= int(SSCD_FULL_SCAN_LIMIT):
        cur = conn.execute("SELECT c.image_id,c.kind,c.feature,c.dim FROM image_copy_features c JOIN images i ON i.id=c.image_id WHERE COALESCE(i.trusted,1)=1")
        while True:
            rows = cur.fetchmany(1000)
            if not rows:
                break
            _score_rows(rows, qmat, image_best, top_k)
    else:
        ids = _lsh_candidate_ids(conn, query_views, int(SSCD_LSH_CANDIDATE_LIMIT))
        if not ids:
            return []
        step = 700
        for start in range(0, len(ids), step):
            chunk = ids[start:start + step]
            ph = ",".join("?" * len(chunk))
            rows = conn.execute(
                f"SELECT c.image_id,c.kind,c.feature,c.dim FROM image_copy_features c JOIN images i ON i.id=c.image_id WHERE COALESCE(i.trusted,1)=1 AND c.image_id IN ({ph})",
                chunk,
            ).fetchall()
            _score_rows(rows, qmat, image_best, top_k)

    ranked = sorted(image_best.items(), key=lambda kv: kv[1][0], reverse=True)[:top_k]
    return [
        {"image_id": image_id, "score": round(score_kind[0], 3), "kind": score_kind[1]}
        for image_id, score_kind in ranked
    ]
