import cv2
import numpy as np
from config import (
    SIMILAR_THRESHOLD,
    PHASH_DIRECT_THRESHOLD,
    MAX_FALLBACK_CANDIDATES,
    FAST_CANDIDATE_LIMIT,
    DEEP_CANDIDATE_LIMIT,
)
from core.database import get_conn, now_iso
from core.image_features import (
    extract_features,
    extract_light_features,
    enrich_deep_features,
    hash_file_pair,
    hamming_score,
    pack_f32,
    unpack_f32,
    pack_orb,
    unpack_orb,
)
from ai.embedding import cosine_score


def _hist_score(a, b):
    if a is None or b is None:
        return 0.0
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if not len(a) or not len(b):
        return 0.0
    corr = float(cv2.compareHist(a, b, cv2.HISTCMP_CORREL))
    return max(0.0, min(100.0, (corr + 1.0) * 50.0))


def _orb_score(a, b):
    if a is None or b is None or len(a) < 5 or len(b) < 5:
        return 0.0
    try:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        pairs = bf.knnMatch(a, b, k=2)
        good = 0
        for pair in pairs:
            if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
                good += 1
        denom = max(12, min(len(a), len(b)))
        return min(100.0, good / denom * 140.0)
    except Exception:
        return 0.0


def _candidate_rows(conn, f):
    """只取轻量字段。ORB/AI 大 BLOB 延迟到 Top-N 精确候选阶段再读。"""
    clauses = []
    params = []
    for i, v in enumerate(f["p_bands"]):
        if v:
            clauses.append(f"p{i}=?")
            params.append(v)

    select = "id,phash,dhash,roi_phash,customer_id"
    if clauses:
        sql = (
            f"SELECT {select} FROM images WHERE (" + " OR ".join(clauses) +
            ") ORDER BY id DESC LIMIT ?"
        )
        rows = conn.execute(sql, params + [FAST_CANDIDATE_LIMIT]).fetchall()
        if rows:
            return rows

    # 无 band 命中时只做有限回退，不再对几千张图全部跑 ORB。
    return conn.execute(
        f"SELECT {select} FROM images ORDER BY id DESC LIMIT ?",
        (MAX_FALLBACK_CANDIDATES,),
    ).fetchall()


def _light_item(f, row):
    p = hamming_score(f["phash"], row["phash"])
    d = hamming_score(f["dhash"], row["dhash"])
    roi = hamming_score(f["roi_phash"], row["roi_phash"])
    # 轻量阶段不读取 color_hist/ORB/AI 大字段，减少远端 PostgreSQL 网络传输。
    rank = 0.56 * p + 0.16 * d + 0.28 * roi
    return {"row": row, "rank": rank, "p": p, "d": d, "roi": roi}


def _fetch_deep_rows(conn, ids):
    if not ids:
        return {}
    placeholders = ",".join(["?"] * len(ids))
    rows = conn.execute(
        f"SELECT id,color_hist,orb_desc,orb_rows,ai_feature FROM images WHERE id IN ({placeholders})",
        ids,
    ).fetchall()
    return {int(r["id"]): r for r in rows}


def check_image(path):
    """
    V1.5 Fast Path：
      1) SHA256/MD5 先查数据库，完全同图不跑 OpenCV/AI；
      2) pHash/dHash/ROI/色彩只做轻量候选排序；
      3) 只对 Top-N 候选运行 ORB + AI。
    """
    sha256, md5 = hash_file_pair(path)
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT id FROM images WHERE sha256=? OR md5=? LIMIT 1",
            (sha256, md5),
        ).fetchone()
        if row:
            return {
                "type": "same",
                "score": 100.0,
                "matched_image_id": row["id"],
                "features": {"sha256": sha256, "md5": md5},
                "match_type": "同一图片",
            }

        f, img = extract_light_features(path, sha256, md5)
        light = [_light_item(f, row) for row in _candidate_rows(conn, f)]
        light.sort(key=lambda x: x["rank"], reverse=True)

        # pHash 近乎一致且 dHash/主体区域至少有一项也很高时，直接结束。
        # 这是压缩、改尺寸、轻微重编码最常见的情况，不必跑 AI。
        if light:
            top = light[0]
            if (
                top["p"] >= PHASH_DIRECT_THRESHOLD
                and max(top["d"], top["roi"]) >= 90.0
            ):
                score = min(
                    99.5,
                    max(
                        90.0 + (top["p"] - PHASH_DIRECT_THRESHOLD) * 1.5,
                        0.72 * top["p"] + 0.18 * top["roi"] + 0.10 * top["d"],
                    ),
                )
                return {
                    "type": "similar",
                    "score": round(score, 2),
                    "matched_image_id": top["row"]["id"],
                    "features": f,
                    "match_type": "相似图片",
                    "detail": {
                        "phash": round(top["p"], 1),
                        "orb": 0.0,
                        "ai": 0.0,
                    },
                }

        # 只给最可能的少量候选做重计算。
        top_candidates = light[:DEEP_CANDIDATE_LIMIT]
        if not top_candidates:
            # 没有历史图片时，新客户直接返回，连 ORB/AI 都不运行。
            return {"type": "new", "score": 0.0, "features": f}

        enrich_deep_features(f, img)
        deep_map = _fetch_deep_rows(
            conn, [int(item["row"]["id"]) for item in top_candidates]
        )

        best = None
        for item in top_candidates:
            row = item["row"]
            deep = deep_map.get(int(row["id"]))
            if deep is None:
                orb = ai = 0.0
            else:
                orb = _orb_score(
                    f["orb"], unpack_orb(deep["orb_desc"], deep["orb_rows"])
                )
                ai = cosine_score(f["ai_feature"], unpack_f32(deep["ai_feature"]))

            p = item["p"]
            d = item["d"]
            roi = item["roi"]
            hist = _hist_score(
                f["color_hist"], unpack_f32(deep["color_hist"]) if deep is not None else None
            )
            score = 0.30 * p + 0.10 * d + 0.15 * roi + 0.10 * hist + 0.25 * orb + 0.10 * ai
            if p >= PHASH_DIRECT_THRESHOLD:
                score = max(score, 90.0 + (p - PHASH_DIRECT_THRESHOLD) * 1.5)
            score = min(100.0, score)

            current = {
                "row": row,
                "score": score,
                "p": p,
                "d": d,
                "roi": roi,
                "hist": hist,
                "orb": orb,
                "ai": ai,
            }
            if best is None or score > best["score"]:
                best = current

        if best and best["score"] >= SIMILAR_THRESHOLD:
            match_type = (
                "AI图片匹配"
                if best["ai"] >= 88 and best["p"] < PHASH_DIRECT_THRESHOLD
                else "相似图片"
            )
            return {
                "type": "similar",
                "score": round(best["score"], 2),
                "matched_image_id": best["row"]["id"],
                "features": f,
                "match_type": match_type,
                "detail": {
                    "phash": round(best["p"], 1),
                    "orb": round(best["orb"], 1),
                    "ai": round(best["ai"], 1),
                },
            }
        return {"type": "new", "score": 0.0, "features": f}
    finally:
        conn.close()


def save_customer_and_image(path, file_id, file_unique_id, submitter, submitter_id, chat_id, customer_data, raw_text, source="live", source_message_id="", object_key=None, features=None):
    """
    Persist a customer + image record.

    features: check_image() 已算过的完整特征可以直接复用，避免入库时再跑一遍
              pHash/ORB/AI。历史导入或旧调用方不传时仍自动计算。
    """
    f = features or extract_features(path)
    # 快速直返路径可能只有轻量特征；正式入库前补齐 ORB/AI。
    if f.get("orb") is None or "ai_feature" not in f:
        f = extract_features(path)

    conn = get_conn()
    try:
        exists = conn.execute(
            "SELECT id,customer_id FROM images WHERE sha256=? OR md5=? LIMIT 1",
            (f["sha256"], f["md5"]),
        ).fetchone()
        if exists:
            return exists["customer_id"], exists["id"], False

        cur = conn.execute("""
            INSERT INTO customers(name,age,job,income,work_year,software,receiver,raw_text,submitter,submitter_id,chat_id,source,source_message_id,created_time)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            customer_data.get("name", ""), customer_data.get("age", ""), customer_data.get("job", ""), customer_data.get("income", ""),
            customer_data.get("work_year", ""), customer_data.get("software", ""), customer_data.get("receiver", ""), raw_text or "",
            submitter, submitter_id, chat_id, source, str(source_message_id or ""), now_iso()
        ))
        customer_id = cur.lastrowid
        orb_blob, orb_rows = pack_orb(f["orb"])
        stored_path = object_key if object_key else str(path)
        values = [
            file_id, file_unique_id, stored_path, f["sha256"], f["md5"], f["phash"], f["dhash"], f["roi_phash"],
            *f["p_bands"], pack_f32(f["color_hist"]), orb_blob, orb_rows, pack_f32(f["ai_feature"]), f["ai_hash"],
            *f["a_bands"], customer_id, submitter, submitter_id, chat_id, source, now_iso()
        ]
        q = """
        INSERT INTO images(file_id,file_unique_id,file_path,sha256,md5,phash,dhash,roi_phash,
        p0,p1,p2,p3,p4,p5,p6,p7,color_hist,orb_desc,orb_rows,ai_feature,ai_hash,
        a0,a1,a2,a3,a4,a5,a6,a7,customer_id,submitter,submitter_id,chat_id,source,created_time)
        VALUES(""" + ",".join(["?"] * 35) + ")"
        cur = conn.execute(q, values)
        image_id = cur.lastrowid
        conn.commit()
        return customer_id, image_id, True
    finally:
        conn.close()


def create_collision(query_sha256, matched_image_id, submitter, submitter_id, chat_id, match_type, score):
    conn = get_conn()
    cur = conn.execute("""
        INSERT INTO collision_records(query_sha256,matched_image_id,query_submitter,query_submitter_id,chat_id,match_type,score,status,created_time)
        VALUES(?,?,?,?,?,?,?,'pending',?)
    """, (query_sha256, matched_image_id, submitter, submitter_id, chat_id, match_type, float(score), now_iso()))
    cid = cur.lastrowid
    conn.commit(); conn.close(); return cid


def save_customer_only(submitter, submitter_id, chat_id, customer_data, raw_text, source="history", source_message_id=""):
    """保存纯文字客户资料（没有关联图片）。用于历史导入时图片文件缺失的情况。"""
    conn = get_conn()
    cur = conn.execute("""
        INSERT INTO customers(name,age,job,income,work_year,software,receiver,raw_text,submitter,submitter_id,chat_id,source,source_message_id,created_time)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        customer_data.get("name", ""), customer_data.get("age", ""), customer_data.get("job", ""),
        customer_data.get("income", ""), customer_data.get("work_year", ""), customer_data.get("software", ""),
        customer_data.get("receiver", ""), raw_text or "",
        submitter, submitter_id, chat_id, source, str(source_message_id or ""), now_iso()
    ))
    customer_id = cur.lastrowid
    conn.commit()
    conn.close()
    return customer_id


def update_customer_fields(customer_id: int, fields: dict, submitter_id: str | None = None, operator: str | None = None, operator_id: str | None = None) -> bool:
    allowed = {"name", "age", "job", "income", "work_year", "software", "receiver"}
    updates = {k: str(v).strip() for k, v in fields.items() if k in allowed}
    if not updates:
        return False
    conn = get_conn()
    try:
        old_row = conn.execute("SELECT * FROM customers WHERE id=? LIMIT 1", (int(customer_id),)).fetchone()
        old_data = dict(old_row) if old_row else {}
        ts = now_iso()
        op_name = operator or submitter_id or "unknown"
        op_id = operator_id or submitter_id
        set_parts = list(updates.keys()) + ["updated_time", "last_updated_by"]
        set_clause = ", ".join(f"{k}=?" for k in set_parts)
        update_values = list(updates.values()) + [ts, op_name]
        if submitter_id is not None:
            values = update_values + [int(customer_id), str(submitter_id)]
            sql = f"UPDATE customers SET {set_clause} WHERE id=? AND submitter_id=?"
        else:
            values = update_values + [int(customer_id)]
            sql = f"UPDATE customers SET {set_clause} WHERE id=?"
        cur = conn.execute(sql, values)
        if cur.rowcount > 0:
            for field_name, new_val in updates.items():
                old_val = str(old_data.get(field_name) or "")
                conn.execute(
                    "INSERT INTO customer_edit_log(customer_id, field_name, old_value, new_value, operator, operator_id, changed_time) VALUES(?,?,?,?,?,?,?)",
                    (int(customer_id), field_name, old_val, new_val, op_name, op_id, ts),
                )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def get_customer_by_id(customer_id: int) -> dict | None:
    try:
        conn = get_conn()
        row = conn.execute("SELECT * FROM customers WHERE id=? LIMIT 1", (int(customer_id),)).fetchone()
        conn.close()
        return dict(row) if row else None
    except Exception:
        return None


def confirm_collision(collision_id, status, confirmer, confirmer_id):
    if status not in {"confirmed", "false_positive"}:
        return False
    conn = get_conn()
    cur = conn.execute("""
        UPDATE collision_records SET status=?,confirmer=?,confirmer_id=?,confirmed_time=?
        WHERE id=? AND status='pending'
    """, (status, confirmer, confirmer_id, now_iso(), collision_id))
    changed = cur.rowcount > 0
    conn.commit(); conn.close(); return changed
