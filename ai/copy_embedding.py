from __future__ import annotations

"""Task-specific AI descriptor for *same-photo / edited-copy* detection.

Primary model: Meta Research SSCD (CVPR 2022), sscd_disc_mixup TorchScript.
The model was trained for image-copy detection, which is a better fit for this
bot than an ImageNet classifier embedding. The implementation is intentionally
optional: if PyTorch/model download fails, callers receive an empty list and the
classical pHash/SIFT/AKAZE/RANSAC pipeline continues to work.

Important: this module does NOT perform face identity recognition. Face/circle
locations are used only to propose image regions; SSCD compares photo content.
"""

import logging
import os
import threading
import urllib.request
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from config import (
    SSCD_ENABLED,
    SSCD_MODEL_URL,
    SSCD_MODEL_PATH,
    SSCD_INPUT_SIZE,
    SSCD_THREADS,
    SSCD_MAX_VIEWS,
    SSCD_BATCH_SIZE,
    SSCD_DOWNLOAD_TIMEOUT,
    SSCD_MAX_MODEL_BYTES,
)

log = logging.getLogger(__name__)

_model = None
_failed = False
_load_lock = threading.Lock()
_infer_lock = threading.Lock()
_face_cascade = None
_face_lock = threading.Lock()


def _download_model() -> bool:
    path = Path(SSCD_MODEL_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    # The official TorchScript model is ~99 MB. Treat tiny files as interrupted
    # downloads and fetch again atomically.
    if path.exists() and path.stat().st_size > 80_000_000:
        return True

    part = path.with_suffix(path.suffix + ".part")
    try:
        part.unlink(missing_ok=True)
    except Exception:
        pass

    try:
        log.info("首次启动：下载 SSCD 同图识别模型（约 99MB）…")
        req = urllib.request.Request(
            SSCD_MODEL_URL,
            headers={"User-Agent": "TG-Anti-Collision-Bot/1.9"},
        )
        total = 0
        with urllib.request.urlopen(req, timeout=SSCD_DOWNLOAD_TIMEOUT) as resp, open(part, "wb") as out:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > SSCD_MAX_MODEL_BYTES:
                    raise RuntimeError("SSCD模型文件异常过大，已中止下载")
                out.write(chunk)
        if total < 80_000_000:
            raise RuntimeError(f"SSCD模型下载不完整：{total} bytes")
        os.replace(part, path)
        log.info("SSCD模型下载完成：%.1f MB", total / 1024 / 1024)
        return True
    except Exception as exc:
        try:
            part.unlink(missing_ok=True)
        except Exception:
            pass
        log.warning("SSCD模型下载失败，将继续使用经典视觉+轻量AI备用：%s", exc)
        return False


def _load() -> bool:
    global _model, _failed
    if not SSCD_ENABLED or _failed:
        return False
    if _model is not None:
        return True
    with _load_lock:
        if _model is not None:
            return True
        if _failed:
            return False
        try:
            import torch

            torch.set_num_threads(max(1, int(SSCD_THREADS)))
            try:
                torch.set_num_interop_threads(1)
            except Exception:
                pass
            if not _download_model():
                _failed = True
                return False
            model = torch.jit.load(str(SSCD_MODEL_PATH), map_location="cpu")
            model.eval()
            _model = model
            log.info("SSCD 同图识别 AI 模型加载成功")
            return True
        except Exception as exc:
            _failed = True
            log.warning("SSCD模型加载失败，将继续使用备用视觉算法：%s", exc)
            return False


def preload() -> bool:
    return _load()


def available() -> bool:
    return _load()


def _get_face_cascade():
    global _face_cascade
    if _face_cascade is not None:
        return _face_cascade
    with _face_lock:
        if _face_cascade is None:
            try:
                c = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                _face_cascade = c if not c.empty() else False
            except Exception:
                _face_cascade = False
    return _face_cascade


def _trim_uniform_border(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    if h < 80 or w < 80:
        return img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edge = max(2, min(h, w) // 80)
    samples = np.concatenate(
        [
            gray[:edge].reshape(-1),
            gray[-edge:].reshape(-1),
            gray[:, :edge].reshape(-1),
            gray[:, -edge:].reshape(-1),
        ]
    )
    med = float(np.median(samples))
    if not (med <= 25 or med >= 230 or float(np.std(samples)) < 8):
        return img
    diff = np.abs(gray.astype(np.float32) - med)
    ys, xs = np.where(diff > 18)
    if len(xs) < 0.18 * h * w:
        return img
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    if x1 - x0 < 0.35 * w or y1 - y0 < 0.35 * h:
        return img
    pad = max(2, int(min(h, w) * 0.01))
    return img[max(0, y0 - pad): min(h, y1 + pad), max(0, x0 - pad): min(w, x1 + pad)]


def _center_crop(img: np.ndarray, frac: float) -> np.ndarray:
    h, w = img.shape[:2]
    nh, nw = max(24, int(h * frac)), max(24, int(w * frac))
    y0, x0 = max(0, (h - nh) // 2), max(0, (w - nw) // 2)
    return img[y0:y0 + nh, x0:x0 + nw]


def _square_window(img: np.ndarray, cx: float, cy: float, frac: float) -> np.ndarray:
    h, w = img.shape[:2]
    size = max(24, int(min(h, w) * frac))
    x0 = int(cx * w - size / 2)
    y0 = int(cy * h - size / 2)
    x0 = max(0, min(w - size, x0))
    y0 = max(0, min(h - size, y0))
    return img[y0:y0 + size, x0:x0 + size]


def _face_views(img: np.ndarray) -> list[tuple[str, np.ndarray]]:
    c = _get_face_cascade()
    if c is False:
        return []
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        scale = min(1.0, 800.0 / max(h, w))
        work = gray if scale == 1.0 else cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        faces = c.detectMultiScale(work, scaleFactor=1.10, minNeighbors=4, minSize=(36, 36))
        if not len(faces):
            return []
        if scale != 1.0:
            inv = 1.0 / scale
            faces = [(int(x * inv), int(y * inv), int(fw * inv), int(fh * inv)) for x, y, fw, fh in faces]
        out = []
        for idx, (x, y, fw, fh) in enumerate(sorted(faces, key=lambda r: r[2] * r[3], reverse=True)[:2]):
            pad = int(max(fw, fh) * 0.55)
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(w, x + fw + pad), min(h, y + fh + pad)
            if (x1 - x0) * (y1 - y0) >= 1600:
                out.append((f"face{idx}", img[y0:y1, x0:x1]))
        return out
    except Exception:
        return []


def _circle_view(img: np.ndarray) -> tuple[str, np.ndarray] | None:
    """Find one likely circular avatar. This proposes a region only; no identity logic."""
    try:
        h, w = img.shape[:2]
        scale = min(1.0, 640.0 / max(h, w))
        small = img if scale == 1.0 else cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 1.4)
        mn = max(16, int(min(gray.shape[:2]) * 0.08))
        mx = max(mn + 2, int(min(gray.shape[:2]) * 0.42))
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(30, mn * 2),
            param1=100,
            param2=28,
            minRadius=mn,
            maxRadius=mx,
        )
        if circles is None:
            return None
        inv = 1.0 / scale
        # Prefer the largest circle; social avatar circles are typically prominent.
        x, y, r = max(circles[0], key=lambda c: c[2])
        x, y, r = int(x * inv), int(y * inv), int(r * inv)
        pad = int(r * 0.18)
        x0, y0 = max(0, x - r - pad), max(0, y - r - pad)
        x1, y1 = min(w, x + r + pad), min(h, y + r + pad)
        if (x1 - x0) * (y1 - y0) < 1600:
            return None
        return "circle", img[y0:y1, x0:x1]
    except Exception:
        return None


def _texture_window(img: np.ndarray) -> tuple[str, np.ndarray] | None:
    """Pick one high-information half-size window for profile-page screenshots."""
    try:
        h, w = img.shape[:2]
        if min(h, w) < 100:
            return None
        candidates = []
        for cy in (0.30, 0.50, 0.70):
            for cx in (0.30, 0.50, 0.70):
                crop = _square_window(img, cx, cy, 0.58)
                gray = cv2.cvtColor(cv2.resize(crop, (128, 128), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 60, 160)
                edge_density = float((edges > 0).mean())
                std = float(gray.std())
                # Avoid selecting mostly text/noise: require both variance and a reasonable edge range.
                score = std + 70.0 * min(edge_density, 0.18)
                candidates.append((score, crop))
        score, crop = max(candidates, key=lambda x: x[0])
        return ("texture", crop) if score > 18 else None
    except Exception:
        return None


def copy_region_views(img: np.ndarray, max_views: int | None = None) -> list[tuple[str, np.ndarray]]:
    """Return diverse regions for copy detection without assuming one app layout.

    Views are deliberately platform-independent: full/trimmed image, center, a few
    overlapping windows, optional face/circle region, and one high-information
    window. This helps when the same underlying customer photo appears as a saved
    image, a circular avatar, or inside a profile-page screenshot.
    """
    max_views = max(1, int(max_views or SSCD_MAX_VIEWS))
    trimmed = _trim_uniform_border(img)
    views: list[tuple[str, np.ndarray]] = [("full", trimmed)]

    # Region proposals that are cheap and deterministic.
    views.extend(_face_views(trimmed))
    cv = _circle_view(trimmed)
    if cv is not None:
        views.append(cv)
    views.append(("center", _center_crop(trimmed, 0.72)))

    # Overlapping windows catch avatars/screenshots that are not centered.
    for kind, cx, cy in (
        ("top", 0.50, 0.30),
        ("left", 0.30, 0.50),
        ("right", 0.70, 0.50),
        ("bottom", 0.50, 0.70),
    ):
        views.append((kind, _square_window(trimmed, cx, cy, 0.62)))

    tv = _texture_window(trimmed)
    if tv is not None:
        views.append(tv)

    # Remove duplicate/near-empty regions cheaply using small pixel fingerprints.
    result: list[tuple[str, np.ndarray]] = []
    seen = set()
    for kind, view in views:
        if view is None or view.size == 0:
            continue
        h, w = view.shape[:2]
        if h < 24 or w < 24:
            continue
        thumb = cv2.resize(view, (24, 24), interpolation=cv2.INTER_AREA)
        key = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 55])[1].tobytes()
        if key in seen:
            continue
        seen.add(key)
        result.append((kind, view))
        if len(result) >= max_views:
            break
    return result


def _preprocess(view: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)
    # SSCD docs recommend direct square resize as an efficient copy-detection preprocessing path.
    x = cv2.resize(rgb, (SSCD_INPUT_SIZE, SSCD_INPUT_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - mean) / std
    return np.transpose(x, (2, 0, 1))


def extract_copy_features(img: np.ndarray, max_views: int | None = None) -> list[dict]:
    if not _load():
        return []
    try:
        import torch

        views = copy_region_views(img, max_views=max_views)
        if not views:
            return []
        out: list[dict] = []
        batch_size = max(1, int(SSCD_BATCH_SIZE))
        # One shared CPU model: serialize inference to avoid Railway memory/CPU
        # spikes when live checks and background reindex overlap.
        with _infer_lock, torch.inference_mode():
            for start in range(0, len(views), batch_size):
                chunk = views[start:start + batch_size]
                arr = np.stack([_preprocess(v) for _, v in chunk], axis=0)
                tensor = torch.from_numpy(arr)
                emb = _model(tensor)
                if isinstance(emb, (tuple, list)):
                    emb = emb[0]
                emb = emb.detach().cpu().float().numpy()
                for (kind, _), vec in zip(chunk, emb):
                    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
                    n = float(np.linalg.norm(vec))
                    if n <= 1e-8:
                        continue
                    vec = vec / n
                    out.append({"kind": kind, "feature": vec})
        return out
    except Exception as exc:
        log.warning("SSCD特征提取失败，本次使用备用视觉算法：%s", exc)
        return []


def max_cosine(query_views: Iterable[dict], candidate_views: Iterable[dict]) -> tuple[float, str, str]:
    q = list(query_views or [])
    c = list(candidate_views or [])
    if not q or not c:
        return 0.0, "", ""
    best = -1.0
    qk = ck = ""
    try:
        cm = np.stack([np.asarray(x["feature"], dtype=np.float32) for x in c], axis=0)
        qm = np.stack([np.asarray(x["feature"], dtype=np.float32) for x in q], axis=0)
        sims = qm @ cm.T
        idx = np.unravel_index(int(np.argmax(sims)), sims.shape)
        best = float(sims[idx])
        qk = str(q[idx[0]].get("kind") or "")
        ck = str(c[idx[1]].get("kind") or "")
    except Exception:
        return 0.0, "", ""
    return max(0.0, min(100.0, best * 100.0)), qk, ck
