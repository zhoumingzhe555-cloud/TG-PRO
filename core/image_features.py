from __future__ import annotations

import hashlib
import io
import math
import threading
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except Exception:
    pass
try:
    import pillow_avif  # noqa: F401
except Exception:
    pass

from ai.embedding import extract_ai_feature, simhash64
from ai.copy_embedding import extract_copy_features
from config import MAX_PROCESS_MEGAPIXELS

_face_cascade = None
_face_lock = threading.Lock()
_orb_local = threading.local()
_sift_local = threading.local()
_akaze_local = threading.local()


def _limit_pixels(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    mp = (h * w) / 1_000_000.0
    if mp <= MAX_PROCESS_MEGAPIXELS or mp <= 0:
        return img
    scale = math.sqrt(MAX_PROCESS_MEGAPIXELS / mp)
    return cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)


def read_image(path):
    """Read with EXIF orientation and optional HEIC/AVIF support, then return BGR."""
    p = Path(path)
    try:
        with Image.open(p) as im:
            im = ImageOps.exif_transpose(im)
            w,h=im.size
            mp=(w*h)/1_000_000.0
            if mp>MAX_PROCESS_MEGAPIXELS and mp>0:
                scale=math.sqrt(MAX_PROCESS_MEGAPIXELS/mp)
                target=(max(1,int(w*scale)),max(1,int(h*scale)))
                im=im.resize(target,Image.Resampling.LANCZOS)
            im = im.convert("RGB")
            arr = np.asarray(im)
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            return _limit_pixels(bgr)
    except Exception:
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("无法读取图片")
        return _limit_pixels(img)


def hash_file(path, algo="sha256"):
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_file_pair(path):
    sha = hashlib.sha256()
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(chunk)
            md5.update(chunk)
    return sha.hexdigest(), md5.hexdigest()


def _bits_to_hex(bits):
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def phash64_img(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(gray)
    block = dct[:8, :8].flatten()
    med = np.median(block[1:])
    return _bits_to_hex(block > med)


def dhash64_img(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return _bits_to_hex(bits)


def hamming_score(h1, h2):
    if not h1 or not h2:
        return 0.0
    try:
        diff = (int(h1, 16) ^ int(h2, 16)).bit_count()
        return max(0.0, 100.0 * (1.0 - diff / 64.0))
    except Exception:
        return 0.0


def bands64(hex64):
    h = (hex64 or "").zfill(16)[-16:]
    return [h[i:i + 2] for i in range(0, 16, 2)] if hex64 else [""] * 8


def _trim_uniform_border(img: np.ndarray) -> np.ndarray:
    """Remove large near-uniform black/white/solid borders without cropping real content aggressively."""
    h, w = img.shape[:2]
    if h < 80 or w < 80:
        return img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edge = max(2, min(h, w) // 80)
    samples = np.concatenate([
        gray[:edge, :].reshape(-1), gray[-edge:, :].reshape(-1),
        gray[:, :edge].reshape(-1), gray[:, -edge:].reshape(-1),
    ])
    med = float(np.median(samples))
    # Border removal is only attempted for very dark/bright edges or low-variance edges.
    if not (med <= 25 or med >= 230 or float(np.std(samples)) < 8):
        return img
    diff = np.abs(gray.astype(np.float32) - med)
    mask = diff > 18
    ys, xs = np.where(mask)
    if len(xs) < 0.20 * h * w:
        return img
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    if (x1 - x0) < 0.45 * w or (y1 - y0) < 0.45 * h:
        return img
    pad = max(2, int(min(h, w) * 0.01))
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(w, x1 + pad), min(h, y1 + pad)
    return img[y0:y1, x0:x1]


def _center_crop(img: np.ndarray, frac: float) -> np.ndarray:
    h, w = img.shape[:2]
    nw, nh = max(16, int(w * frac)), max(16, int(h * frac))
    x0 = max(0, (w - nw) // 2)
    y0 = max(0, (h - nh) // 2)
    return img[y0:y0 + nh, x0:x0 + nw]


def _circle_masked(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    size = min(h, w)
    crop = _center_crop(img, min(1.0, size / max(h, w))) if h != w else img.copy()
    crop = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_AREA)
    mask = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(mask, (128, 128), 122, 255, -1)
    bg = np.full_like(crop, int(np.median(crop)))
    return np.where(mask[..., None] > 0, crop, bg)


def signature_views(img: np.ndarray):
    """Multi-scale structural views for crop/border/circle-avatar recall."""
    trimmed = _trim_uniform_border(img)
    views = [("full", img), ("trim", trimmed)]
    for frac in (0.90, 0.80, 0.70, 0.60):
        views.append((f"center{int(frac * 100)}", _center_crop(trimmed, frac)))
    views.append(("circle", _circle_masked(trimmed)))
    # Deduplicate near-identical sizes/kinds cheaply by hash later.
    return views


def build_signatures(img: np.ndarray):
    out = []
    seen = set()
    for kind, view in signature_views(img):
        p = phash64_img(view)
        d = dhash64_img(view)
        key = (p, d)
        if key in seen:
            continue
        seen.add(key)
        out.append({"kind": kind, "phash": p, "dhash": d, "bands": bands64(p)})
    return out


def _get_face_cascade():
    global _face_cascade
    if _face_cascade is not None:
        return _face_cascade
    with _face_lock:
        if _face_cascade is None:
            try:
                _face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            except Exception:
                _face_cascade = False
    return _face_cascade


def roi_phash(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    try:
        cascade = _get_face_cascade()
        if cascade is False or cascade.empty():
            faces = []
        else:
            h0, w0 = gray.shape[:2]
            scale = min(1.0, 720.0 / max(h0, w0))
            detect_gray = gray if scale == 1 else cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            faces = cascade.detectMultiScale(detect_gray, scaleFactor=1.12, minNeighbors=4, minSize=(40, 40))
            if len(faces) and scale < 1.0:
                inv = 1.0 / scale
                faces = [(int(x * inv), int(y * inv), int(w * inv), int(h * inv)) for x, y, w, h in faces]
    except Exception:
        faces = []
    if len(faces):
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        pad = int(0.18 * max(w, h))
        roi = img[max(0, y-pad):min(img.shape[0], y+h+pad), max(0, x-pad):min(img.shape[1], x+w+pad)]
    else:
        roi = _center_crop(_trim_uniform_border(img), 0.70)
    return phash64_img(roi)


def color_hist(img):
    hsv = cv2.cvtColor(cv2.resize(_trim_uniform_border(img), (192, 192)), cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256]).astype(np.float32).flatten()
    n = float(np.linalg.norm(hist))
    return hist / n if n > 1e-8 else hist


def _enhanced_gray(img: np.ndarray) -> np.ndarray:
    img = _trim_uniform_border(img)
    h, w = img.shape[:2]
    # Tiny avatars need upscaling before keypoint detection.
    min_side = min(h, w)
    if min_side < 320:
        scale = min(4.0, 320.0 / max(1, min_side))
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    elif max(h, w) > 1100:
        scale = 1100.0 / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _get_orb():
    orb = getattr(_orb_local, "orb", None)
    if orb is None:
        orb = cv2.ORB_create(nfeatures=700, fastThreshold=10)
        _orb_local.orb = orb
    return orb


def _get_sift():
    sift = getattr(_sift_local, "sift", None)
    if sift is None:
        sift = cv2.SIFT_create(nfeatures=900, contrastThreshold=0.025, edgeThreshold=12)
        _sift_local.sift = sift
    return sift


def _get_akaze():
    ak = getattr(_akaze_local, "akaze", None)
    if ak is None:
        ak = cv2.AKAZE_create(threshold=0.0008)
        _akaze_local.akaze = ak
    return ak


def _kp_array(kps):
    if not kps:
        return np.empty((0, 2), dtype=np.float32)
    return np.asarray([kp.pt for kp in kps], dtype=np.float32)


def local_features(img: np.ndarray):
    gray = _enhanced_gray(img)
    out = {}
    for name, detector in (("sift", _get_sift()), ("akaze", _get_akaze())):
        try:
            kps, desc = detector.detectAndCompute(gray, None)
        except Exception:
            kps, desc = [], None
        if desc is None:
            if name == "sift":
                desc = np.empty((0, 128), dtype=np.float32)
            else:
                desc = np.empty((0, 61), dtype=np.uint8)
        out[name] = {"kp": _kp_array(kps), "desc": desc}
    return out


def orb_descriptors(img):
    gray = _enhanced_gray(img)
    _, desc = _get_orb().detectAndCompute(gray, None)
    return np.empty((0, 32), dtype=np.uint8) if desc is None else desc.astype(np.uint8)


def pack_f32(arr):
    if arr is None:
        return None
    return np.asarray(arr, dtype=np.float32).tobytes()


def unpack_f32(blob):
    if not blob:
        return None
    return np.frombuffer(blob, dtype=np.float32)


def pack_orb(desc):
    if desc is None or not len(desc):
        return None, 0
    return np.asarray(desc, dtype=np.uint8).tobytes(), int(desc.shape[0])


def unpack_orb(blob, rows):
    if not blob or not rows:
        return np.empty((0, 32), dtype=np.uint8)
    return np.frombuffer(blob, dtype=np.uint8).reshape(int(rows), 32)


def pack_local_feature(feature: dict):
    _kp = feature.get("kp")
    kp = np.asarray(_kp if _kp is not None else np.empty((0, 2), dtype=np.float32), dtype=np.float32)
    desc = feature.get("desc")
    if desc is None:
        return None, None, 0, 0, ""
    desc = np.asarray(desc)
    return kp.tobytes(), desc.tobytes(), int(desc.shape[0]), int(desc.shape[1]) if desc.ndim == 2 else 0, str(desc.dtype)


def unpack_local_feature(kp_blob, desc_blob, rows, cols, dtype):
    rows, cols = int(rows or 0), int(cols or 0)
    if not rows or not cols or not desc_blob:
        return {"kp": np.empty((0, 2), dtype=np.float32), "desc": np.empty((0, max(cols, 1)), dtype=np.float32)}
    kp = np.frombuffer(kp_blob or b"", dtype=np.float32).reshape(rows, 2)
    desc = np.frombuffer(desc_blob, dtype=np.dtype(dtype or "float32")).reshape(rows, cols)
    return {"kp": kp, "desc": desc}



def image_quality(img: np.ndarray) -> dict:
    """Cheap quality/information metrics used to prevent low-information pHash false positives."""
    try:
        small = cv2.resize(img, (160, 160), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256]).reshape(-1).astype(np.float64)
        p = hist / max(1.0, hist.sum())
        nz = p[p > 1e-12]
        entropy = float(-(nz * np.log2(nz)).sum())
        edges = cv2.Canny(gray, 60, 160)
        edge_density = float((edges > 0).mean())
        std = float(gray.std())
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        # Low-information images (solid colours, very simple logos/default icons)
        # are dangerous for pHash because unrelated images can share the same hash.
        low_information = bool(
            (entropy < 2.5 and edge_density < 0.025)
            or (std < 16.0 and edge_density < 0.018)
        )
        blurry = bool(lap_var < 25.0)
        return {
            "entropy": entropy,
            "edge_density": edge_density,
            "std": std,
            "lap_var": lap_var,
            "low_information": low_information,
            "blurry": blurry,
        }
    except Exception:
        return {
            "entropy": 0.0, "edge_density": 0.0, "std": 0.0, "lap_var": 0.0,
            "low_information": False, "blurry": False,
        }

def extract_light_features(path, sha256=None, md5=None):
    if not sha256 or not md5:
        sha256, md5 = hash_file_pair(path)
    img = read_image(path)
    signatures = build_signatures(img)
    p = signatures[0]["phash"] if signatures else phash64_img(img)
    d = signatures[0]["dhash"] if signatures else dhash64_img(img)
    r = roi_phash(img)
    f = {
        "sha256": sha256, "md5": md5, "phash": p, "dhash": d, "roi_phash": r,
        "p_bands": bands64(p), "signatures": signatures, "color_hist": color_hist(img),
        "orb": None, "ai_feature": None, "ai_hash": "", "a_bands": [""] * 8,
        "copy_views": None, "quality": image_quality(img),
        "local": None,
    }
    return f, img


def enrich_deep_features(features, img, include_local=False):
    if features.get("orb") is None:
        features["orb"] = orb_descriptors(img)

    # V1.9: task-specific SSCD runs first. The older ImageNet classifier
    # embedding is only a fallback when SSCD is unavailable, reducing CPU and
    # avoiding two neural-network passes for every photo.
    if features.get("copy_views") is None:
        features["copy_views"] = extract_copy_features(_trim_uniform_border(img))

    if features.get("ai_feature") is None and not features.get("copy_views"):
        ai = extract_ai_feature(_trim_uniform_border(img))
        features["ai_feature"] = ai
        ah = simhash64(ai) if ai is not None else ""
        features["ai_hash"] = ah
        features["a_bands"] = bands64(ah) if ah else [""] * 8

    if include_local and features.get("local") is None:
        features["local"] = local_features(img)
    return features


def extract_features(path, include_local=False):
    sha256, md5 = hash_file_pair(path)
    f, img = extract_light_features(path, sha256, md5)
    return enrich_deep_features(f, img, include_local=include_local)


def prepared_gray(img: np.ndarray) -> np.ndarray:
    """Public helper: deterministic grayscale used by local feature geometry."""
    return _enhanced_gray(img)
