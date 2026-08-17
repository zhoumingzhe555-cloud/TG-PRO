import hashlib
import threading
import cv2
import numpy as np
from ai.embedding import extract_ai_feature, simhash64

# Haar cascade 初始化本身有开销。以前每张图片都会重新加载 XML；
# 现在每个进程只初始化一次。
_face_cascade = None
_face_lock = threading.Lock()
_orb_local = threading.local()


def read_image(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("无法读取图片")
    return img


def hash_file(path, algo="sha256"):
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_file_pair(path):
    """一次读盘同时计算 SHA256 + MD5，避免旧版重复读取整张图片两次。"""
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


def _get_face_cascade():
    global _face_cascade
    if _face_cascade is not None:
        return _face_cascade
    with _face_lock:
        if _face_cascade is None:
            try:
                _face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                )
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
            # 在大图上做人脸检测很慢，先缩小到最长边 720，再映射回原图。
            h0, w0 = gray.shape[:2]
            scale = min(1.0, 720.0 / max(h0, w0))
            detect_gray = gray
            if scale < 1.0:
                detect_gray = cv2.resize(
                    gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
                )
            faces = cascade.detectMultiScale(
                detect_gray, scaleFactor=1.12, minNeighbors=4, minSize=(40, 40)
            )
            if len(faces) and scale < 1.0:
                inv = 1.0 / scale
                faces = [
                    (int(x * inv), int(y * inv), int(w * inv), int(h * inv))
                    for x, y, w, h in faces
                ]
    except Exception:
        faces = []

    if len(faces):
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        pad = int(0.18 * max(w, h))
        y0 = max(0, y - pad)
        y1 = min(img.shape[0], y + h + pad)
        x0 = max(0, x - pad)
        x1 = min(img.shape[1], x + w + pad)
        roi = img[y0:y1, x0:x1]
    else:
        h, w = img.shape[:2]
        y0 = int(h * 0.15)
        y1 = int(h * 0.85)
        x0 = int(w * 0.15)
        x1 = int(w * 0.85)
        roi = img[y0:y1, x0:x1]
    return phash64_img(roi)


def color_hist(img):
    hsv = cv2.cvtColor(cv2.resize(img, (192, 192)), cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256]).astype(np.float32).flatten()
    n = float(np.linalg.norm(hist))
    if n > 1e-8:
        hist /= n
    return hist


def _get_orb():
    orb = getattr(_orb_local, "orb", None)
    if orb is None:
        # 350 个关键点足够用于同图/截图/裁剪判断，比旧版 500 更快。
        orb = cv2.ORB_create(nfeatures=350, fastThreshold=14)
        _orb_local.orb = orb
    return orb


def orb_descriptors(img):
    h, w = img.shape[:2]
    scale = min(1.0, 760.0 / max(h, w))
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, desc = _get_orb().detectAndCompute(gray, None)
    if desc is None:
        return np.empty((0, 32), dtype=np.uint8)
    return desc.astype(np.uint8)


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
    return desc.tobytes(), int(desc.shape[0])


def unpack_orb(blob, rows):
    if not blob or not rows:
        return np.empty((0, 32), dtype=np.uint8)
    return np.frombuffer(blob, dtype=np.uint8).reshape(int(rows), 32)


def extract_light_features(path, sha256=None, md5=None):
    """快速层：文件指纹 + pHash/dHash/ROI + 色彩，不运行 ORB/AI。"""
    if not sha256 or not md5:
        sha256, md5 = hash_file_pair(path)
    img = read_image(path)
    p = phash64_img(img)
    d = dhash64_img(img)
    r = roi_phash(img)
    hist = color_hist(img)
    f = {
        "sha256": sha256,
        "md5": md5,
        "phash": p,
        "dhash": d,
        "roi_phash": r,
        "p_bands": bands64(p),
        "color_hist": hist,
        "orb": None,
        "ai_feature": None,
        "ai_hash": "",
        "a_bands": [""] * 8,
    }
    return f, img


def enrich_deep_features(features, img):
    """精确层：只在候选图片需要精确判断时运行 ORB + AI。"""
    if features.get("orb") is None:
        features["orb"] = orb_descriptors(img)
    if features.get("ai_feature") is None:
        ai = extract_ai_feature(img)
        features["ai_feature"] = ai
        ah = simhash64(ai) if ai is not None else ""
        features["ai_hash"] = ah
        features["a_bands"] = bands64(ah) if ah else [""] * 8
    return features


def extract_features(path):
    """完整特征。入库/历史导入仍可直接调用此函数。"""
    sha256, md5 = hash_file_pair(path)
    f, img = extract_light_features(path, sha256, md5)
    return enrich_deep_features(f, img)
