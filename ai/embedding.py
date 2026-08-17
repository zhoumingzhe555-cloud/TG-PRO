import logging
import os
import threading
import urllib.request
import numpy as np
import cv2
from config import AI_ENABLED, AI_MODEL_URL, AI_MODEL_PATH

log = logging.getLogger(__name__)
_session = None
_input_name = None
_failed = False
_load_lock = threading.Lock()
_project_cache = {}
_simhash_cache = {}


def _download_model():
    AI_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if AI_MODEL_PATH.exists() and AI_MODEL_PATH.stat().st_size > 1_000_000:
        return True
    try:
        log.info("首次启动：下载轻量 AI 视觉模型…")
        urllib.request.urlretrieve(AI_MODEL_URL, AI_MODEL_PATH)
        return AI_MODEL_PATH.exists() and AI_MODEL_PATH.stat().st_size > 1_000_000
    except Exception as e:
        log.warning("AI模型下载失败，将使用 pHash/ORB 备用匹配：%s", e)
        return False


def _load():
    global _session, _input_name, _failed
    if not AI_ENABLED or _failed:
        return False
    if _session is not None:
        return True
    with _load_lock:
        if _session is not None:
            return True
        if _failed:
            return False
        try:
            import onnxruntime as ort
            if not _download_model():
                _failed = True
                return False
            opts = ort.SessionOptions()
            # Railway CPU 环境，限制 ORT 自己再开大量线程，避免一次检测把进程拖死。
            opts.intra_op_num_threads = max(1, int(os.getenv("AI_INTRA_THREADS", "1")))
            opts.inter_op_num_threads = 1
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            _session = ort.InferenceSession(
                str(AI_MODEL_PATH),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            _input_name = _session.get_inputs()[0].name
            log.info("AI图片特征模型加载成功")
            return True
        except Exception as e:
            _failed = True
            log.warning("AI模型加载失败，将使用备用视觉特征：%s", e)
            return False


def preload():
    _load()


def _projection_matrix(in_dim: int, out_dim: int):
    key = (in_dim, out_dim)
    mat = _project_cache.get(key)
    if mat is None:
        rng = np.random.default_rng(20260816)
        mat = rng.standard_normal((out_dim, in_dim), dtype=np.float32) / np.sqrt(in_dim)
        _project_cache[key] = mat
    return mat


def _project(vec, out_dim=128):
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    if vec.size <= out_dim:
        out = np.pad(vec, (0, max(0, out_dim - vec.size)))[:out_dim]
    else:
        # 旧版每张图都会重新生成 128xN 随机矩阵，浪费大量 CPU；现在缓存一次复用。
        out = _projection_matrix(vec.size, out_dim) @ vec
    n = np.linalg.norm(out)
    if n > 1e-8:
        out = out / n
    return out.astype(np.float32)


def extract_ai_feature(bgr):
    if not _load():
        return None
    try:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        x = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        x = (x - mean) / std
        x = np.transpose(x, (2, 0, 1))[None, ...]
        outputs = _session.run(None, {_input_name: x})
        return _project(outputs[-1])
    except Exception as e:
        log.warning("AI特征提取失败：%s", e)
        return None


def cosine_score(a, b):
    if a is None or b is None:
        return 0.0
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    cos = float(np.dot(a, b) / (na * nb))
    return max(0.0, min(100.0, cos * 100.0))


def _simhash_planes(dim: int):
    hp = _simhash_cache.get(dim)
    if hp is None:
        rng = np.random.default_rng(5601)
        hp = rng.standard_normal((64, dim), dtype=np.float32)
        _simhash_cache[dim] = hp
    return hp


def simhash64(vec):
    if vec is None:
        return ""
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    bits = (_simhash_planes(vec.size) @ vec) >= 0
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:016x}"
