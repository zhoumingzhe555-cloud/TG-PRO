import os
from pathlib import Path

# GitHub -> Railway production configuration.
# Only BOT_TOKEN is required. Persist customer DB/images by mounting a Railway Volume at /data.
BOT_TOKEN: str = os.environ.get("BOT_TOKEN", "").strip()


def _data_dir() -> Path:
    # Dockerfile sets DATA_DIR=/data. This fallback also lets the project run locally.
    raw = os.environ.get("DATA_DIR", "").strip()
    if raw:
        return Path(raw)
    return Path("/data") if os.environ.get("RAILWAY_ENVIRONMENT") else Path("data")


DATA_DIR = _data_dir()
DATABASE_PATH = os.environ.get("DATABASE_PATH", str(DATA_DIR / "customers.db"))
IMAGE_DIR = Path(os.environ.get("IMAGE_DIR", str(DATA_DIR / "images")))
MODEL_DIR = Path(os.environ.get("MODEL_DIR", str(DATA_DIR / "models")))
IMPORT_DIR = Path(os.environ.get("IMPORT_DIR", str(DATA_DIR / "imports")))

for _p in (DATA_DIR, IMAGE_DIR, MODEL_DIR, IMPORT_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# 图片匹配阈值
SIMILAR_THRESHOLD = float(os.getenv("SIMILAR_THRESHOLD", "86"))
# 相似度达到此阈值时直接确认为撞客，不再要求人工按钮确认。
AUTO_COLLISION_THRESHOLD = float(os.getenv("AUTO_COLLISION_THRESHOLD", "90"))
PHASH_DIRECT_THRESHOLD = float(os.getenv("PHASH_DIRECT_THRESHOLD", "96"))
MAX_FALLBACK_CANDIDATES = int(os.getenv("MAX_FALLBACK_CANDIDATES", "500"))
FAST_CANDIDATE_LIMIT = int(os.getenv("FAST_CANDIDATE_LIMIT", "600"))
DEEP_CANDIDATE_LIMIT = int(os.getenv("DEEP_CANDIDATE_LIMIT", "24"))
IMAGE_CHECK_TIMEOUT = float(os.getenv("IMAGE_CHECK_TIMEOUT", "30"))
MAX_CONCURRENT_IMAGE_CHECKS = max(1, int(os.getenv("MAX_CONCURRENT_IMAGE_CHECKS", "2")))

# 本地 ONNX 视觉特征。模型下载失败时自动回退到 pHash + ORB + 色彩/主体特征。
AI_ENABLED = os.getenv("AI_ENABLED", "1").lower() not in {"0", "false", "no"}
AI_MODEL_URL = os.getenv(
    "AI_MODEL_URL",
    "https://github.com/onnx/models/raw/refs/heads/main/validated/vision/classification/mobilenet/model/mobilenetv2-10.onnx",
)
AI_MODEL_PATH = Path(os.getenv("AI_MODEL_PATH", str(MODEL_DIR / "mobilenetv2-10.onnx")))
AI_INTRA_THREADS = max(1, int(os.getenv("AI_INTRA_THREADS", "1")))

# 图片没有 caption/客户资料时，是否尝试 OCR。普通照片会先经过轻量文字门控，不会每张都跑 OCR。
OCR_FALLBACK = os.getenv("OCR_FALLBACK", "1").lower() not in {"0", "false", "no"}
OCR_LANGS = os.getenv("OCR_LANGS", "chi_tra+chi_sim+eng")

# 历史导入默认只允许当前 Telegram 群管理员。
IMPORT_ADMINS_ONLY = os.getenv("IMPORT_ADMINS_ONLY", "1").lower() not in {"0", "false", "no"}

# 可选：机器人崩溃时发送告警。留空不影响运行。
ADMIN_CHAT_ID: str = os.getenv("ADMIN_CHAT_ID", "").strip()
