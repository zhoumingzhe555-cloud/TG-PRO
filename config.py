import os
from pathlib import Path

BOT_TOKEN: str = os.environ.get("BOT_TOKEN", "").strip()


def _data_dir() -> Path:
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

# ── Matching policy ───────────────────────────────────────────────────────────
# Scores >= 90 are reserved for strong same-photo evidence. Weak semantic/colour
# similarity is capped below this threshold so AUTO90 remains safe.
SIMILAR_THRESHOLD = float(os.getenv("SIMILAR_THRESHOLD", "84"))
AUTO_COLLISION_THRESHOLD = float(os.getenv("AUTO_COLLISION_THRESHOLD", "90"))
PHASH_DIRECT_THRESHOLD = float(os.getenv("PHASH_DIRECT_THRESHOLD", "96"))

# Candidate recall: all history is considered through cheap signatures; expensive
# geometry is only run on a small union of the strongest candidates.
FAST_CANDIDATE_LIMIT = int(os.getenv("FAST_CANDIDATE_LIMIT", "120"))
DEEP_CANDIDATE_LIMIT = int(os.getenv("DEEP_CANDIDATE_LIMIT", "32"))
GEOMETRY_CANDIDATE_LIMIT = int(os.getenv("GEOMETRY_CANDIDATE_LIMIT", "12"))
FULL_LIGHT_SCAN_LIMIT = int(os.getenv("FULL_LIGHT_SCAN_LIMIT", "100000"))
MAX_FALLBACK_CANDIDATES = int(os.getenv("MAX_FALLBACK_CANDIDATES", "100000"))  # backward-compatible name

IMAGE_CHECK_TIMEOUT = float(os.getenv("IMAGE_CHECK_TIMEOUT", "45"))
MAX_CONCURRENT_IMAGE_CHECKS = max(1, int(os.getenv("MAX_CONCURRENT_IMAGE_CHECKS", "2")))
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(30 * 1024 * 1024)))
MAX_PROCESS_MEGAPIXELS = float(os.getenv("MAX_PROCESS_MEGAPIXELS", "24"))

# Legacy multi-view hash reindex for databases upgraded from V1.7/V1.8.
REINDEX_BATCH_SIZE = max(1, int(os.getenv("REINDEX_BATCH_SIZE", "20")))
REINDEX_INTERVAL_SEC = max(15, int(os.getenv("REINDEX_INTERVAL_SEC", "60")))

# V1.9 SSCD copy-detection AI. This model is trained for edited-copy retrieval,
# not face identity recognition. It is stored on the Railway Volume so the ~99MB
# model is downloaded only once.
SSCD_ENABLED = os.getenv("SSCD_ENABLED", "1").lower() not in {"0", "false", "no"}
SSCD_MODEL_URL = os.getenv(
    "SSCD_MODEL_URL",
    "https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_mixup.torchscript.pt",
)
SSCD_MODEL_PATH = Path(os.getenv("SSCD_MODEL_PATH", str(MODEL_DIR / "sscd_disc_mixup.torchscript.pt")))
SSCD_INPUT_SIZE = max(224, int(os.getenv("SSCD_INPUT_SIZE", "320")))
SSCD_THREADS = max(1, int(os.getenv("SSCD_THREADS", "1")))
SSCD_BATCH_SIZE = max(1, int(os.getenv("SSCD_BATCH_SIZE", "4")))
SSCD_MAX_VIEWS = max(1, min(10, int(os.getenv("SSCD_MAX_VIEWS", "8"))))
SSCD_TOP_K = max(8, int(os.getenv("SSCD_TOP_K", "48")))
SSCD_FULL_SCAN_LIMIT = max(1000, int(os.getenv("SSCD_FULL_SCAN_LIMIT", "15000")))
SSCD_LSH_CANDIDATE_LIMIT = max(100, int(os.getenv("SSCD_LSH_CANDIDATE_LIMIT", "2500")))
SSCD_DOWNLOAD_TIMEOUT = max(10, int(os.getenv("SSCD_DOWNLOAD_TIMEOUT", "45")))
SSCD_MAX_MODEL_BYTES = max(100_000_000, int(os.getenv("SSCD_MAX_MODEL_BYTES", str(160 * 1024 * 1024))))
COPY_REINDEX_BATCH_SIZE = max(1, int(os.getenv("COPY_REINDEX_BATCH_SIZE", "6")))
COPY_REINDEX_INTERVAL_SEC = max(10, int(os.getenv("COPY_REINDEX_INTERVAL_SEC", "20")))

# Production safety: Railway must keep /data on a persistent Volume. If the
# Volume disappears, fail closed rather than silently starting a blank DB.
REQUIRE_PERSISTENT_STORAGE = os.getenv(
    "REQUIRE_PERSISTENT_STORAGE",
    "1" if os.getenv("RAILWAY_ENVIRONMENT") else "0",
).lower() not in {"0", "false", "no"}

# Local ONNX visual feature remains an auxiliary signal only.
AI_ENABLED = os.getenv("AI_ENABLED", "1").lower() not in {"0", "false", "no"}
AI_MODEL_URL = os.getenv(
    "AI_MODEL_URL",
    "https://github.com/onnx/models/raw/refs/heads/main/validated/vision/classification/mobilenet/model/mobilenetv2-10.onnx",
)
AI_MODEL_PATH = Path(os.getenv("AI_MODEL_PATH", str(MODEL_DIR / "mobilenetv2-10.onnx")))
AI_INTRA_THREADS = max(1, int(os.getenv("AI_INTRA_THREADS", "1")))

OCR_FALLBACK = os.getenv("OCR_FALLBACK", "1").lower() not in {"0", "false", "no"}
OCR_LANGS = os.getenv("OCR_LANGS", "chi_tra+chi_sim+eng")
IMPORT_ADMINS_ONLY = os.getenv("IMPORT_ADMINS_ONLY", "1").lower() not in {"0", "false", "no"}
ADMIN_CHAT_ID: str = os.getenv("ADMIN_CHAT_ID", "").strip()
