"""Railway Volume backed image storage.

The database stores stable relative keys such as ``images/abc.jpg``.
The actual files live under DATA_DIR (normally /data on Railway), so they survive
redeploys when a Railway Volume is mounted at /data.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from config import DATA_DIR


def _safe_target(object_key: str) -> Path:
    key = str(object_key or "").replace("\\", "/").lstrip("/")
    if not key or ".." in Path(key).parts:
        raise ValueError("invalid object key")
    root = DATA_DIR.resolve()
    target = (DATA_DIR / key).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError("object key escapes DATA_DIR") from exc
    return target


def upload_image(local_path: Path, object_key: str) -> str:
    src = Path(local_path)
    if not src.exists():
        raise FileNotFoundError(src)
    dst = _safe_target(object_key)
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        same = src.resolve() == dst.resolve()
    except Exception:
        same = False
    if not same:
        tmp = dst.with_name(dst.name + ".tmp")
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
    return object_key


def download_image(object_key: str, local_path: Path) -> None:
    src = _safe_target(object_key)
    if not src.exists():
        raise FileNotFoundError(src)
    dst = Path(local_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def image_exists(object_key: str) -> bool:
    try:
        return _safe_target(object_key).is_file()
    except Exception:
        return False


def image_object_key(file_unique_id: str) -> str:
    safe = "".join(c for c in str(file_unique_id or "unknown") if c.isalnum() or c in "_-.")
    return f"images/{safe or 'unknown'}.jpg"


def history_object_key(sha256: str) -> str:
    safe = "".join(c for c in str(sha256 or "unknown") if c.isalnum())[:80]
    return f"images/hist_{safe or 'unknown'}.jpg"


def is_object_key(path_or_key: str) -> bool:
    s = str(path_or_key or "").replace("\\", "/")
    return s.startswith("images/") and not s.startswith("/")


def object_key_path(object_key: str) -> Path:
    """Resolve a stored key to its real Railway Volume path."""
    return _safe_target(object_key)
