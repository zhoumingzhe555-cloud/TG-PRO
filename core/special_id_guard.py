from __future__ import annotations

"""Ultra-light special HeyID page guard layered on top of V1.9.9.

Only the known profile-card screenshots are special-cased:
- a prominent magenta/pink/orange circular badge near the upper/middle area;
- a mostly flat light OR dark page canvas;
- a readable literal ``HeyID: XXXXX`` row under the badge.

Ordinary images never run Tesseract. They only pay a very small OpenCV probe and
then continue through the original V1.9.9 matcher unchanged.
"""

import re
import shutil
from pathlib import Path

import cv2
import numpy as np

try:
    import pytesseract
except Exception:  # pragma: no cover
    pytesseract = None


def _resize_max(img: np.ndarray, max_side: int = 520) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    scale = min(1.0, float(max_side) / max(h, w))
    if scale < 1.0:
        return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), scale
    return img, 1.0


def _find_gradient_circle(img: np.ndarray):
    """Return (cx, cy, radius) for the characteristic gradient badge."""
    work, scale = _resize_max(img, 520)
    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
    h, w = work.shape[:2]

    sat = hsv[:, :, 1]
    hue = hsv[:, :, 0]
    # The badge spans magenta/pink through orange/red.
    target_hue = (hue < 24) | (hue > 132)
    mask = np.uint8((sat > 78) & target_hue) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area <= 0:
            continue
        area_ratio = area / max(1.0, float(h * w))
        # Wide range is intentional: some users send a full page, others a tight crop
        # where the badge occupies > 1/3 of the image.
        if area_ratio < 0.020 or area_ratio > 0.46:
            continue
        x, y, cw, ch = cv2.boundingRect(cnt)
        aspect = cw / max(1.0, float(ch))
        if not 0.70 <= aspect <= 1.42:
            continue
        peri = float(cv2.arcLength(cnt, True))
        circularity = 4.0 * np.pi * area / max(1.0, peri * peri)
        if circularity < 0.47:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(cnt)
        nx = cx / max(1.0, w)
        ny = cy / max(1.0, h)
        if not (0.18 <= nx <= 0.82 and 0.04 <= ny <= 0.68):
            continue
        rank = area_ratio * 2.4 + circularity * 0.8 - abs(nx - 0.5) * 0.25
        if best is None or rank > best[0]:
            best = (rank, cx, cy, r)

    if best is None:
        return None
    inv = 1.0 / scale
    return float(best[1] * inv), float(best[2] * inv), float(best[3] * inv)


def _flat_page_score(img: np.ndarray, circle) -> bool:
    """Cheaply reject ordinary photographs before OCR.

    We deliberately ignore the colorful badge itself and inspect the remaining
    canvas.  This supports both white-theme and black-theme HeyID pages.
    """
    work, scale = _resize_max(img, 360)
    h, w = work.shape[:2]
    cx, cy, r = circle
    cx *= scale
    cy *= scale
    r *= scale

    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
    yy, xx = np.ogrid[:h, :w]
    outside = ((xx - cx) ** 2 + (yy - cy) ** 2) > (r * 1.08) ** 2
    if not np.any(outside):
        return False

    g = gray[outside]
    s = hsv[:, :, 1][outside]
    light = float(np.mean(g > 200))
    dark = float(np.mean(g < 55))
    low_sat = float(np.mean(s < 85))

    # Edge density is a strong photo-vs-flat-UI discriminator and is cheap.
    edges = cv2.Canny(gray, 70, 160)
    edge_density = float(np.mean(edges[outside] > 0))

    # White page: usually very low saturation and bright. Dark theme: HSV
    # saturation can be noisy in near-black pixels, so darkness+low edge density
    # is enough.
    light_page = light >= 0.45 and low_sat >= 0.52
    dark_page = dark >= 0.45 and edge_density <= 0.16
    mixed_flat = max(light, dark) >= 0.34 and edge_density <= 0.10
    return bool(light_page or dark_page or mixed_flat)


def _looks_like_special_page(img: np.ndarray):
    if img is None or img.size == 0 or min(img.shape[:2]) < 80:
        return None
    circle = _find_gradient_circle(img)
    if circle is None:
        return None
    if not _flat_page_score(img, circle):
        return None
    return circle


def normalize_external_id(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (value or "").upper())


def _parse_heyid(text: str) -> str:
    if not text:
        return ""
    up = text.upper()
    # Prefix OCR is allowed to confuse E/3 and I/1/L. The ID payload itself is
    # kept as read except for punctuation cleanup; we do not silently rewrite IDs.
    patterns = (
        # Normal / lightly corrupted HeyID prefix.
        r"H\s*[E3]\s*[YV]\s*[I1L|]\s*D\s*[:：]?\s*([A-Z0-9]{6,24})",
        r"H[E3][YV][I1L|]D\s*[:：]?\s*([A-Z0-9]{6,24})",
        # Reference pages consistently use a colon before a 3-letter + 8-digit ID.
        # This fallback is only reached after the special-page visual gate.
        r"[:：]\s*([A-Z]{3}[0-9]{8})\b",
        r"\b([A-Z]{3}[0-9]{8})\b",
    )
    for pat in patterns:
        m = re.search(pat, up, flags=re.IGNORECASE)
        if m:
            value = normalize_external_id(m.group(1))
            if 6 <= len(value) <= 24:
                return value
    return ""


def extract_heyid(img: np.ndarray, circle) -> str:
    """OCR only the narrow HeyID row; ordinary images never call this."""
    if pytesseract is None or not shutil.which("tesseract") or circle is None:
        return ""
    h, w = img.shape[:2]
    cx, cy, r = circle

    # Covers both supplied references:
    # - tight black crop: HeyID begins ~0.6 radius below badge bottom;
    # - white full card: HeyID sits directly below badge.
    y0 = max(0, int(cy + r * 0.95))
    y1 = min(h, int(cy + r * 1.60))
    x0 = 0
    x1 = w
    crop = img[y0:y1, x0:x1]
    if crop.size == 0 or min(crop.shape[:2]) < 12:
        return ""

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    scale = max(1.0, min(3.0, 850.0 / max(gray.shape[:2])))
    if scale > 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    # One OCR call in the normal case. A second layout mode is only a failure
    # fallback, never an always-on cost.
    try:
        text = pytesseract.image_to_string(gray, lang="eng", config="--psm 6")
    except Exception:
        text = ""
    value = _parse_heyid(text)
    if value:
        return value

    try:
        text = pytesseract.image_to_string(gray, lang="eng", config="--psm 11")
    except Exception:
        return ""
    return _parse_heyid(text)


def analyze_special_page(img: np.ndarray, run_ocr: bool = True) -> dict:
    circle = _looks_like_special_page(img)
    if circle is None:
        return {"is_special": False, "external_id": "", "circle": None}
    external_id = extract_heyid(img, circle) if run_ocr else ""
    # Authoritative special mode activates only after an actual HeyID is read.
    # If OCR fails, the original V1.9.9 image matcher remains in charge.
    return {
        "is_special": bool(external_id),
        "external_id": external_id,
        "circle": circle,
        "visual_candidate": True,
    }


def analyze_special_path(path: str | Path, run_ocr: bool = True) -> dict:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return {"is_special": False, "external_id": "", "circle": None}
    return analyze_special_page(img, run_ocr=run_ocr)
