from __future__ import annotations

"""Fast special-page ID guard for one narrow social-profile screenshot type.

Design goal:
- Ordinary photos keep the V1.9.9 pipeline unchanged and pay only a tiny OpenCV
  template probe (no OCR).
- Only screenshots that look like the known "gradient circle + HeyID" profile
  page run Tesseract.
- For this special page type, HeyID is authoritative: same ID => collision;
  different/no matched ID => do not let shared UI/template visuals create a hit.
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
    """Find the prominent pink/orange gradient circle used by this page type."""
    work, scale = _resize_max(img, 520)
    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
    h, w = work.shape[:2]

    # High saturation is the strongest cheap cue. The target UI uses a large
    # magenta/pink/orange circular badge on an otherwise low-saturation canvas.
    sat = hsv[:, :, 1]
    hue = hsv[:, :, 0]
    target_hue = (hue < 22) | (hue > 135)
    mask = np.uint8((sat > 85) & target_hue) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area <= 0:
            continue
        area_ratio = area / max(1.0, float(h * w))
        # The known badge is prominent but not full-screen.
        if area_ratio < 0.025 or area_ratio > 0.30:
            continue
        x, y, cw, ch = cv2.boundingRect(cnt)
        aspect = cw / max(1.0, float(ch))
        if not 0.72 <= aspect <= 1.38:
            continue
        peri = float(cv2.arcLength(cnt, True))
        circularity = 4.0 * np.pi * area / max(1.0, peri * peri)
        if circularity < 0.50:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(cnt)
        # The badge is approximately centered horizontally and in the upper/mid area.
        nx = cx / max(1.0, w)
        ny = cy / max(1.0, h)
        if not (0.25 <= nx <= 0.75 and 0.08 <= ny <= 0.62):
            continue
        rank = area_ratio * 3.0 + circularity * 0.7 - abs(nx - 0.5) * 0.4
        if best is None or rank > best[0]:
            best = (rank, cx, cy, r)

    if best is None:
        return None
    inv = 1.0 / scale
    return float(best[1] * inv), float(best[2] * inv), float(best[3] * inv)


def _looks_like_special_page(img: np.ndarray):
    if img is None or img.size == 0 or min(img.shape[:2]) < 80:
        return None
    work, _ = _resize_max(img, 360)
    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)

    low_sat = float(np.mean(hsv[:, :, 1] < 70))
    light = float(np.mean(gray > 205))
    dark = float(np.mean(gray < 48))
    # Supports both light and dark app themes, but requires a large uniform,
    # low-saturation canvas so ordinary photos do not trigger OCR.
    page_like = bool(low_sat >= 0.62 and max(light, dark) >= 0.34)
    if not page_like:
        return None
    return _find_gradient_circle(img)


def normalize_external_id(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (value or "").upper())


def _parse_heyid(text: str) -> str:
    if not text:
        return ""
    up = text.upper()
    # Be tolerant only in the literal prefix. The captured customer ID itself
    # is left unchanged except for alphanumeric cleanup.
    patterns = (
        r"H\s*[E3]\s*Y\s*[I1L|]\s*D\s*[:：]?\s*([A-Z0-9]{6,24})",
        r"H[E3]Y[I1L|]D\s*[:：]?\s*([A-Z0-9]{6,24})",
    )
    for pat in patterns:
        m = re.search(pat, up, flags=re.IGNORECASE)
        if m:
            value = normalize_external_id(m.group(1))
            if 6 <= len(value) <= 24:
                return value
    return ""


def extract_heyid(img: np.ndarray, circle) -> str:
    """OCR only the narrow HeyID row. Called only after the fast page probe."""
    if pytesseract is None or not shutil.which("tesseract") or circle is None:
        return ""
    h, w = img.shape[:2]
    cx, cy, r = circle

    # The HeyID row sits just below the badge. Keep the crop narrow to avoid
    # wasting OCR on the rest of the screenshot.
    y0 = max(0, int(cy + r * 0.72))
    y1 = min(h, int(cy + r * 2.10))
    x0 = max(0, int(w * 0.12))
    x1 = min(w, int(w * 0.88))
    crop = img[y0:y1, x0:x1]
    if crop.size == 0 or min(crop.shape[:2]) < 12:
        return ""

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    scale = max(1.0, min(3.2, 900.0 / max(gray.shape[:2])))
    if scale > 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    # One OCR call is normally enough and keeps this special path fast.
    try:
        text = pytesseract.image_to_string(gray, lang="eng", config="--psm 6")
    except Exception:
        text = ""
    value = _parse_heyid(text)
    if value:
        return value

    # Fallback only when the first narrow pass failed.
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
    # A page is considered authoritative-special only when the literal HeyID row
    # was successfully read. If OCR fails, V1.9.9's original image matcher remains
    # in charge rather than changing behavior for other images.
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
