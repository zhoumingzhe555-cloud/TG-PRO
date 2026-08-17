import shutil
import cv2
import numpy as np
import pytesseract
from config import OCR_LANGS


def _preprocess(gray: np.ndarray) -> np.ndarray:
    """多阶段预处理，提升低对比度/小字资料卡识别率。"""
    h, w = gray.shape[:2]
    if max(h, w) < 1400:
        scale = 1400.0 / max(h, w)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 10
    )


def likely_contains_text(image_path) -> bool:
    """
    OCR 快速门控：普通人物/生活照片通常不需要跑 Tesseract。
    这里只用低分辨率 OpenCV 运算判断“像不像资料卡/截图”，耗时远低于 OCR。
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return False
    h, w = img.shape[:2]
    scale = min(1.0, 720.0 / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]

    # 截图/资料卡往往有大面积低饱和背景，并包含密集细边缘文字。
    low_sat_ratio = float(np.mean(sat < 85))
    edges = cv2.Canny(gray, 80, 180)
    edge_ratio = float(np.mean(edges > 0))

    # 再统计小型连通块，中文/英文字符会形成较多这种组件。
    bw = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 12
    )
    n, _, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    textish = 0
    area_total = gray.shape[0] * gray.shape[1]
    for x, y, cw, ch, area in stats[1:]:
        if 3 <= ch <= 60 and 2 <= cw <= 160 and 8 <= area <= max(2500, area_total * 0.01):
            aspect = cw / max(ch, 1)
            if 0.08 <= aspect <= 12:
                textish += 1
                if textish >= 28:
                    break

    return (
        (low_sat_ratio >= 0.42 and edge_ratio >= 0.012 and textish >= 14)
        or textish >= 28
    )


def extract_text(image_path) -> str:
    if not shutil.which("tesseract"):
        return ""
    img = cv2.imread(str(image_path))
    if img is None:
        return ""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    processed = _preprocess(gray)

    cfg = "--psm 6"
    try:
        text = pytesseract.image_to_string(processed, lang=OCR_LANGS, config=cfg).strip()
        if text:
            return text
        return pytesseract.image_to_string(gray, lang=OCR_LANGS, config=cfg).strip()
    except Exception:
        try:
            return pytesseract.image_to_string(processed, lang="eng", config=cfg).strip()
        except Exception:
            return ""
