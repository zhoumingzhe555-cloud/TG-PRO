from __future__ import annotations

"""Anti-template false-positive guard for social/profile-page screenshots.

Many dating/social apps use the same large white/grey page, gradient avatar circle,
buttons and text layout for every customer. Local features can therefore align the
*template* instead of the actual customer picture.  This module detects page-like
screenshots and, when a prominent circular avatar exists in both images, compares
the high-frequency content *inside* the avatar rather than the surrounding UI.

The guard is deliberately conservative: it can veto a weak/suspect match, but it
never creates a collision on its own. Exact file hashes are handled before this
module is called.
"""

import cv2
import numpy as np


def _resize_max(img: np.ndarray, max_side: int = 720) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    scale = min(1.0, float(max_side) / max(h, w))
    if scale < 1.0:
        return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), scale
    return img, 1.0


def _page_metrics(img: np.ndarray) -> dict:
    small, _ = _resize_max(img, 360)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    low_sat = float(np.mean(hsv[:, :, 1] < 70))
    near_light = float(np.mean(gray > 215))
    near_dark = float(np.mean(gray < 35))
    edges = cv2.Canny(gray, 70, 170)
    edge_density = float(np.mean(edges > 0))
    # Typical profile pages have a large low-saturation/uniform canvas with a
    # small amount of foreground content.  Do not require white specifically:
    # dark-theme profile pages are also possible.
    uniform_canvas = max(near_light, near_dark)
    page_like = bool(
        (low_sat >= 0.52 and uniform_canvas >= 0.28 and edge_density <= 0.18)
        or (low_sat >= 0.70 and edge_density <= 0.13)
    )
    return {
        "low_sat": low_sat,
        "uniform_canvas": uniform_canvas,
        "edge_density": edge_density,
        "page_like": page_like,
    }


def _hough_circle(img: np.ndarray):
    work, scale = _resize_max(img, 720)
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 1.4)
    h, w = gray.shape[:2]
    mn = max(14, int(min(h, w) * 0.07))
    mx = max(mn + 4, int(min(h, w) * 0.48))
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(28, mn * 2),
        param1=105,
        param2=24,
        minRadius=mn,
        maxRadius=mx,
    )
    if circles is None:
        return None
    inv = 1.0 / scale
    ih, iw = img.shape[:2]
    candidates = []
    for x, y, r in circles[0]:
        x, y, r = float(x * inv), float(y * inv), float(r * inv)
        # Ignore tiny decorative icons and circles mostly outside the page.
        area_ratio = np.pi * r * r / max(1.0, ih * iw)
        if area_ratio < 0.018 or area_ratio > 0.72:
            continue
        if x - r < -0.05 * iw or x + r > 1.05 * iw or y - r < -0.05 * ih or y + r > 1.05 * ih:
            continue
        # Prefer a large, reasonably central circle; this matches profile/avatar
        # pages while avoiding tiny Telegram reaction/status icons.
        center_penalty = abs(x / max(1.0, iw) - 0.5) + 0.55 * abs(y / max(1.0, ih) - 0.42)
        rank = area_ratio * 4.0 - center_penalty * 0.20
        candidates.append((rank, x, y, r))
    if not candidates:
        return None
    _, x, y, r = max(candidates, key=lambda z: z[0])
    return x, y, r


def _saturation_circle(img: np.ndarray):
    """Fallback for filled/gradient circles when Hough misses the boundary."""
    work, scale = _resize_max(img, 720)
    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    # Social-app gradient avatars are often much more saturated than the canvas.
    mask = np.uint8(sat > 85) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = work.shape[:2]
    best = None
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < 0.015 * h * w or area > 0.72 * h * w:
            continue
        peri = float(cv2.arcLength(cnt, True))
        if peri <= 1:
            continue
        circularity = 4.0 * np.pi * area / (peri * peri)
        x, y, cw, ch = cv2.boundingRect(cnt)
        aspect = cw / max(1.0, float(ch))
        if not (0.72 <= aspect <= 1.38 and circularity >= 0.45):
            continue
        (cx, cy), r = cv2.minEnclosingCircle(cnt)
        center_penalty = abs(cx / max(1.0, w) - 0.5) + 0.55 * abs(cy / max(1.0, h) - 0.42)
        rank = area / (h * w) * 4.0 + circularity * 0.35 - center_penalty * 0.20
        if best is None or rank > best[0]:
            best = (rank, cx, cy, r)
    if best is None:
        return None
    inv = 1.0 / scale
    return float(best[1] * inv), float(best[2] * inv), float(best[3] * inv)


def find_avatar_circle(img: np.ndarray):
    # Filled/gradient avatar circles are best recovered from saturation first;
    # Hough can be distracted by text baselines or large rounded UI panels.
    return _saturation_circle(img) or _hough_circle(img)


def _circle_patch(img: np.ndarray, circle, size: int = 192) -> np.ndarray | None:
    if circle is None:
        return None
    x, y, r = circle
    h, w = img.shape[:2]
    pad = 1.03
    x0 = max(0, int(round(x - r * pad)))
    y0 = max(0, int(round(y - r * pad)))
    x1 = min(w, int(round(x + r * pad)))
    y1 = min(h, int(round(y + r * pad)))
    if x1 - x0 < 24 or y1 - y0 < 24:
        return None
    crop = img[y0:y1, x0:x1]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA if max(crop.shape[:2]) > size else cv2.INTER_CUBIC)


def _highpass(gray: np.ndarray) -> np.ndarray:
    g = gray.astype(np.float32)
    blur = cv2.GaussianBlur(g, (0, 0), 5.0)
    hp = g - blur
    hp = cv2.normalize(hp, None, 0, 255, cv2.NORM_MINMAX)
    return hp.astype(np.uint8)


def _phash_bits(gray: np.ndarray) -> np.ndarray:
    x = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    d = cv2.dct(x)[:8, :8].reshape(-1)
    vals = d[1:]  # remove DC term
    med = float(np.median(vals))
    return vals > med


def _phash_score(a: np.ndarray, b: np.ndarray) -> float:
    ba, bb = _phash_bits(a), _phash_bits(b)
    return 100.0 * (1.0 - float(np.mean(ba != bb)))


def _edge_f1(a: np.ndarray, b: np.ndarray) -> float:
    ea = cv2.Canny(a, 55, 150)
    eb = cv2.Canny(b, 55, 150)
    # Ignore the shared outer circle boundary: the customer-specific signal is
    # inside the avatar.  Keep only the inner 72% radius.
    n = ea.shape[0]
    yy, xx = np.ogrid[:n, :n]
    rr = n * 0.36
    mask = ((xx - n / 2) ** 2 + (yy - n / 2) ** 2) <= rr * rr
    ea = (ea > 0) & mask
    eb = (eb > 0) & mask
    ca, cb = int(ea.sum()), int(eb.sum())
    if ca < 8 and cb < 8:
        return 100.0
    if ca < 5 or cb < 5:
        return 0.0
    k = np.ones((3, 3), np.uint8)
    da = cv2.dilate(ea.astype(np.uint8), k, iterations=1) > 0
    db = cv2.dilate(eb.astype(np.uint8), k, iterations=1) > 0
    pa = float(np.sum(ea & db)) / max(1, ca)
    pb = float(np.sum(eb & da)) / max(1, cb)
    if pa + pb <= 1e-9:
        return 0.0
    return 100.0 * (2.0 * pa * pb / (pa + pb))


def _corr_score(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.astype(np.float32).reshape(-1)
    bb = b.astype(np.float32).reshape(-1)
    aa -= aa.mean(); bb -= bb.mean()
    na = float(np.linalg.norm(aa)); nb = float(np.linalg.norm(bb))
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    corr = float(np.dot(aa, bb) / (na * nb))
    return max(0.0, min(100.0, (corr + 1.0) * 50.0))


def avatar_content_score(a_img: np.ndarray, b_img: np.ndarray, a_circle=None, b_circle=None) -> float:
    a_circle = a_circle or find_avatar_circle(a_img)
    b_circle = b_circle or find_avatar_circle(b_img)
    pa = _circle_patch(a_img, a_circle)
    pb = _circle_patch(b_img, b_circle)
    if pa is None or pb is None:
        return 0.0

    # Compare both normal and mirrored versions. Mirrored copies are legitimate
    # edited copies in this project and should not be vetoed by the guard.
    scores = []
    for candidate in (pb, cv2.flip(pb, 1)):
        ga = cv2.cvtColor(pa, cv2.COLOR_BGR2GRAY)
        gb = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY)
        ha, hb = _highpass(ga), _highpass(gb)
        p = _phash_score(ha, hb)
        e = _edge_f1(ha, hb)
        c = _corr_score(ha, hb)
        scores.append(0.40 * p + 0.42 * e + 0.18 * c)
    return float(max(scores))


def analyze_template(img: np.ndarray) -> dict:
    metrics = _page_metrics(img)
    sat_circle = _saturation_circle(img)
    circle = sat_circle or _hough_circle(img)
    return {
        **metrics,
        "circle": circle,
        "has_circle": circle is not None,
        # Saturation-connected-component detection is much more reliable for
        # filled/gradient profile avatars than Hough on UI screenshots. Hard
        # vetoes require reliable circles on both sides; Hough-only detections
        # can still contribute a caution signal but never suppress a match alone.
        "circle_reliable": sat_circle is not None,
    }


def compare_template(query_img: np.ndarray, cand_img: np.ndarray, query_profile: dict | None = None) -> dict:
    q = query_profile or analyze_template(query_img)
    c = analyze_template(cand_img)
    both_page = bool(q.get("page_like") and c.get("page_like"))
    both_circle = bool(q.get("has_circle") and c.get("has_circle"))
    avatar_score = 0.0
    if both_circle:
        avatar_score = avatar_content_score(query_img, cand_img, q.get("circle"), c.get("circle"))

    # Hard veto is intentionally narrow: only a page-template pair with two
    # prominent avatar circles whose *interior content* is clearly different.
    # Shared gradient circles/white background/text layout then cannot create a
    # false 'same photo' suspect. True same-avatar copies remain high here.
    reliable_pair = bool(q.get("circle_reliable") and c.get("circle_reliable"))
    veto = bool(both_page and both_circle and reliable_pair and avatar_score < 56.0)
    caution = bool(both_page and both_circle and avatar_score < 68.0)
    return {
        "template_like": both_page,
        "both_circle": both_circle,
        "avatar_content_score": round(float(avatar_score), 2),
        "template_veto": veto,
        "template_caution": caution,
        "circle_reliable_pair": reliable_pair,
        "query_page_like": bool(q.get("page_like")),
        "candidate_page_like": bool(c.get("page_like")),
    }
