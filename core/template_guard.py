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
import re
import shutil
from difflib import SequenceMatcher

try:
    import pytesseract
except Exception:  # optional at import-time; Docker installs it
    pytesseract = None


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



def _normalize_account_id(value: str) -> str:
    """Normalize HeyID OCR with conservative position-aware glyph repair.

    Observed HeyID pages use a 3-letter prefix plus an 8-digit suffix. OCR often
    confuses S/5, O/0, I/1 or B/8 in the numeric suffix. Repair only when the
    whole token has the expected 11-character shape; otherwise leave it as-is.
    """
    raw=re.sub(r"[^A-Z0-9]", "", (value or "").upper())
    if len(raw)==11:
        pfx=list(raw[:3]); sfx=list(raw[3:])
        pmap={"0":"O","1":"I","5":"S","8":"B"}
        dmap={"O":"0","Q":"0","D":"0","I":"1","L":"1","Z":"2","S":"5","G":"6","B":"8"}
        pfx=[pmap.get(ch,ch) for ch in pfx]
        sfx=[dmap.get(ch,ch) for ch in sfx]
        cand="".join(pfx+sfx)
        if re.fullmatch(r"[A-Z]{3}[0-9]{8}",cand):
            return cand
    return raw


def _account_id_similarity(a: str, b: str) -> float:
    a = _normalize_account_id(a)
    b = _normalize_account_id(b)
    if not a or not b:
        return 0.0
    return float(SequenceMatcher(None, a, b).ratio())


# OCR on small grey account IDs can confuse visually similar glyphs.  These
# groups are used only to decide whether two OCR strings are *probably the same
# account*.  They never turn an ID into positive collision evidence by itself.
_ID_CONFUSION_GROUPS = (
    set("0ODQ"), set("1IL|"), set("2Z"), set("5S"), set("6G"), set("8B"),
)

def _id_char_equivalent(a: str, b: str) -> bool:
    if a == b:
        return True
    return any(a in g and b in g for g in _ID_CONFUSION_GROUPS)

def _account_id_same_confident(a: str, b: str) -> bool:
    """Conservative OCR-tolerant equality check for template account IDs.

    Same-ID is used only to *remove* the generic-placeholder veto.  It is never
    sufficient to declare a collision.  We therefore prefer false negatives
    here over allowing two public placeholder avatars to become a false match.
    """
    a = _normalize_account_id(a)
    b = _normalize_account_id(b)
    if len(a) < 7 or len(b) < 7:
        return False
    if a == b:
        return True
    # Most OCR mistakes keep length. Allow at most one visually-confusable
    # position or one ordinary mismatch on a long ID.
    if len(a) == len(b):
        hard = 0
        soft = 0
        for x, y in zip(a, b):
            if x == y:
                continue
            if _id_char_equivalent(x, y):
                soft += 1
            else:
                hard += 1
        return hard == 0 and soft <= 2 or (hard <= 1 and soft == 0 and len(a) >= 9)
    # One dropped/inserted OCR character is tolerated only when the rest aligns
    # almost perfectly.
    if abs(len(a) - len(b)) == 1 and SequenceMatcher(None, a, b).ratio() >= 0.92:
        return True
    return False


def _cluster_account_ids(values: list[str]) -> list[str]:
    """Collapse OCR variants of the same HeyID while preserving multiple cards.

    One screenshot may contain several profile cards. We keep one canonical ID
    per visual/OCR cluster instead of returning only the first ID.
    """
    vals=[_normalize_account_id(v) for v in values if 6 <= len(_normalize_account_id(v)) <= 24]
    clusters=[]
    for v in vals:
        placed=False
        for cl in clusters:
            if any(_account_id_same_confident(v,x) for x in cl):
                cl.append(v); placed=True; break
        if not placed:
            clusters.append([v])
    out=[]
    for cl in clusters:
        counts={x:cl.count(x) for x in set(cl)}
        # Prefer the most repeated OCR reading, then the longest. Stable lexical
        # tie-break keeps deterministic DB indexing.
        best=max(counts, key=lambda x:(counts[x],len(x),x))
        out.append(best)
    return sorted(set(out))


def extract_template_account_ids(img: np.ndarray, circle=None) -> list[str]:
    """Extract *all* stable account IDs such as ``HeyID: ABC123...``.

    V2.0.3: a Telegram image may itself contain two or more profile cards. The
    old code only OCR'd below one prominent circle and returned the first ID,
    which allowed template/UI similarity to win even when the real IDs were
    different. We OCR the whole page plus overlapping vertical regions and
    return every distinct HeyID.
    """
    if pytesseract is None or not shutil.which("tesseract"):
        return []
    h,w=img.shape[:2]
    if h < 40 or w < 40:
        return []

    regions=[img]
    # Overlapping vertical windows greatly improve small grey HeyID text and
    # multi-card screenshots without relying on one chosen avatar circle.
    if h >= 180:
        regions.extend([
            img[0:max(1,int(h*0.52)),:],
            img[max(0,int(h*0.32)):max(1,int(h*0.78)),:],
            img[max(0,int(h*0.56)):h,:],
        ])
    # Keep the old focused below-avatar crop as one extra high-resolution view.
    if circle is not None:
        _,cy,r=circle
        y0=max(0,int(cy+r*0.45)); y1=min(h,int(cy+r*3.0))
        if y1-y0 >= 30:
            regions.append(img[y0:y1,:])

    patterns=[
        r"H[e3]y\s*[Iil1|]\s*[Dd]\s*[:：]?\s*([A-Z0-9]{6,24})",
        r"H[e3]y[Iil1|][Dd]\s*[:：]?\s*([A-Z0-9]{6,24})",
    ]
    found=[]
    for reg in regions:
        if reg is None or reg.size == 0:
            continue
        gray=cv2.cvtColor(reg,cv2.COLOR_BGR2GRAY)
        scale=max(1.0,min(3.5,1400.0/max(gray.shape[:2])))
        if scale>1.0:
            gray=cv2.resize(gray,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
        gray=cv2.normalize(gray,None,0,255,cv2.NORM_MINMAX)
        # Two segmentation modes are enough for both single-card and grid/card
        # screenshots; avoid excessive OCR latency.
        for psm in (6,11):
            try:
                text=pytesseract.image_to_string(gray,lang="eng",config=f"--psm {psm}")
            except Exception:
                continue
            up=(text or "").upper()
            for pat in patterns:
                for m in re.finditer(pat,up,flags=re.IGNORECASE):
                    value=_normalize_account_id(m.group(1))
                    if 6 <= len(value) <= 24:
                        found.append(value)
    normalized=[_normalize_account_id(v) for v in found]
    strict=[v for v in normalized if re.fullmatch(r"[A-Z]{3}[0-9]{8}",v)]
    return _cluster_account_ids(strict if strict else normalized)


def extract_template_account_id(img: np.ndarray, circle=None) -> str:
    """Backward-compatible first-ID accessor."""
    ids=extract_template_account_ids(img,circle)
    return ids[0] if ids else ""


def account_ids_overlap(a_ids, b_ids) -> bool:
    """OCR-tolerant overlap between two sets of template account IDs."""
    aa=[_normalize_account_id(x) for x in (a_ids or []) if x]
    bb=[_normalize_account_id(x) for x in (b_ids or []) if x]
    return any(_account_id_same_confident(a,b) for a in aa for b in bb)


def normalize_account_id(value: str) -> str:
    return _normalize_account_id(value)

def account_id_same_confident(a: str, b: str) -> bool:
    return _account_id_same_confident(a,b)

def _avatar_information(img: np.ndarray, circle) -> float:
    """Return a rough 0..1 texture/content score for the avatar interior.

    Generic placeholders (single digit/letter on a smooth gradient) have very
    little interior structure and must not be treated as a unique customer
    photo merely because the placeholder itself is identical.
    """
    patch = _circle_patch(img, circle, size=192)
    if patch is None:
        return 1.0
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    n = gray.shape[0]
    yy, xx = np.ogrid[:n, :n]
    mask = ((xx - n / 2) ** 2 + (yy - n / 2) ** 2) <= (n * 0.34) ** 2
    edges = cv2.Canny(gray, 60, 160) > 0
    edge_density = float(np.sum(edges & mask)) / max(1, int(mask.sum()))
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    lap_std = float(np.std(lap[mask])) if np.any(mask) else 0.0
    # Conservative score: portraits/photos tend to exceed this comfortably,
    # while a single numeral on a smooth gradient remains low.
    return float(min(1.0, edge_density / 0.12 * 0.65 + lap_std / 45.0 * 0.35))

def analyze_template(img: np.ndarray) -> dict:
    metrics = _page_metrics(img)
    sat_circle = _saturation_circle(img)
    circle = sat_circle or _hough_circle(img)
    avatar_info = _avatar_information(img, circle) if circle is not None else 1.0
    # V2.0.3: ID extraction is allowed on page-like screenshots even if one
    # prominent circle cannot be found. Multi-card screenshots often contain
    # several smaller circles and are exactly where ID should be authoritative.
    account_ids=[]
    if metrics.get("page_like") or circle is not None:
        account_ids=extract_template_account_ids(img,circle)
    return {
        **metrics,
        "circle": circle,
        "has_circle": circle is not None,
        "avatar_information": round(float(avatar_info), 4),
        "generic_avatar": bool(circle is not None and avatar_info < 0.42),
        "account_ids": account_ids,
        "account_id": account_ids[0] if account_ids else "",
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

    reliable_pair = bool(q.get("circle_reliable") and c.get("circle_reliable"))
    qids=list(q.get("account_ids") or ([q.get("account_id")] if q.get("account_id") else []))
    cids=list(c.get("account_ids") or ([c.get("account_id")] if c.get("account_id") else []))
    id_primary_match=bool(qids and cids and account_ids_overlap(qids,cids))
    # If both screenshots expose one or more HeyIDs and no ID overlaps, the IDs
    # are authoritative for this template family. Shared UI / same placeholder
    # numeral is never allowed to override a different account ID.
    id_conflict=bool(qids and cids and not id_primary_match)
    both_generic=bool(q.get("generic_avatar") and c.get("generic_avatar"))

    avatar_veto=bool(both_page and both_circle and reliable_pair and avatar_score < 56.0 and not id_primary_match)
    id_veto=bool(id_conflict)
    generic_placeholder_veto=bool(
        both_page and both_circle and both_generic and not id_primary_match
    )
    veto=bool(avatar_veto or id_veto or generic_placeholder_veto)
    caution=bool(
        (both_page and both_circle and avatar_score < 68.0 and not id_primary_match)
        or (both_generic and not id_primary_match)
        or id_conflict
    )
    return {
        "template_like": both_page,
        "both_circle": both_circle,
        "avatar_content_score": round(float(avatar_score), 2),
        "template_veto": veto,
        "template_caution": caution,
        "circle_reliable_pair": reliable_pair,
        "query_page_like": bool(q.get("page_like")),
        "candidate_page_like": bool(c.get("page_like")),
        "query_account_ids": qids,
        "candidate_account_ids": cids,
        "query_account_id": qids[0] if qids else "",
        "candidate_account_id": cids[0] if cids else "",
        "account_id_primary_match": id_primary_match,
        "account_id_conflict": id_conflict,
        "generic_avatar_pair": both_generic,
        "generic_placeholder_veto": generic_placeholder_veto,
    }

