from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

from bs4 import BeautifulSoup

from core.customer import parse_customer_info, is_customer_record, public_customer_data
from core.image_match import save_customer_and_image, check_image
from core.object_storage import upload_image, history_object_key

log = logging.getLogger(__name__)

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MANIFEST_NAME = "history_manifest.json"
SELF_RECEIVER = {"自引", "自己", "本人", "self", "SELF"}


def _stats():
    return {
        "scanned": 0,
        "images": 0,
        "skipped": 0,
        "photo_refs": 0,
        "files_missing": 0,
        "profile_records": 0,
        "resolved_profiles": 0,
        "unresolved_profiles": 0,
        "query_photos": 0,
        "duplicates": 0,
    }


def _upload_image_safe(local_path: Path, sha256: str) -> str | None:
    try:
        return upload_image(local_path, history_object_key(sha256))
    except Exception:
        log.warning("保存历史图片失败: %s", local_path, exc_info=True)
        return None


def _safe_extract(zpath, dest):
    dest = Path(dest).resolve()
    with zipfile.ZipFile(zpath) as z:
        for m in z.infolist():
            target = (dest / m.filename).resolve()
            try:
                target.relative_to(dest)
            except ValueError:
                continue
            z.extract(m, dest)


def _flat_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        out = []
        for x in value:
            if isinstance(x, str):
                out.append(x)
            elif isinstance(x, dict):
                out.append(str(x.get("text", x.get("href", ""))))
        return "".join(out)
    return str(value or "")


def _parse_date(value):
    if not value:
        return None
    s = str(value).strip()
    for fmt in ("%d.%m.%Y %H:%M:%S UTC+08:00", "%d.%m.%Y %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            pass
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def _customer_data(info: dict, sender: str) -> dict:
    data = public_customer_data(info)
    receiver = str(data.get("receiver") or "").strip()
    if receiver in SELF_RECEIVER:
        # “自引”=实际接粉人为提交资料的人。原文字仍完整保存在 raw_text。
        data["receiver"] = sender or receiver
    return data


def _query_like(text: str) -> bool:
    t = re.sub(r"\s+", "", text or "").lower()
    if not t:
        return False
    words = (
        "有人聊", "有聊过", "有聊過", "有没有聊", "有沒有聊",
        "有人加", "有加过", "有加過", "有没有加", "有沒有加",
        "有见过", "有見過", "撞客", "撞的", "有人做", "有做过", "有做過",
        "这个tg头像", "這個tg頭像", "whatsapp头像", "whatsapp頭像",
    )
    return any(w in t for w in words)


def _html_message_id(block) -> str:
    return str(block.get("id") or "")


def _html_reply_id(block) -> str | None:
    a = block.select_one('div.reply_to a[href^="#go_to_message"]')
    if not a:
        return None
    m = re.search(r"go_to_message(\d+)", a.get("href", ""))
    return f"message{m.group(1)}" if m else None


def _html_photo_ref(block) -> str | None:
    """Telegram HTML 有两种常见照片结构：photo_wrap 与 media_photo。"""
    selectors = (
        "a.photo_wrap[href]",
        "a.media_photo[href]",
        'a[href$=".jpg"]', 'a[href$=".jpeg"]', 'a[href$=".png"]',
        'a[href$=".webp"]', 'a[href$=".bmp"]',
    )
    for sel in selectors:
        a = block.select_one(sel)
        if a:
            href = str(a.get("href") or "").strip()
            if href and not href.startswith(("http://", "https://", "#", "javascript:")):
                if Path(href).suffix.lower() in IMAGE_EXT and "_thumb." not in href:
                    return href
    return None


def _html_pages(paths: list[Path]):
    """一次解析所有 messages*.html，保证跨分页 reply_to 仍可关联。"""
    records = []
    chat_name = "history"
    last_sender = None

    def page_no(p: Path):
        m = re.search(r"messages(\d*)\.html$", p.name, flags=re.I)
        if not m:
            return 999999
        return int(m.group(1) or "1")

    for path in sorted(paths, key=page_no):
        soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
        title = soup.select_one(".page_header .text.bold")
        if title and title.get_text(" ", strip=True):
            chat_name = title.get_text(" ", strip=True)

        for block in soup.select("div.message"):
            classes = set(block.get("class") or [])
            if "service" in classes:
                last_sender = None
                records.append({
                    "id": _html_message_id(block), "service": True, "sender": "",
                    "text": "", "photo_ref": None, "reply_id": None,
                    "date": None, "joined": False, "root": path.parent,
                })
                continue

            from_el = block.select_one(".from_name")
            explicit_sender = from_el.get_text(" ", strip=True) if from_el else ""
            joined = "joined" in classes
            if explicit_sender:
                sender = explicit_sender
                last_sender = sender
            elif joined and last_sender:
                sender = last_sender
            else:
                sender = "history"

            text_el = block.select_one(".text")
            text = text_el.get_text("\n", strip=True) if text_el else ""
            date_el = block.select_one(".date.details")
            date = _parse_date(date_el.get("title") if date_el else None)
            rec = {
                "id": _html_message_id(block),
                "service": False,
                "sender": sender,
                "text": text,
                "photo_ref": _html_photo_ref(block),
                "reply_id": _html_reply_id(block),
                "date": date,
                "joined": joined,
                "root": path.parent,
            }
            records.append(rec)
    return records, chat_name


def _resolve_reply_photo(rec, index: dict[str, dict], max_depth=4):
    rid = rec.get("reply_id")
    seen = set()
    for _ in range(max_depth):
        if not rid or rid in seen:
            return None
        seen.add(rid)
        target = index.get(rid)
        if not target:
            return None
        if target.get("photo_ref"):
            return target
        rid = target.get("reply_id")
    return None


def _resolve_joined_photo(rec, records, pos):
    """无 reply_to 时，只做保守关联：同发送者、紧邻/同组、10 分钟内。"""
    if not rec.get("date"):
        return None
    i = pos.get(rec.get("id"), -1)
    if i < 0:
        return None
    sender = rec.get("sender")
    for j in range(i - 1, max(-1, i - 8), -1):
        prev = records[j]
        if prev.get("service"):
            break
        # 遇到其他发送者就不跨过去猜，避免把普通查询图配错客户资料。
        if prev.get("sender") != sender:
            if not prev.get("joined"):
                break
            continue
        if is_customer_record(parse_customer_info(prev.get("text", ""))):
            break
        if prev.get("photo_ref"):
            if prev.get("date"):
                diff = (rec["date"] - prev["date"]).total_seconds()
                if 0 <= diff <= 600:
                    return prev
            return None
    return None


def _resolve_photo_path(photo_rec: dict) -> Path | None:
    ref = photo_rec.get("photo_ref")
    if not ref:
        return None
    p = (Path(photo_rec["root"]) / ref).resolve()
    return p if p.exists() and p.is_file() else None


def _build_html_candidates(paths: list[Path]):
    records, chat_name = _html_pages(paths)
    index = {r["id"]: r for r in records if r.get("id")}
    pos = {r["id"]: i for i, r in enumerate(records) if r.get("id")}
    candidates = []
    unresolved = []

    for rec in records:
        if rec.get("service"):
            continue
        info = parse_customer_info(rec.get("text", ""))
        if not is_customer_record(info):
            continue

        photo_rec = rec if rec.get("photo_ref") else _resolve_reply_photo(rec, index)
        link_type = "same_message" if rec.get("photo_ref") else "reply"
        if photo_rec is None:
            photo_rec = _resolve_joined_photo(rec, records, pos)
            link_type = "adjacent" if photo_rec else "unresolved"

        if not photo_rec:
            unresolved.append(rec)
            continue

        candidates.append({
            "profile_id": rec.get("id", ""),
            "photo_message_id": photo_rec.get("id", ""),
            "photo_ref": photo_rec.get("photo_ref"),
            "photo_path": _resolve_photo_path(photo_rec),
            "sender": rec.get("sender") or photo_rec.get("sender") or "history",
            "raw_text": rec.get("text", ""),
            "customer_data": _customer_data(info, rec.get("sender") or photo_rec.get("sender") or "history"),
            "link_type": link_type,
            "chat_id": chat_name,
        })

    return records, candidates, unresolved, chat_name


def analyze_html_export(paths: list[Path]):
    records, candidates, unresolved, _ = _build_html_candidates(paths)
    return {
        "scanned": sum(1 for r in records if not r.get("service")),
        "photo_refs": sum(1 for r in records if r.get("photo_ref")),
        "profile_records": len(candidates) + len(unresolved),
        "resolved_profiles": len(candidates),
        "unresolved_profiles": len(unresolved),
        "query_photos": sum(1 for r in records if r.get("photo_ref") and _query_like(r.get("text", ""))),
    }


def _import_candidate(candidate: dict, stats: dict):
    image = candidate.get("photo_path")
    if image is None or not Path(image).exists():
        stats["files_missing"] += 1
        stats["skipped"] += 1
        return
    try:
        result = check_image(image)
        if result["type"] == "same":
            stats["duplicates"] += 1
            stats["skipped"] += 1
            return
        obj_key = _upload_image_safe(Path(image), result["features"]["sha256"])
        save_customer_and_image(
            Path(image), "", "",
            candidate.get("sender", "history"), "", candidate.get("chat_id", "history"),
            candidate.get("customer_data") or {}, candidate.get("raw_text", ""),
            "history", candidate.get("profile_id", ""), obj_key,
            features=result.get("features"),
        )
        stats["images"] += 1
    except Exception:
        log.exception("历史客户图片导入失败: %s", image)
        stats["skipped"] += 1


def import_html_pages(paths: list[Path]):
    """导入 Telegram HTML：一张照片 = 一个客户。

    正式资料存在时附加到该照片客户；没有资料的照片也照样建立客户记录。
    Sticker / 缩略图不作为客户照片。
    """
    stats = _stats()
    records, candidates, unresolved, chat_name = _build_html_candidates(paths)
    stats["scanned"] = sum(1 for r in records if not r.get("service"))
    stats["photo_refs"] = sum(1 for r in records if r.get("photo_ref"))
    stats["profile_records"] = len(candidates) + len(unresolved)
    stats["resolved_profiles"] = len(candidates)
    stats["unresolved_profiles"] = len(unresolved)
    stats["query_photos"] = sum(1 for r in records if r.get("photo_ref") and _query_like(r.get("text", "")))

    # photo message id -> best available formal customer profile
    profile_by_photo = {}
    for c in candidates:
        pid = str(c.get("photo_message_id") or "")
        if pid:
            profile_by_photo[pid] = c

    for rec in records:
        if rec.get("service") or not rec.get("photo_ref"):
            continue
        image = _resolve_photo_path(rec)
        c = profile_by_photo.get(str(rec.get("id") or ""))
        if c:
            candidate = c
        else:
            candidate = {
                "profile_id": str(rec.get("id") or ""),
                "photo_message_id": str(rec.get("id") or ""),
                "photo_ref": rec.get("photo_ref"),
                "photo_path": image,
                "sender": rec.get("sender") or "history",
                "raw_text": "",
                "customer_data": {"name":"", "age":"", "job":"", "income":"", "work_year":"", "software":"", "receiver":""},
                "link_type": "photo_only",
                "chat_id": chat_name,
            }
            stats.setdefault("photo_only_customers", 0)
            stats["photo_only_customers"] += 1
        _import_candidate(candidate, stats)
    return stats

def import_html(path):
    return import_html_pages([Path(path)])


def _image_ref_from_json_msg(msg):
    for key in ("photo", "file", "thumbnail"):
        v = msg.get(key)
        if isinstance(v, str) and Path(v).suffix.lower() in IMAGE_EXT:
            return v
    return None


def import_json(path):
    """Telegram result.json：同样只导入正式资料+明确图片关联。"""
    path = Path(path)
    root = path.parent
    data = json.loads(path.read_text(encoding="utf-8"))
    messages = [m for m in (data.get("messages", []) if isinstance(data, dict) else []) if isinstance(m, dict)]
    stats = _stats()
    stats["scanned"] = len(messages)
    chat_id = str(data.get("id", data.get("name", "history"))) if isinstance(data, dict) else "history"
    index = {str(m.get("id")): m for m in messages if m.get("id") is not None}

    for msg in messages:
        if _image_ref_from_json_msg(msg):
            stats["photo_refs"] += 1
            if _query_like(_flat_text(msg.get("text", ""))):
                stats["query_photos"] += 1

    for msg in messages:
        text = _flat_text(msg.get("text", ""))
        info = parse_customer_info(text)
        if not is_customer_record(info):
            continue
        stats["profile_records"] += 1
        photo_msg = msg if _image_ref_from_json_msg(msg) else None
        if photo_msg is None:
            rid = msg.get("reply_to_message_id")
            seen = set()
            for _ in range(4):
                if rid is None or str(rid) in seen:
                    break
                seen.add(str(rid))
                target = index.get(str(rid))
                if not target:
                    break
                if _image_ref_from_json_msg(target):
                    photo_msg = target
                    break
                rid = target.get("reply_to_message_id")

        if photo_msg is None:
            stats["unresolved_profiles"] += 1
            continue

        stats["resolved_profiles"] += 1
        ref = _image_ref_from_json_msg(photo_msg)
        image = (root / ref).resolve() if ref else None
        sender = str(msg.get("from") or msg.get("actor") or "history")
        candidate = {
            "profile_id": str(msg.get("id", "")),
            "photo_path": image if image and image.exists() else None,
            "sender": sender,
            "raw_text": text,
            "customer_data": _customer_data(info, sender),
            "chat_id": chat_id,
        }
        _import_candidate(candidate, stats)
    return stats


def import_manifest(path: Path):
    """导入紧凑历史客户包；允许 customer 为空，一张照片仍算一个客户。"""
    path = Path(path)
    root = path.parent
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("format") != "tg_customer_bundle_v1":
        return {**_stats(), "error": "不支持的 history_manifest.json 格式"}
    entries = data.get("customers") or []
    stats = _stats()
    stats["scanned"] = len(entries)
    stats["photo_refs"] = len(entries)
    stats["profile_records"] = sum(1 for e in entries if any(str(v or "").strip() for v in (e.get("customer") or {}).values()))
    stats["resolved_profiles"] = stats["profile_records"]
    stats["photo_only_customers"] = len(entries) - stats["profile_records"]
    for e in entries:
        image = (root / str(e.get("image", ""))).resolve()
        info = e.get("customer") or {}
        candidate = {
            "profile_id": str(e.get("profile_message_id") or e.get("photo_message_id") or ""),
            "photo_path": image if image.exists() else None,
            "sender": str(e.get("sender") or "history"),
            "raw_text": str(e.get("raw_text") or ""),
            "customer_data": info,
            "chat_id": str(e.get("chat_id") or "history"),
        }
        _import_candidate(candidate, stats)
    return stats

def import_images_direct(image_paths):
    """纯图片 ZIP：按用户规则，一张图片就是一个客户；无资料也正式入库。"""
    paths = [Path(x) for x in image_paths]
    stats = _stats()
    stats["scanned"] = len(paths)
    stats["photo_refs"] = len(paths)
    stats["photo_only_customers"] = len(paths)
    blank = {"name":"", "age":"", "job":"", "income":"", "work_year":"", "software":"", "receiver":""}
    for image in paths:
        candidate = {
            "profile_id": image.name,
            "photo_path": image,
            "sender": "history",
            "raw_text": "",
            "customer_data": blank,
            "chat_id": "history",
        }
        _import_candidate(candidate, stats)
    return stats

def import_history(file_path):
    p = Path(file_path)
    if p.suffix.lower() == ".json":
        if p.name == MANIFEST_NAME:
            return import_manifest(p)
        return import_json(p)
    if p.suffix.lower() in {".html", ".htm"}:
        return import_html(p)
    if p.suffix.lower() == ".zip":
        with tempfile.TemporaryDirectory(prefix="tgimport_") as td:
            _safe_extract(p, td)
            root = Path(td)

            manifests = list(root.rglob(MANIFEST_NAME))
            if manifests:
                return import_manifest(manifests[0])

            jsons = list(root.rglob("result.json"))
            if jsons:
                return import_json(jsons[0])

            htmls = [x for x in root.rglob("messages*.html") if re.fullmatch(r"messages\d*\.html", x.name, flags=re.I)]
            if htmls:
                return import_html_pages(htmls)

            all_images = [f for f in root.rglob("*") if f.is_file() and f.suffix.lower() in IMAGE_EXT and "_thumb." not in f.name]
            if all_images:
                return import_images_direct(all_images)

            return {**_stats(), "error": "压缩包内没有找到 Telegram messages*.html / result.json / history_manifest.json"}

    return {**_stats(), "error": "仅支持 zip / json / html"}
