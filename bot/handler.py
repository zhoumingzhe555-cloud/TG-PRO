"""
主消息处理器 — 所有阻塞 CPU 操作统一丢进线程池，不卡 Telegram 异步事件循环。
"""
import asyncio
import logging
from functools import partial
from pathlib import Path

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes

from config import IMAGE_DIR, IMPORT_DIR, OCR_FALLBACK, IMPORT_ADMINS_ONLY, IMAGE_CHECK_TIMEOUT, MAX_CONCURRENT_IMAGE_CHECKS, AUTO_COLLISION_THRESHOLD, MAX_IMAGE_BYTES, SHOW_MATCHED_IMAGE
from core.customer import parse_customer_info, is_customer_record, public_customer_data
from core.database import get_conn, save_pending_buffer, delete_pending_buffer, load_pending_buffers, cleanup_expired_pending_buffers
from core.object_storage import upload_image, image_object_key
from core.notify import notify_dashboard_update
from ocr.extractor import extract_text, likely_contains_text
from history_ai.importer import import_history
from core.image_match import (
    check_image,
    save_customer_and_image,
    save_image_alias,
    create_collision,
    confirm_collision,
    get_collision,
    get_customer_by_id,
    update_customer_fields,
    get_image_by_id,
)

log = logging.getLogger(__name__)
_IMAGE_CHECK_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_IMAGE_CHECKS)

# ── 待匹配缓冲 ───────────────────────────────────────────────────────────────
# 群聊中，有时先发图片再发文字资料。
# key: (chat_id, user_id)  value: {"ts": float, "path": Path, "file_id": str,
#                                   "file_unique_id": str, "obj_key": str|None}
_pending: dict = {}
# message-level index prevents customer data from being attached to the wrong image
# when one staff member sends several photos quickly and replies to a specific one.
_pending_by_message: dict = {}
# 跨发送者精确回复绑定：只看 chat_id + 原图片 message_id。
# 例如 A 发图片，B 回复该图片补客户资料，也必须能正确入库。
_pending_by_chat_message: dict = {}
_PENDING_TTL = 3600  # 图片等待客户资料：1 小时

# 群聊中，有时先发文字资料再发图片。
# key: (chat_id, user_id)  value: {"ts": float, "raw_text": str, "info": dict}
_pending_text: dict = {}
_PENDING_TEXT_TTL = 3600  # 客户资料等待图片：1 小时

# ── 字段元数据 ────────────────────────────────────────────────────────────────
FIELD_LABELS: dict[str, str] = {
    "name":      "姓名",
    "age":       "年龄",
    "job":       "职业",
    "income":    "收入",
    "work_year": "工作年限",
    "software":  "引流软件",
    "receiver":  "接粉人员",
}
# 中文标签 → 数据库字段名
FIELD_KEY_MAP: dict[str, str] = {v: k for k, v in FIELD_LABELS.items()}


async def photo_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    if not msg:
        return
    # Telegram 既可能把图片作为 PHOTO 发送，也可能“以文件发送”成为 image/* DOCUMENT。
    # 两种都走同一套防撞逻辑，避免高质量原图发送时机器人无反应。
    if msg.photo:
        photo = msg.photo[-1]
    elif msg.document and (
        (msg.document.mime_type or "").lower().startswith("image/")
        or (msg.document.file_name or "").lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif", ".avif"))
    ):
        photo = msg.document
    else:
        return
    user = update.effective_user
    chat = update.effective_chat
    # 私聊模式已关闭：只处理群聊 / 超级群。
    if not chat or chat.type not in {"group", "supergroup"}:
        return

    # 拒绝异常超大文件，避免一张坏图耗尽 Railway 内存。
    if getattr(photo, "file_size", None) and int(photo.file_size) > MAX_IMAGE_BYTES:
        await msg.reply_text(f"❌ 图片文件过大，请控制在 {MAX_IMAGE_BYTES // (1024*1024)}MB 以内")
        return

    # 下载图片
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    path = IMAGE_DIR / f"{photo.file_unique_id or photo.file_id}.jpg"
    tgfile = await context.bot.get_file(photo.file_id)
    await tgfile.download_to_drive(path)

    log.info("[photo] 下载完成 chat_type=%s file=%s", chat.type, photo.file_id)

    # ——— 图片检测（CPU密集，放入线程池）———
    try:
        # 限制同时运行的重型视觉任务，避免 Replit 小 CPU 被多个图片请求同时打满。
        async with _IMAGE_CHECK_SEMAPHORE:
            result = await asyncio.wait_for(
                _run_sync(check_image, path), timeout=IMAGE_CHECK_TIMEOUT
            )
    except asyncio.TimeoutError:
        log.error("[photo] check_image 超时（%.0fs）", IMAGE_CHECK_TIMEOUT)
        await msg.reply_text("❌ 检测超时，请重新发送图片")
        return
    except Exception:
        log.exception("[photo] 图片检测失败")
        await msg.reply_text("❌ 检测失败，请重新发送图片")
        return

    log.info("[photo] 检测完成 type=%s score=%s", result.get("type"), result.get("score"))

    # 同一图片：私聊→显示摘要+覆盖按钮；群聊→报撞
    if result["type"] == "same":
        matched = await _run_sync(_get_matched_customer, result["matched_image_id"])
        if chat.type == "private":
            # 私聊：提示已有客户资料，并在有新说明文字时提供覆盖选项
            customer_id = matched["id"] if matched else None
            summary = _format_customer_summary(matched) if matched else ""
            raw_text = (msg.caption or "").strip()
            text = f"ℹ️ *该图片已有客户资料*（ID: {customer_id}）" if customer_id else "ℹ️ 该图片已存在"
            if summary:
                text += f"\n\n{summary}"
            is_owner = matched and str(matched.get("submitter_id", "")) == str(user.id)
            if raw_text and customer_id and is_owner:
                info = parse_customer_info(raw_text)
                context.user_data[f"pending_override:{customer_id}"] = {
                    "raw_text": raw_text,
                    "info": public_customer_data(info),
                }
                kb = InlineKeyboardMarkup([[
                    InlineKeyboardButton(
                        "✏️ 覆盖资料",
                        callback_data=f"override_customer:{customer_id}",
                    ),
                    InlineKeyboardButton(
                        "❌ 取消",
                        callback_data=f"override_cancel:{customer_id}",
                    ),
                ]])
                text += "\n\n发现新的说明文字，是否用新资料覆盖已有客户信息？"
                await msg.reply_text(text, parse_mode="Markdown", reply_markup=kb)
            elif raw_text and customer_id and not is_owner:
                text += "\n\n⚠️ 只有原提交者可以覆盖该客户资料"
                await msg.reply_text(text, parse_mode="Markdown")
            else:
                await msg.reply_text(text, parse_mode="Markdown")
        else:
            # 群聊：同时把真正命中的历史图片发出来，便于人工核对。
            text = "🔴 撞客\n\n匹配类型：同一图片\n相似度：100%"
            if matched and matched.get("name"):
                text += f"\n已有客户：{matched['name']}"
            await _reply_match_result_with_history(msg, text, result["matched_image_id"])
        return

    # 相似图片：90% 及以上直接自动确认为撞客；低于 90% 才需要人工确认。
    if result["type"] == "similar":
        matched = await _run_sync(_get_matched_customer, result["matched_image_id"])
        collision_id = await _run_sync(
            create_collision,
            result["features"]["sha256"],
            result["matched_image_id"],
            _user_name(user),
            str(user.id),
            str(chat.id),
            result["match_type"],
            result["score"],
            result["features"].get("phash", ""),
            str(path),
            photo.file_id,
            photo.file_unique_id or "",
            next((x.get("feature") for x in (result.get("features",{}).get("copy_views") or []) if x.get("feature") is not None), None),
        )

        score = float(result.get("score") or 0.0)
        if score >= AUTO_COLLISION_THRESHOLD:
            # 直接把碰撞记录从 pending 变更为 confirmed，避免统计里仍显示“待确认”。
            await _run_sync(
                confirm_collision,
                collision_id,
                "confirmed",
                "系统自动确认",
                "system",
            )
            # 已确认同图变体会加入该客户的图片簇。以后同一裁剪/压缩版本可直接命中，
            # 机器人会随着真实确认结果积累更多同图版本，而不是每次重新猜。
            if result.get("learn_safe"):
                try:
                    await _run_sync(
                        save_image_alias, path, result["matched_image_id"],
                        photo.file_id, photo.file_unique_id or "", _user_name(user),
                        str(user.id), str(chat.id), "auto_alias", None, result.get("features"),
                    )
                except Exception:
                    log.warning("保存自动确认图片别名失败（不影响撞客结果）", exc_info=True)
            else:
                log.info("AUTO90 已确认撞客，但证据未达到自动学习标准，未写入 alias")
            text = (
                f"🔴 撞客\n\n"
                f"匹配类型：{result['match_type']}\n"
                f"相似度：{score:.2f}%"
            )
            if matched and matched.get("name"):
                text += f"\n已有客户：{matched['name']}"
                extra = _format_customer_summary(matched)
                if extra:
                    text += f"\n{extra}"
            text += f"\n\n✅ 已自动确认撞客（≥{AUTO_COLLISION_THRESHOLD:.0f}%）"
            await _reply_match_result_with_history(msg, text, result["matched_image_id"])
            return

        # 低于自动确认阈值的相似图片保留人工确认按钮。
        kb = InlineKeyboardMarkup([[
            InlineKeyboardButton("✅ 确认撞客", callback_data=f"collision:confirmed:{collision_id}"),
            InlineKeyboardButton("❌ 误判", callback_data=f"collision:false_positive:{collision_id}"),
        ]])
        text = (
            f"🟠 疑似撞客\n\n"
            f"匹配类型：{result['match_type']}\n"
            f"相似度：{score:.2f}%"
        )
        if matched and matched.get("name"):
            text += f"\n已有客户：{matched['name']}"
            extra = _format_customer_summary(matched)
            if extra:
                text += f"\n{extra}"
        # 人工确认按钮直接挂在历史命中图片上，审核者看到两张图后即可判断。
        await _reply_match_result_with_history(msg, text, result["matched_image_id"], reply_markup=kb)
        return

    # ——— 新图：群聊专用快速流程 ———
    # 图片“备注/说明文字”在 Telegram Bot API 中就是 caption。
    # caption 是正式入库资料的最高优先级：图 + 可识别客户资料 = 直接进入正式入库流程。
    # 同时支持多行备注，以及“姓名：... 年龄：... 职业：...”这种单行备注。
    raw_text = (msg.caption or "").strip()
    info = parse_customer_info(raw_text)
    if is_customer_record(info):
        log.info("[entry] 图片 caption/备注识别到正式客户资料 message_id=%s", msg.message_id)

    import time
    key = (str(chat.id), str(user.id))
    consumed_text_entry = None

    # 图片可以直接“回复一条客户资料”发送。即使资料和图片不是同一个人发送，
    # reply_to_message 也是最可靠的一对一关系，优先于任何时间邻近猜测。
    if not is_customer_record(info) and msg.reply_to_message is not None:
        replied_text = (msg.reply_to_message.text or msg.reply_to_message.caption or "").strip()
        replied_info = parse_customer_info(replied_text)
        if is_customer_record(replied_info):
            raw_text = replied_text
            info = replied_info
            try:
                replied_user = msg.reply_to_message.from_user
                replied_uid = str(replied_user.id) if replied_user else None
                if replied_uid:
                    replied_key = (str(chat.id), replied_uid)
                    pending_reply = _pending_text.get(replied_key)
                    if pending_reply and str(pending_reply.get("message_id")) == str(msg.reply_to_message.message_id):
                        _pending_text.pop(replied_key, None)
                        try:
                            await _run_sync(delete_pending_buffer, "text", str(chat.id), replied_uid)
                        except Exception:
                            log.warning("删除已通过回复绑定的文字缓冲失败", exc_info=True)
                        if context.job_queue is not None:
                            for _j in context.job_queue.get_jobs_by_name(f"pending_text_expire_{chat.id}_{replied_uid}"):
                                _j.schedule_removal()
            except Exception:
                log.debug("清理回复绑定的文字缓冲失败（不影响入库）", exc_info=True)

    if not is_customer_record(info):
        text_entry = _pending_text.get(key)
        if text_entry:
            if time.time() - text_entry["ts"] > _PENDING_TEXT_TTL:
                await _do_expire_pending_text(chat.id, user.id, text_entry["ts"], context.bot)
            else:
                del _pending_text[key]
                consumed_text_entry = text_entry
                if context.job_queue is not None:
                    for _j in context.job_queue.get_jobs_by_name(
                        f"pending_text_expire_{chat.id}_{user.id}"
                    ):
                        _j.schedule_removal()
                raw_text = text_entry["raw_text"]
                info = text_entry["info"]

    # OCR 是最慢的一层。先用一个轻量 OpenCV 门控判断图片是否像资料卡/截图；
    # 普通客户照片直接跳过 OCR，显著缩短“新客户”返回时间。
    if not is_customer_record(info) and OCR_FALLBACK:
        try:
            need_ocr = await _run_sync(likely_contains_text, path)
        except Exception:
            need_ocr = False
        if need_ocr:
            ocr_text = await _run_sync(extract_text, path)
            ocr_info = parse_customer_info(ocr_text)
            if is_customer_record(ocr_info):
                raw_text = ocr_text
                info = ocr_info

    if is_customer_record(info):
        saved = await _save_group_customer(
            msg,
            photo.file_id,
            photo.file_unique_id,
            path,
            user,
            chat,
            info,
            raw_text,
            features=result.get("features"),
        )
        if consumed_text_entry is not None:
            if saved:
                try:
                    await _run_sync(
                        delete_pending_buffer, "text", str(chat.id), str(user.id)
                    )
                except Exception:
                    log.warning("删除文字缓冲DB记录失败", exc_info=True)
            else:
                _pending_text[key] = consumed_text_entry
        return

    # 纯图片 = 查询模式。先立刻给最终结果，不等待 Object Storage / DB。
    # 图片仅在内存/临时文件中保留 1 小时，若随后补发正式客户资料才真正入库。
    _ts = time.time()
    _entry = {
        "ts": _ts,
        "path": path,
        "file_id": photo.file_id,
        "file_unique_id": photo.file_unique_id,
        "obj_key": None,
        "message_id": msg.message_id,
        "user_name": _user_name(user),
        "owner_user_id": str(user.id),
        "features": result.get("features"),
    }
    _pending[key] = _entry
    _pending_by_message[(str(chat.id), str(user.id), str(msg.message_id))] = _entry
    _pending_by_chat_message[(str(chat.id), str(msg.message_id))] = _entry

    _job_name = f"pending_expire_{chat.id}_{user.id}"
    if context.job_queue is not None:
        for _j in context.job_queue.get_jobs_by_name(_job_name):
            _j.schedule_removal()
        context.job_queue.run_once(
            _notify_pending_expired,
            when=_PENDING_TTL,
            data={"chat_id": chat.id, "user_id": user.id, "gen": _ts},
            name=_job_name,
        )

    # 用户看到的只有最终结果；后台缓冲持久化在回复之后执行。
    await msg.reply_text("🔍 检测结果\n\n恭喜您，是新客户")

    try:
        await _run_sync(
            save_pending_buffer,
            "image",
            str(chat.id),
            str(user.id),
            _ts,
            file_id=photo.file_id,
            file_unique_id=photo.file_unique_id,
            file_path=str(path),
            obj_key=None,
            message_id=msg.message_id,
            user_name=_user_name(user),
        )
    except Exception:
        log.warning("保存图片缓冲到DB失败", exc_info=True)


async def _do_expire_pending(chat_id, user_id: str, gen: float, bot) -> None:
    """纯图片查询缓冲到期后静默清理，不再在群里追加催资料提示。"""
    key = (str(chat_id), str(user_id))
    entry = _pending.get(key)
    if entry is None or entry["ts"] != gen:
        return
    _pending.pop(key, None)
    mid = entry.get("message_id")
    if mid is not None:
        _pending_by_message.pop((str(chat_id), str(user_id), str(mid)), None)
        _pending_by_chat_message.pop((str(chat_id), str(mid)), None)
    try:
        await _run_sync(delete_pending_buffer, "image", str(chat_id), str(user_id), mid)
    except Exception:
        log.warning("删除图片缓冲DB记录失败", exc_info=True)
    # 未正式入库的查询图片只属于临时文件，过期后释放磁盘。
    try:
        path = entry.get("path")
        if path and Path(path).exists():
            Path(path).unlink(missing_ok=True)
    except Exception:
        log.debug("清理临时查询图片失败", exc_info=True)


async def _do_expire_pending_text(chat_id, user_id: str, gen: float, bot) -> None:
    """集中式 _pending_text 过期处理。同 _do_expire_pending，通过 gen 防止竞态。"""
    key = (str(chat_id), str(user_id))
    entry = _pending_text.get(key)
    if entry is None or entry["ts"] != gen:
        return
    _pending_text.pop(key, None)
    try:
        delete_pending_buffer("text", str(chat_id), str(user_id))
    except Exception:
        log.warning("删除文字缓冲DB记录失败", exc_info=True)
    message_id = entry.get("message_id")
    user_name = entry.get("user_name", "")
    user_label = f"*{_md_escape(user_name)}*，" if user_name else ""
    try:
        await bot.send_message(
            chat_id=chat_id,
            text=(
                f"⏰ {user_label}你之前发送的客户资料等待图片超过 1 小时，已自动清除。\n"
                "如需入库，请重新发送客户资料和图片。"
            ),
            parse_mode="Markdown",
            reply_to_message_id=message_id,
        )
    except Exception:
        log.warning("发送文字缓冲过期通知失败", exc_info=True)


async def _notify_pending_expired(context: ContextTypes.DEFAULT_TYPE):
    """JobQueue 回调：图片缓冲（_pending）到期。"""
    data = context.job.data or {}
    await _do_expire_pending(data["chat_id"], data["user_id"], data["gen"], context.bot)


async def _notify_pending_text_expired(context: ContextTypes.DEFAULT_TYPE):
    """JobQueue 回调：文字缓冲（_pending_text）到期。"""
    data = context.job.data or {}
    await _do_expire_pending_text(data["chat_id"], data["user_id"], data["gen"], context.bot)


async def _save_group_customer(msg, file_id, file_unique_id, path, user, chat, info, raw_text, obj_key=None, features=None) -> bool:
    """群聊新客户入库公共逻辑（被 photo_handler 和 text_handler 共用）。

    返回 True  — 操作已终止（入库成功 或 重复图片），调用方可安全删除缓冲记录。
    返回 False — 入库失败，缓冲记录应保留以供重试或下次恢复。
    """
    if obj_key is None:
        obj_key = image_object_key(file_unique_id or file_id)
        try:
            await _run_sync(upload_image, path, obj_key)
        except Exception:
            log.warning("上传图片到 Object Storage 失败", exc_info=True)
            obj_key = None
    try:
        _, _, inserted = await _run_sync(
            save_customer_and_image,
            path, file_id, file_unique_id,
            _user_name(user), str(user.id), str(chat.id),
            public_customer_data(info), raw_text,
            "group", str(msg.message_id), obj_key, features,
        )
    except Exception:
        log.exception("群聊入库失败")
        await msg.reply_text("❌ 入库失败，请重试")
        return False
    if inserted:
        summary = _format_customer_summary(public_customer_data(info))
        reply = "✅ *新客户已入库*"
        if summary:
            reply += f"\n\n{summary}"
        await msg.reply_text(reply, parse_mode="Markdown")
    else:
        await msg.reply_text("ℹ️ 该图片已存在，未重复添加")
    return True


async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """监听群聊客户资料：支持同消息、先图后文、先文后图，以及跨发送者“回复图片补资料”。"""
    import time
    msg = update.effective_message
    if not msg or not msg.text:
        return
    chat = update.effective_chat
    user = update.effective_user
    if not chat or not user:
        return
    # 私聊文字由私聊图片流程处理，这里只处理群聊
    if chat.type == "private":
        return

    raw_text = msg.text.strip()
    info = parse_customer_info(raw_text)
    if not is_customer_record(info):
        return

    key = (str(chat.id), str(user.id))
    reply_mid = None
    try:
        if msg.reply_to_message is not None:
            reply_mid = str(msg.reply_to_message.message_id)
    except Exception:
        reply_mid = None
    entry = None
    if reply_mid:
        # 先允许“跨发送者回复绑定”：A 发图，B 回复该图发资料，也视为同一客户。
        entry = _pending_by_chat_message.get((str(chat.id), reply_mid))
        # 兼容旧内存索引。
        if entry is None:
            entry = _pending_by_message.get((str(chat.id), str(user.id), reply_mid))
    if entry is None:
        # 没有明确回复关系时，只自动配对同一发送者最近的一张图片，避免多人群串客。
        entry = _pending.get(key)

    # 路径 A：已有缓冲图片（先图后文）→ 合并入库。回复某张图片时优先精确绑定 message_id。
    if entry:
        # 超时则通过集中式 helper 过期并通知（兜底路径，防止被动到来时静默丢失）
        if time.time() - entry["ts"] > _PENDING_TTL:
            _expired_owner = str(entry.get("owner_user_id") or user.id)
            _expired_mid = entry.get("message_id")
            _expired_owner_key = (str(chat.id), _expired_owner)
            if _pending.get(_expired_owner_key) is entry:
                _pending.pop(_expired_owner_key, None)
            if _expired_mid is not None:
                _pending_by_chat_message.pop((str(chat.id), str(_expired_mid)), None)
                for _k, _v in list(_pending_by_message.items()):
                    if _v is entry:
                        _pending_by_message.pop(_k, None)
            try:
                await _run_sync(delete_pending_buffer, "image", str(chat.id), _expired_owner, _expired_mid)
            except Exception:
                log.warning("删除过期图片缓冲失败", exc_info=True)
            try:
                _p = entry.get("path")
                if _p and Path(_p).exists():
                    Path(_p).unlink(missing_ok=True)
            except Exception:
                pass
        else:
            # 先从内存移除，防止并发重入；DB 记录保留到入库成功后再删。
            # 跨发送者回复时，图片的 owner 不是当前资料发送者，必须按原图片发送者清理。
            _owner_user_id = str(entry.get("owner_user_id") or user.id)
            _owner_key = (str(chat.id), _owner_user_id)
            _was_current = _pending.get(_owner_key) is entry
            if _was_current:
                _pending.pop(_owner_key, None)
            mid = entry.get("message_id")
            if mid is not None:
                # entry 可能来自另一个发送者，跨发送者 map 必须按 chat+message 清理。
                _pending_by_chat_message.pop((str(chat.id), str(mid)), None)
                # 旧 map 的 user_id 可能不是当前资料发送者，因此按 identity 扫描删除。
                for _k, _v in list(_pending_by_message.items()):
                    if _v is entry:
                        _pending_by_message.pop(_k, None)
            # 只有处理当前“最近一张”图片时才取消/删除其 DB 缓冲；
            # 回复更早图片时不能误伤后面刚发的新图片。
            if _was_current and context.job_queue is not None:
                for _j in context.job_queue.get_jobs_by_name(f"pending_expire_{chat.id}_{_owner_user_id}"):
                    _j.schedule_removal()
            obj_key = entry["obj_key"]
            path = entry["path"]
            # 机器人重启后本地文件可能丢失，需要重新从 Telegram 下载
            if path is None or not Path(path).exists():
                fuid = entry.get("file_unique_id") or entry.get("file_id", "unknown")
                IMAGE_DIR.mkdir(parents=True, exist_ok=True)
                path = IMAGE_DIR / f"{fuid}.jpg"
                try:
                    tgfile = await context.bot.get_file(entry["file_id"])
                    await tgfile.download_to_drive(path)
                except Exception:
                    log.warning("无法重新下载缓冲图片，提示用户重新发送", exc_info=True)
                    # 重下失败：把内存条目还原，DB 记录不动，下次重启仍可恢复
                    _pending[_owner_key] = entry
                    if entry.get("message_id") is not None:
                        _pending_by_message[(str(chat.id), _owner_user_id, str(entry["message_id"]))] = entry
                        _pending_by_chat_message[(str(chat.id), str(entry["message_id"]))] = entry
                    await msg.reply_text(
                        "⚠️ 之前缓存的图片在重启后丢失，请重新发送图片和客户资料。"
                    )
                    return
            saved = await _save_group_customer(
                msg, entry["file_id"], entry["file_unique_id"],
                path, user, chat, info, raw_text, obj_key=obj_key,
                features=entry.get("features"),
            )
            if saved:
                # V1.9.3 图片缓冲按 message_id 持久化；无论是否为最近一张，都精确删除本次已入库图片。
                try:
                    await _run_sync(
                        delete_pending_buffer, "image", str(chat.id), _owner_user_id, mid
                    )
                except Exception:
                    log.warning("删除图片缓冲DB记录失败", exc_info=True)
            else:
                # 入库失败：恢复原有索引；较早图片不会覆盖当前最近图片。
                if _was_current:
                    _pending[_owner_key] = entry
                if entry.get("message_id") is not None:
                    _pending_by_message[(str(chat.id), _owner_user_id, str(entry["message_id"]))] = entry
                    _pending_by_chat_message[(str(chat.id), str(entry["message_id"]))] = entry
            return

    # 路径 B：没有缓冲图片（先文后图）→ 把资料文字暂存，等图片到来
    _text_ts = time.time()
    _pending_text[key] = {
        "ts": _text_ts,
        "raw_text": raw_text,
        "info": info,
        "message_id": msg.message_id,
        "user_name": _user_name(user),
    }
    try:
        save_pending_buffer(
            "text", str(chat.id), str(user.id), _text_ts,
            raw_text=raw_text,
            info=info,
            message_id=msg.message_id,
            user_name=_user_name(user),
        )
    except Exception:
        log.warning("保存文字缓冲到DB失败", exc_info=True)
    # 调度过期通知（若该用户已有待触发的任务，先取消旧的）
    # gen == ts，用于回调时校验是否为当代缓冲，防止竞态误通知
    _text_job_name = f"pending_text_expire_{chat.id}_{user.id}"
    if context.job_queue is not None:
        for _j in context.job_queue.get_jobs_by_name(_text_job_name):
            _j.schedule_removal()
        context.job_queue.run_once(
            _notify_pending_text_expired,
            when=_PENDING_TEXT_TTL,
            data={"chat_id": chat.id, "user_id": user.id, "gen": _text_ts},
            name=_text_job_name,
        )
    await msg.reply_text(
        "📋 *客户资料已暂存*\n\n"
        "请在 1 小时内发送该客户的图片，即可自动完成入库。",
        parse_mode="Markdown",
    )


async def collision_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    if not q:
        return
    chat = update.effective_chat
    if not chat or chat.type not in {"group", "supergroup"}:
        await q.answer("私聊模式已关闭", show_alert=True)
        return
    await q.answer()
    try:
        _, status, cid = q.data.split(":", 2)
        user = update.effective_user
        collision = await _run_sync(get_collision, int(cid))
        ok = await _run_sync(confirm_collision, int(cid), status, _user_name(user), str(user.id))
        if not ok:
            await q.answer("该记录已被处理", show_alert=True)
            return
        if status == "confirmed" and collision:
            try:
                qp = collision.get("query_file_path")
                if qp and Path(qp).exists():
                    await _run_sync(
                        save_image_alias, Path(qp), collision.get("matched_image_id"),
                        collision.get("query_file_id") or "", collision.get("query_file_unique_id") or "",
                        _user_name(user), str(user.id), str(chat.id), "confirmed_alias", None, None,
                    )
            except Exception:
                log.warning("保存人工确认图片别名失败（不影响确认结果）", exc_info=True)
        label = "✅ 已确认撞客" if status == "confirmed" else "✅ 已标记为误判"
        base = q.message.text or ""
        await q.edit_message_text(
            base + f"\n\n{label}\n确认人：{_user_name(user)}",
            parse_mode="Markdown",
        )
    except Exception:
        log.exception("撞客确认失败")
        await q.answer("操作失败", show_alert=True)

async def override_customer_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """处理私聊重复图片的「覆盖资料」/ 「取消」回调。"""
    q = update.callback_query
    if not q:
        return
    await q.answer()

    try:
        action, cid_str = q.data.split(":", 1)
        customer_id = int(cid_str)
        user = update.effective_user

        if action == "override_cancel":
            context.user_data.pop(f"pending_override:{customer_id}", None)
            base = q.message.text or ""
            await q.edit_message_text(base + "\n\n❌ 已取消覆盖")
            return

        # action == "override_customer"
        pending = context.user_data.pop(f"pending_override:{customer_id}", None)
        if not pending:
            await q.answer("操作已过期，请重新发送图片", show_alert=True)
            return

        new_info: dict = pending.get("info", {})
        # 仅覆盖非空字段，避免将已有值清空
        fields_to_update = {k: v for k, v in new_info.items() if v and str(v).strip()}

        if fields_to_update:
            caller_id = str(user.id) if user else None
            op_name = _user_name(user)
            ok = await _run_sync(
                update_customer_fields,
                customer_id,
                fields_to_update,
                caller_id,  # 只允许原始提交者覆盖，与 /edit 行为一致
                op_name,
                caller_id,
            )
        else:
            ok = False

        if ok:
            # 通知后台立即刷新客户列表（fire-and-forget，失败不影响主流程）
            asyncio.create_task(notify_dashboard_update(customer_id))
            updated_customer = await _run_sync(get_customer_by_id, customer_id)
            summary = _format_customer_summary(updated_customer or {})
            text = f"✅ *客户资料已覆盖更新*（ID: {customer_id}）"
            if summary:
                text += f"\n\n{summary}"
            confirmer = _user_name(user)
            text += f"\n\n操作人：{confirmer}"
            await q.edit_message_text(text, parse_mode="Markdown")
        else:
            await q.edit_message_text(
                (q.message.text or "") + "\n\n⚠️ 新说明文字未含可识别字段，资料未变更",
                parse_mode="Markdown",
            )
    except Exception:
        log.exception("覆盖资料回调失败")
        await q.answer("操作失败", show_alert=True)
async def _can_import(update, context) -> bool:
    chat = update.effective_chat
    user = update.effective_user
    if not chat or chat.type not in {"group", "supergroup"}:
        return False
    if not IMPORT_ADMINS_ONLY:
        return True
    try:
        m = await context.bot.get_chat_member(chat.id, user.id)
        return m.status in {"administrator", "creator", "owner"}
    except Exception:
        return False


async def document_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    chat = update.effective_chat
    if not chat or chat.type not in {"group", "supergroup"}:
        return
    doc = msg.document if msg else None
    if not doc:
        return
    # 图片作为“文件”发送时也必须触发防撞检测，而不是静默忽略。
    name = (doc.file_name or "").lower()
    if (doc.mime_type or "").lower().startswith("image/") or name.endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif", ".avif")):
        await photo_handler(update, context)
        return
    if not name.endswith((".zip", ".json", ".html", ".htm")):
        return
    if not await _can_import(update, context):
        await msg.reply_text("⚠️ 只有本群管理员可以导入历史客户资料")
        return

    # Telegram Bot API 限制：单文件最大 20 MB
    MAX_BYTES = 20 * 1024 * 1024
    if doc.file_size and doc.file_size > MAX_BYTES:
        mb = doc.file_size / 1024 / 1024
        await msg.reply_text(
            f"⚠️ 文件太大（{mb:.0f} MB），当前机器人下载上限设为 20 MB。\n\n"
            "不要使用 .001/.002 这种分卷文件，机器人无法单独解析。\n\n"
            "请使用独立完整的历史客户分包（每包 20 MB 内），或按日期分批导出。",
            parse_mode="Markdown",
        )
        return

    IMPORT_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = Path(doc.file_name or f"import_{doc.file_unique_id}").name
    path = IMPORT_DIR / safe_name
    try:
        f = await context.bot.get_file(doc.file_id)
        await f.download_to_drive(path)
        await msg.reply_text("⏳ 开始导入历史客户资料，请稍候…")
        # 历史导入可能很慢，放入线程池
        stats = await _run_sync(import_history, path)
        if stats.get("error"):
            await msg.reply_text(f"❌ 导入失败：{stats['error']}")
            return
        missing = stats.get("files_missing", 0)
        unresolved = stats.get("unresolved_profiles", 0)
        query_photos = stats.get("query_photos", 0)
        await msg.reply_text(
            "✅ *历史客户导入完成*\n\n"
            f"📨 扫描记录：{stats['scanned']:,} 条\n"
            f"🖼 发现客户照片：{stats.get('photo_refs', 0):,} 张\n"
            f"📋 有客户资料：{stats.get('profile_records', 0):,} 张\n"
            f"🗂 无资料照片客户：{stats.get('photo_only_customers', 0):,} 张\n"
            f"✅ 实际新增客户：{stats['images']:,} 人\n"
            f"♻️ 重复/同图变体：{stats.get('duplicates', 0):,} 张\n"
            f"🧩 合并到已有客户的裁剪/压缩版本：{stats.get('merged_variants', 0):,} 张\n"
            f"⚠️ 客户资料冲突记录：{stats.get('profile_conflicts', 0):,} 条\n"
            f"⚠️ 未能关联图片的资料：{unresolved:,} 条\n"
            f"📁 图片文件缺失：{missing:,} 张",
            parse_mode="Markdown",
        )
        if missing:
            await msg.reply_text(
                "📋 *图片文件缺失说明*\n\n"
                "你的 ZIP 里有聊天记录，但对应客户图片文件不在包内。\n\n"
                "请在 Telegram Desktop 重新导出该群，并同时勾选 *Photos*，然后把整个导出文件夹打包。\n\n"
                "当前规则：一张照片就是一个客户；即使没有客户资料也会加入客户库。",
                parse_mode="Markdown",
            )
    except Exception as e:
        log.exception("历史导入失败")
        await msg.reply_text(f"❌ 导入失败：{type(e).__name__}")

def _format_customer_card(customer: dict, customer_id: int) -> str:
    """格式化客户资料卡片，含 ID。所有用户输入值均经 Markdown 转义。"""
    lines = [f"📋 *客户资料 \\#{customer_id}*\n"]
    for key, label in FIELD_LABELS.items():
        val = _md_escape((customer.get(key) or "").strip())
        lines.append(f"  {label}：{val or '—'}")
    return "\n".join(lines)

async def edit_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /edit <客户ID> [字段:值 ...]

    不带参数时显示用法；仅有 ID 时展示资料并弹出交互编辑键盘；
    带字段时直接更新。
    """
    msg = update.effective_message
    if not msg:
        return
    if msg.chat.type != "private":
        await msg.reply_text("⚠️ /edit 命令只能在私聊中使用")
        return

    args = context.args or []
    if not args:
        await msg.reply_text(
            "📝 *编辑客户资料*\n\n"
            "用法 1（交互式）：`/edit <客户ID>`\n"
            "用法 2（直接更新）：`/edit <客户ID> 姓名:张三 年龄:28`\n\n"
            "支持字段：姓名、年龄、职业、收入、工作年限、引流软件、接粉人员\n\n"
            "💡 客户ID 可在入库成功提示中或管理后台找到。",
            parse_mode="Markdown",
        )
        return

    try:
        customer_id = int(args[0])
    except ValueError:
        await msg.reply_text("❌ 客户ID 须为数字，例如 `/edit 42`", parse_mode="Markdown")
        return

    customer = await _run_sync(get_customer_by_id, customer_id)
    if not customer:
        await msg.reply_text(f"❌ 未找到客户 \\#{customer_id}", parse_mode="Markdown")
        return

    # 仅允许原始提交者编辑自己提交的客户记录
    if not _can_edit_customer(str(update.effective_user.id), customer):
        await msg.reply_text("⚠️ 只能编辑自己提交的客户资料")
        return

    # 仅 ID：展示当前资料 + 交互键盘
    if len(args) == 1:
        await _send_edit_menu(msg, customer_id, customer)
        return

    # 解析字段:值 对
    fields: dict[str, str] = {}
    for arg in args[1:]:
        # 支持中文冒号和英文冒号
        for sep in ("：", ":"):
            if sep in arg:
                label_part, _, val_part = arg.partition(sep)
                label_part = label_part.strip()
                val_part = val_part.strip()
                key = FIELD_KEY_MAP.get(label_part)
                if key and val_part:
                    fields[key] = val_part
                break

    if not fields:
        await msg.reply_text(
            "❌ 未识别到有效字段。\n"
            "格式示例：`/edit 42 姓名:张三 年龄:28`",
            parse_mode="Markdown",
        )
        return

    user = update.effective_user
    ok = await _run_sync(update_customer_fields, customer_id, fields, str(user.id), _user_name(user), str(user.id))
    if ok:
        asyncio.create_task(notify_dashboard_update(customer_id))
        updated = "、".join(FIELD_LABELS[k] for k in fields if k in FIELD_LABELS)
        await msg.reply_text(
            f"✅ 客户 \\#{customer_id} 已更新：{updated}",
            parse_mode="Markdown",
        )
    else:
        await msg.reply_text(f"❌ 更新失败，未找到客户 \\#{customer_id}", parse_mode="Markdown")

async def edit_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """处理编辑字段选择按钮的回调。"""
    q = update.callback_query
    if not q:
        return
    await q.answer()

    try:
        parts = q.data.split(":", 2)
        action = parts[0]

        user_id = str(q.from_user.id) if q.from_user else ""

        if action == "edit_done":
            customer_id = int(parts[1])
            customer = await _run_sync(get_customer_by_id, customer_id)
            if customer and _can_edit_customer(user_id, customer):
                text = _format_customer_card(customer, customer_id)
                await q.edit_message_text(text + "\n\n✅ 编辑完成", parse_mode="Markdown")
            else:
                await q.edit_message_text("✅ 编辑完成")
            context.user_data.pop("awaiting_edit", None)
            return

        if action == "edit_select":
            customer_id = int(parts[1])
            customer = await _run_sync(get_customer_by_id, customer_id)
            if not customer:
                await q.answer("未找到客户记录", show_alert=True)
                return
            if not _can_edit_customer(user_id, customer):
                await q.answer("只能编辑自己提交的客户资料", show_alert=True)
                return
            text = _format_customer_card(customer, customer_id)
            text += "\n\n点击要修改的字段："
            await q.edit_message_text(
                text, parse_mode="Markdown", reply_markup=_build_edit_keyboard(customer_id)
            )
            return

        if action == "edit_field":
            customer_id = int(parts[1])
            # Verify ownership before allowing field edit
            customer = await _run_sync(get_customer_by_id, customer_id)
            if not customer:
                await q.answer("未找到客户记录", show_alert=True)
                return
            if not _can_edit_customer(user_id, customer):
                await q.answer("只能编辑自己提交的客户资料", show_alert=True)
                return
            field_key = parts[2]
            field_label = FIELD_LABELS.get(field_key)
            if not field_label:
                await q.answer("未知字段", show_alert=True)
                return

            # 记录等待状态（含 user_id 以便 reply handler 二次核验）
            context.user_data["awaiting_edit"] = {
                "customer_id": customer_id,
                "field_key": field_key,
                "field_label": field_label,
                "user_id": user_id,
            }
            await q.edit_message_text(
                f"请输入新的【{field_label}】内容：\n\n"
                f"（直接发送文字即可；发送 /cancel 取消）",
                parse_mode="Markdown",
            )
    except Exception:
        log.exception("编辑回调失败")
        await q.answer("操作失败", show_alert=True)

async def _send_edit_menu(msg, customer_id: int, customer: dict):
    """发送客户资料卡片 + 字段选择键盘。"""
    text = _format_customer_card(customer, customer_id)
    text += "\n\n点击要修改的字段："
    await msg.reply_text(text, parse_mode="Markdown", reply_markup=_build_edit_keyboard(customer_id))

async def edit_reply_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """当用户处于编辑等待状态时，接收文字并更新字段。"""
    msg = update.effective_message
    if not msg or not msg.text:
        return

    # /cancel 取消
    if msg.text.strip().lower() in {"/cancel", "/取消"}:
        context.user_data.pop("awaiting_edit", None)
        await msg.reply_text("❌ 已取消编辑")
        return

    state = context.user_data.get("awaiting_edit")
    if not state:
        return  # 不在等待状态，交给其他 handler

    customer_id = state["customer_id"]
    field_key = state["field_key"]
    field_label = state["field_label"]
    expected_user_id = state.get("user_id", "")
    actual_user_id = str(update.effective_user.id) if update.effective_user else ""

    # 二次校验：确保回复的是触发编辑流程的同一用户
    if expected_user_id and actual_user_id != expected_user_id:
        return  # 忽略其他用户的消息

    new_value = msg.text.strip()

    op_name = _user_name(update.effective_user) if update.effective_user else actual_user_id
    ok = await _run_sync(update_customer_fields, customer_id, {field_key: new_value}, actual_user_id, op_name, actual_user_id)
    context.user_data.pop("awaiting_edit", None)

    if ok:
        asyncio.create_task(notify_dashboard_update(customer_id))
        customer = await _run_sync(get_customer_by_id, customer_id)
        safe_val = _md_escape(new_value)
        text = f"✅ 【{field_label}】已更新为：{safe_val}\n\n"
        if customer:
            text += _format_customer_card(customer, customer_id)
            text += f"\n\n继续修改其他字段，或 /edit {customer_id} 再次编辑。"
        await msg.reply_text(text, parse_mode="Markdown")
    else:
        await msg.reply_text(f"❌ 更新失败，未找到客户 \\#{customer_id}", parse_mode="Markdown")

async def _reply_match_result_with_history(msg, text: str, matched_image_id: int, reply_markup=None):
    """Reply with the exact historical image that the matcher hit.

    The current customer image is already the message being replied to, so sending
    the historical hit as the reply gives staff an immediate visual A/B audit.
    Falls back to a normal text reply if the old file is unavailable.
    """
    if not SHOW_MATCHED_IMAGE:
        await msg.reply_text(text, reply_markup=reply_markup)
        return False
    try:
        image = await _run_sync(get_image_by_id, int(matched_image_id))
        rp = Path((image or {}).get("resolved_path") or "")
        if not rp.is_file():
            await msg.reply_text(text + "\n\n⚠️ 历史匹配图片文件暂不可读取", reply_markup=reply_markup)
            return False
        label = "📌 历史命中图片"
        if image.get("customer_name"):
            label += f"：{image['customer_name']}"
        label += f"（图片ID {image.get('id')}）"
        caption = f"{text}\n\n{label}\n↕️ 可直接与上方本次提交图片人工对比"
        # Telegram photo captions are limited; keep the most useful portion.
        if len(caption) > 1000:
            caption = caption[:997] + "..."
        try:
            with rp.open("rb") as fh:
                await msg.reply_photo(photo=fh, caption=caption, reply_markup=reply_markup)
        except Exception:
            # Some historical images may be too large/unusual for sendPhoto; send
            # them as documents so review is still possible.
            with rp.open("rb") as fh:
                await msg.reply_document(document=fh, caption=caption, reply_markup=reply_markup)
        return True
    except Exception:
        log.warning("发送历史匹配图片失败 image_id=%s", matched_image_id, exc_info=True)
        await msg.reply_text(text + "\n\n⚠️ 历史匹配图片发送失败", reply_markup=reply_markup)
        return False


def _can_edit_customer(user_id: str, customer: dict) -> bool:
    """Return True only if this Telegram user submitted the customer record."""
    return str(customer.get("submitter_id", "")) == str(user_id)

def _md_escape(text: str) -> str:
    """Escape special characters for Telegram Markdown v1 mode.

    Prevents user-supplied field values from breaking message formatting.
    """
    # Markdown v1 special chars: * _ ` [
    for ch in ("\\", "*", "_", "`", "["):
        text = text.replace(ch, "\\" + ch)
    return text

def _user_name(user):
    if not user:
        return "未知"
    return user.full_name or user.username or str(user.id)

def _get_matched_customer(image_id: int) -> dict | None:
    """根据 image_id 反查客户资料，用于撞客消息展示。"""
    try:
        conn = get_conn()
        row = conn.execute(
            "SELECT c.* FROM images i LEFT JOIN customers c ON i.customer_id=c.id WHERE i.id=? LIMIT 1",
            (image_id,)
        ).fetchone()
        conn.close()
        if row:
            return dict(row)
    except Exception:
        pass
    return None

def _format_customer_summary(data: dict) -> str:
    """把客户资料 dict 格式化为多行摘要。"""
    labels = [
        ("姓名", "name"),
        ("年龄", "age"),
        ("职业", "job"),
        ("收入", "income"),
        ("工作年限", "work_year"),
        ("引流软件", "software"),
        ("接粉人员", "receiver"),
    ]
    lines = []
    for label, key in labels:
        val = (data.get(key) or "").strip()
        if val:
            lines.append(f"  {label}：{val}")
    return "\n".join(lines)

async def restore_pending_buffers(bot, job_queue=None) -> None:
    """机器人重启后，从数据库恢复未完成的群聊缓冲到内存字典。

    - image 类型（先图后文）：恢复到 _pending；本地文件路径保留供后续匹配时校验，
      若文件不存在则在文字到来时按需重新从 Telegram 下载。
    - text 类型（先文后图）：直接恢复到 _pending_text（纯文本，无文件依赖）。
    - 已过期的条目立即向用户发送提示并清除，不会静默丢失。
    - job_queue 不为 None 时，为每条恢复的条目重新注册过期通知 Job，
      when 设为剩余 TTL（ts + TTL - now），确保重启后超时仍能准时触发。
    """
    import time

    # 兜底清理：删除超过 2 小时的孤立记录（崩溃/通知失败后残留的脏数据）。
    # 必须在加载前执行，使 load_pending_buffers 只返回可能仍有效的记录。
    _MAX_ORPHAN_AGE = max(_PENDING_TTL, _PENDING_TEXT_TTL) * 2  # 7200 秒 / 2 小时
    try:
        swept = await _run_sync(cleanup_expired_pending_buffers, _MAX_ORPHAN_AGE)
        if swept:
            log.info("孤立缓冲兜底清理：删除 %d 条超过 %.0f 秒的记录", swept, _MAX_ORPHAN_AGE)
    except Exception:
        log.warning("孤立缓冲兜底清理失败（不影响恢复流程）", exc_info=True)

    try:
        entries = await _run_sync(load_pending_buffers)
    except Exception:
        log.exception("从数据库恢复缓冲数据失败")
        return

    now = time.time()
    restored_image = 0
    restored_text = 0
    expired = 0

    for entry in entries:
        btype = entry.get("buffer_type")
        chat_id = entry.get("chat_id")
        user_id = entry.get("user_id")
        ts = entry.get("ts", 0)
        key = (str(chat_id), str(user_id))

        if btype == "image":
            if now - ts > _PENDING_TTL:
                # 纯图片查询缓冲已过期：静默清理，不在群里追加提示。
                try:
                    await _run_sync(
                        delete_pending_buffer, "image", chat_id, user_id, entry.get("message_id")
                    )
                except Exception:
                    pass
                try:
                    file_path = entry.get("file_path")
                    if file_path and Path(file_path).exists():
                        Path(file_path).unlink(missing_ok=True)
                except Exception:
                    pass
                expired += 1
                continue

            # 恢复到内存
            file_path = entry.get("file_path")
            path = Path(file_path) if file_path else None
            _restored_entry = {
                "ts": ts,
                "path": path,
                "file_id": entry.get("file_id"),
                "file_unique_id": entry.get("file_unique_id"),
                "obj_key": entry.get("obj_key"),
                "message_id": int(entry["message_id"]) if entry.get("message_id") else None,
                "user_name": entry.get("user_name", ""),
                "owner_user_id": str(user_id),
                "features": None,
            }
            _pending[key] = _restored_entry
            if _restored_entry.get("message_id") is not None:
                _pending_by_message[(str(chat_id), str(user_id), str(_restored_entry["message_id"]))] = _restored_entry
                _pending_by_chat_message[(str(chat_id), str(_restored_entry["message_id"]))] = _restored_entry
            # 重新注册过期通知 Job，when = 剩余 TTL（至少 1 秒）
            if job_queue is not None:
                _remaining = max(ts + _PENDING_TTL - now, 1)
                _job_name = f"pending_expire_{chat_id}_{user_id}"
                for _j in job_queue.get_jobs_by_name(_job_name):
                    _j.schedule_removal()
                job_queue.run_once(
                    _notify_pending_expired,
                    when=_remaining,
                    data={"chat_id": chat_id, "user_id": user_id, "gen": ts},
                    name=_job_name,
                )
            restored_image += 1

        elif btype == "text":
            if now - ts > _PENDING_TEXT_TTL:
                # 已过期：删除 DB 记录并通知用户
                try:
                    await _run_sync(delete_pending_buffer, "text", chat_id, user_id)
                except Exception:
                    pass
                message_id = entry.get("message_id")
                user_name = entry.get("user_name", "")
                user_label = f"*{_md_escape(user_name)}*，" if user_name else ""
                try:
                    await bot.send_message(
                        chat_id=chat_id,
                        text=(
                            f"⏰ {user_label}你之前发送的客户资料等待图片超过 1 小时，已自动清除。\n"
                            "如需入库，请重新发送客户资料和图片。"
                        ),
                        parse_mode="Markdown",
                        reply_to_message_id=int(message_id) if message_id else None,
                    )
                except Exception:
                    log.warning("发送恢复过期文字缓冲通知失败 chat_id=%s", chat_id, exc_info=True)
                expired += 1
                continue

            # 恢复到内存
            _pending_text[key] = {
                "ts": ts,
                "raw_text": entry.get("raw_text", ""),
                "info": entry.get("info") or {},
                "message_id": int(entry["message_id"]) if entry.get("message_id") else None,
                "user_name": entry.get("user_name", ""),
            }
            # 重新注册过期通知 Job，when = 剩余 TTL（至少 1 秒）
            if job_queue is not None:
                _remaining = max(ts + _PENDING_TEXT_TTL - now, 1)
                _text_job_name = f"pending_text_expire_{chat_id}_{user_id}"
                for _j in job_queue.get_jobs_by_name(_text_job_name):
                    _j.schedule_removal()
                job_queue.run_once(
                    _notify_pending_text_expired,
                    when=_remaining,
                    data={"chat_id": chat_id, "user_id": user_id, "gen": ts},
                    name=_text_job_name,
                )
            restored_text += 1

    if restored_image or restored_text or expired:
        log.info(
            "缓冲恢复完成：图片缓冲 %d 条，文字缓冲 %d 条，过期清理 %d 条",
            restored_image, restored_text, expired,
        )


def _run_sync(func, *args, **kwargs):
    """在默认线程池中运行同步阻塞函数，返回 awaitable。"""
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(None, partial(func, *args, **kwargs))

def _build_edit_keyboard(customer_id: int) -> InlineKeyboardMarkup:
    """构建字段选择键盘，每行两个按钮。"""
    buttons = [
        InlineKeyboardButton(label, callback_data=f"edit_field:{customer_id}:{key}")
        for key, label in FIELD_LABELS.items()
    ]
    rows = [buttons[i:i+2] for i in range(0, len(buttons), 2)]
    rows.append([InlineKeyboardButton("✅ 完成编辑", callback_data=f"edit_done:{customer_id}")])
    return InlineKeyboardMarkup(rows)
