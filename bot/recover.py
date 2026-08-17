"""/recover_images - restore missing persisted image files from Telegram.

Only the current Telegram group/supergroup administrators may run this command.
No Telegram user-id whitelist is required.
"""
import asyncio
import logging
import tempfile
from functools import partial
from pathlib import Path

from telegram import Update
from telegram.ext import ContextTypes

from core.database import get_conn
from core.object_storage import image_exists, image_object_key, upload_image

log = logging.getLogger(__name__)
_recover_lock = asyncio.Lock()


def _run_sync(func, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(None, partial(func, *args, **kwargs))


async def _is_authorized(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    chat = update.effective_chat
    user = update.effective_user
    if not chat or not user or chat.type not in {"group", "supergroup"}:
        return False
    try:
        member = await context.bot.get_chat_member(chat.id, user.id)
        return member.status in {"administrator", "creator", "owner"}
    except Exception:
        log.warning("recover_images: 无法读取群管理员状态", exc_info=True)
        return False


def _fetch_stale_images():
    conn = get_conn()
    try:
        rows = conn.execute(
            """
            SELECT id, file_id, file_unique_id, file_path
            FROM images
            WHERE file_id IS NOT NULL
              AND (
                file_path IS NULL
                OR file_path = ''
                OR file_path NOT LIKE 'images/%'
              )
            ORDER BY id
            """
        ).fetchall()
        return [(r["id"], r["file_id"], r["file_unique_id"], r["file_path"]) for r in rows]
    finally:
        conn.close()


def _update_file_path(image_id: int, object_key: str):
    conn = get_conn()
    try:
        conn.execute("UPDATE images SET file_path=? WHERE id=?", (object_key, image_id))
        conn.commit()
    finally:
        conn.close()


async def recover_images_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    msg0 = update.effective_message
    if not msg0:
        return
    if not await _is_authorized(update, context):
        await msg0.reply_text("❌ 无权执行此命令。此命令仅允许当前群管理员使用。")
        return

    if _recover_lock.locked():
        await msg0.reply_text("⏳ 恢复任务正在运行中，请等待当前任务完成后再试。")
        return

    async with _recover_lock:
        msg = await msg0.reply_text("🔍 正在扫描数据库中图片路径丢失的记录……")
        try:
            stale = await _run_sync(_fetch_stale_images)
        except Exception as exc:
            log.exception("recover_images: 查询数据库失败")
            await msg.edit_text(f"❌ 查询数据库失败：{type(exc).__name__}")
            return

        if not stale:
            await msg.edit_text("✅ 未发现需要修复的图片记录，数据库已是最新状态。")
            return

        await msg.edit_text(
            f"📋 发现 {len(stale)} 条需要修复的图片记录，开始恢复……\n"
            "（这可能需要几分钟）"
        )

        ok = 0
        fail = 0
        fail_ids = []
        for image_id, file_id, file_unique_id, _old_path in stale:
            key = image_object_key(file_unique_id or file_id)
            if await _run_sync(image_exists, key):
                try:
                    await _run_sync(_update_file_path, image_id, key)
                    ok += 1
                except Exception:
                    fail += 1
                    fail_ids.append(image_id)
                continue

            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            try:
                tg_file = await context.bot.get_file(file_id)
                await tg_file.download_to_drive(tmp_path)
                await _run_sync(upload_image, tmp_path, key)
                await _run_sync(_update_file_path, image_id, key)
                ok += 1
            except Exception as exc:
                log.warning("recover_images: 恢复失败 id=%s: %s", image_id, exc)
                fail += 1
                fail_ids.append(image_id)
            finally:
                tmp_path.unlink(missing_ok=True)

            if (ok + fail) % 10 == 0:
                try:
                    await msg.edit_text(
                        f"⏳ 进度：{ok + fail}/{len(stale)}\n✅ 已恢复：{ok}　❌ 失败：{fail}"
                    )
                except Exception:
                    pass

        lines = ["🎉 *图片恢复完成*\n", f"✅ 成功恢复：{ok} 张", f"❌ 失败：{fail} 张"]
        if fail_ids:
            ids = ", ".join(str(x) for x in fail_ids[:20])
            if len(fail_ids) > 20:
                ids += f" … 等 {len(fail_ids)} 条"
            lines.append(f"失败 image ID：{ids}")
        await msg.edit_text("\n".join(lines), parse_mode="Markdown")
        log.info("recover_images 完成: user_id=%s ok=%d fail=%d", user.id if user else None, ok, fail)
