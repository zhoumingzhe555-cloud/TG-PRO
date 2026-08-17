import asyncio
import logging
from telegram.ext import Application, MessageHandler, CommandHandler, CallbackQueryHandler, filters
from config import BOT_TOKEN, REINDEX_BATCH_SIZE, REINDEX_INTERVAL_SEC, COPY_REINDEX_BATCH_SIZE, COPY_REINDEX_INTERVAL_SEC
from bot.handler import (
    photo_handler, document_handler, collision_callback, text_handler,
    restore_pending_buffers, _PENDING_TTL, _PENDING_TEXT_TTL,
)
from bot.commands import start_command, help_command, stats_command, ai_status_command
from bot.recover import recover_images_command
from ai.embedding import preload
from ai.copy_embedding import preload as preload_copy_ai
from core.database import cleanup_expired_pending_buffers
from core.image_match import reindex_missing_signatures, reindex_missing_copy_features, count_copy_index_status, count_signature_index_status, cleanup_orphan_image_files

log = logging.getLogger(__name__)

# 仅群聊 / 超级群启用。私聊模式已完全关闭。
GROUPS_ONLY = filters.ChatType.GROUPS

# 孤立缓冲记录的最大保留时长：取两种缓冲 TTL 中较大者的 2 倍
_ORPHAN_MAX_AGE = max(_PENDING_TTL, _PENDING_TEXT_TTL) * 2
_CLEANUP_INTERVAL = 3600


async def _periodic_cleanup_pending_buffers(context) -> None:
    """定期清理 pending_buffer 表中的孤立记录。"""
    try:
        loop = asyncio.get_running_loop()
        deleted = await loop.run_in_executor(
            None, cleanup_expired_pending_buffers, _ORPHAN_MAX_AGE
        )
        if deleted:
            log.info(
                "定期清理 pending_buffer：删除 %d 条孤立记录（超过 %.0f 秒）",
                deleted, _ORPHAN_MAX_AGE,
            )
    except Exception:
        log.warning("定期清理 pending_buffer 失败（不影响机器人运行）", exc_info=True)




async def _periodic_reindex(context) -> None:
    """Backfill old multi-view hashes, then remove this repeating job at 100%."""
    try:
        loop = asyncio.get_running_loop()
        total,indexed,missing = await loop.run_in_executor(None, count_signature_index_status)
        if missing <= 0:
            if getattr(context, "job", None):
                context.job.schedule_removal()
            log.info("多尺度Hash旧图库索引已完成 %d/%d，后台任务停止", indexed, total)
            return
        done = await loop.run_in_executor(None, reindex_missing_signatures, REINDEX_BATCH_SIZE)
        total,indexed,missing = await loop.run_in_executor(None, count_signature_index_status)
        if done:
            log.info("多尺度Hash索引后台升级：本轮 %d 张；已完成 %d/%d，剩余 %d", done, indexed, total, missing)
        if missing <= 0 and getattr(context, "job", None):
            context.job.schedule_removal()
            log.info("多尺度Hash旧图库索引100%%完成，后台任务已停止")
    except Exception:
        log.warning("多尺度Hash索引后台升级失败（不影响实时检测）", exc_info=True)

async def _periodic_copy_reindex(context) -> None:
    """Backfill SSCD only while missing rows exist; stop permanently at 100%."""
    try:
        loop=asyncio.get_running_loop()
        total,indexed,missing=await loop.run_in_executor(None,count_copy_index_status)
        if missing <= 0:
            if getattr(context, "job", None):
                context.job.schedule_removal()
            log.info("V1.9 SSCD AI旧图库索引已完成 %d/%d，后台任务停止",indexed,total)
            return
        done=await loop.run_in_executor(None,reindex_missing_copy_features,COPY_REINDEX_BATCH_SIZE)
        total,indexed,missing=await loop.run_in_executor(None,count_copy_index_status)
        if done:
            log.info("V1.9 SSCD AI索引升级：本轮 %d 张；已完成 %d/%d，剩余 %d",done,indexed,total,missing)
        if missing <= 0 and getattr(context, "job", None):
            context.job.schedule_removal()
            log.info("V1.9 SSCD AI旧图库索引100%%完成，后台任务已停止；新客户入库时即时生成SSCD特征")
    except Exception:
        log.warning("V1.9 SSCD AI索引升级失败（经典视觉仍可用）",exc_info=True)


async def _periodic_cleanup_orphan_images(context) -> None:
    try:
        loop=asyncio.get_running_loop()
        deleted=await loop.run_in_executor(None,cleanup_orphan_image_files,24)
        if deleted:
            log.info("清理未入库临时图片：%d 张",deleted)
    except Exception:
        log.warning("清理未入库临时图片失败（不影响机器人运行）",exc_info=True)


async def _post_init(app):
    try:
        await app.bot.delete_webhook(drop_pending_updates=False)
    except Exception:
        pass

    loop=asyncio.get_running_loop()
    copy_ok=False
    try:
        copy_ok=bool(await loop.run_in_executor(None,preload_copy_ai))
        if not copy_ok:
            log.warning("SSCD同图AI暂不可用，机器人将自动使用经典视觉+轻量AI备用")
    except Exception:
        log.exception("SSCD同图AI预加载失败（不影响经典视觉检测）")
    if not copy_ok:
        try:
            await loop.run_in_executor(None,preload)
        except Exception:
            log.exception("轻量AI备用模型预加载失败")

    if app.job_queue is None:
        log.warning(
            "JobQueue 未初始化（apscheduler 未安装），群聊缓冲过期通知及恢复不可用，其余功能正常。"
        )
        return

    try:
        await restore_pending_buffers(app.bot, job_queue=app.job_queue)
    except Exception:
        log.exception("恢复群聊缓冲失败（不影响机器人启动）")

    try:
        app.job_queue.run_repeating(
            _periodic_cleanup_pending_buffers,
            interval=_CLEANUP_INTERVAL,
            first=_CLEANUP_INTERVAL,
            name="periodic_cleanup_pending_buffers",
        )
        log.info(
            "已注册定期清理任务：每 %.0f 秒清理超过 %.0f 秒的孤立缓冲记录",
            _CLEANUP_INTERVAL, _ORPHAN_MAX_AGE,
        )
    except Exception:
        log.warning("注册定期清理任务失败（不影响机器人运行）", exc_info=True)

    try:
        total,indexed,missing=await loop.run_in_executor(None,count_signature_index_status)
        if missing > 0:
            app.job_queue.run_repeating(
                _periodic_reindex, interval=REINDEX_INTERVAL_SEC, first=8, name="v18_reindex_signatures",
            )
            log.info("多尺度Hash待补 %d 张：已启动后台任务，每 %d 秒最多 %d 张", missing, REINDEX_INTERVAL_SEC, REINDEX_BATCH_SIZE)
        else:
            log.info("多尺度Hash索引已完成 %d/%d：不启动后台任务", indexed, total)
    except Exception:
        log.warning("注册 多尺度Hash索引升级任务失败（不影响实时检测）", exc_info=True)

    try:
        total,indexed,missing=await loop.run_in_executor(None,count_copy_index_status)
        if missing > 0:
            app.job_queue.run_repeating(
                _periodic_copy_reindex, interval=COPY_REINDEX_INTERVAL_SEC, first=12, name="v19_reindex_sscd_copy_ai",
            )
            log.info("SSCD AI待补 %d 张：已启动后台任务，每 %d 秒最多 %d 张；补完自动停止",missing,COPY_REINDEX_INTERVAL_SEC,COPY_REINDEX_BATCH_SIZE)
        else:
            log.info("SSCD AI索引已完成 %d/%d：不启动后台任务；新客户入库即时生成特征",indexed,total)
    except Exception:
        log.warning("注册 V1.9 SSCD AI索引升级任务失败（不影响实时检测）",exc_info=True)

    try:
        app.job_queue.run_repeating(
            _periodic_cleanup_orphan_images,
            interval=6*3600,
            first=3600,
            name="v18_cleanup_orphan_images",
        )
    except Exception:
        log.warning("注册临时图片清理任务失败（不影响实时检测）",exc_info=True)


async def _error(update, context):
    log.exception("Telegram处理异常", exc_info=context.error)


def start():
    if not BOT_TOKEN:
        raise RuntimeError("缺少 BOT_TOKEN，请在环境变量中配置")

    app = (
        Application.builder()
        .token(BOT_TOKEN)
        .post_init(_post_init)
        .concurrent_updates(True)
        .build()
    )

    # ── 仅群聊命令 ──────────────────────────────────────────────────────────
    app.add_handler(CommandHandler("start", start_command, filters=GROUPS_ONLY))
    app.add_handler(CommandHandler("help", help_command, filters=GROUPS_ONLY))
    app.add_handler(CommandHandler("stats", stats_command, filters=GROUPS_ONLY))
    app.add_handler(CommandHandler("aistatus", ai_status_command, filters=GROUPS_ONLY))
    app.add_handler(CommandHandler("recover_images", recover_images_command, filters=GROUPS_ONLY))

    # /edit、/cancel 以及私聊文字编辑流程已取消。

    # ── 仅群聊消息 ──────────────────────────────────────────────────────────
    app.add_handler(MessageHandler(GROUPS_ONLY & filters.PHOTO, photo_handler))
    app.add_handler(MessageHandler(GROUPS_ONLY & filters.Document.ALL, document_handler))
    app.add_handler(MessageHandler(GROUPS_ONLY & filters.TEXT & ~filters.COMMAND, text_handler))

    # 撞客人工确认按钮保留；handler 内部再次校验必须来自群聊。
    app.add_handler(CallbackQueryHandler(collision_callback, pattern=r"^collision:"))

    app.add_error_handler(_error)
    print("TG防撞客机器人 V1.9.5 REVIEW-FIX CAPTION-ENTRY MATCH-PREVIEW SAFE-MATCH SSCD-AI COPY-GUARD AUTO90 GitHub + Railway 启动（群聊专用 / 私聊关闭）")
    app.run_polling(
        drop_pending_updates=False,
        allowed_updates=["message", "callback_query"],
    )
