import asyncio
import logging
from telegram.ext import Application, MessageHandler, CommandHandler, CallbackQueryHandler, filters
from config import BOT_TOKEN
from bot.handler import (
    photo_handler, document_handler, collision_callback, text_handler,
    restore_pending_buffers, _PENDING_TTL, _PENDING_TEXT_TTL,
)
from bot.commands import start_command, help_command, stats_command
from bot.recover import recover_images_command
from ai.embedding import preload
from core.database import cleanup_expired_pending_buffers

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


async def _post_init(app):
    try:
        await app.bot.delete_webhook(drop_pending_updates=False)
    except Exception:
        pass

    try:
        preload()
    except Exception:
        log.exception("AI模型预加载失败")

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
    app.add_handler(CommandHandler("recover_images", recover_images_command, filters=GROUPS_ONLY))

    # /edit、/cancel 以及私聊文字编辑流程已取消。

    # ── 仅群聊消息 ──────────────────────────────────────────────────────────
    app.add_handler(MessageHandler(GROUPS_ONLY & filters.PHOTO, photo_handler))
    app.add_handler(MessageHandler(GROUPS_ONLY & filters.Document.ALL, document_handler))
    app.add_handler(MessageHandler(GROUPS_ONLY & filters.TEXT & ~filters.COMMAND, text_handler))

    # 撞客人工确认按钮保留；handler 内部再次校验必须来自群聊。
    app.add_handler(CallbackQueryHandler(collision_callback, pattern=r"^collision:"))

    app.add_error_handler(_error)
    print("TG防撞客机器人 V1.7 ONE-PHOTO-ONE-CUSTOMER GitHub + Railway 启动（群聊专用 / 私聊关闭）")
    app.run_polling(
        drop_pending_updates=False,
        allowed_updates=["message", "callback_query"],
    )
