import logging
import os
import sys
import traceback
from core.database import init_db
from bot.app import start

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

log = logging.getLogger(__name__)

# Avoid printing Telegram Bot API URLs (which contain the bot token) in Railway logs.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


def _check_singleton():
    """Railway worker starts normally. Keep one service replica for Telegram polling."""
    return

def _send_crash_alert(exc: BaseException) -> None:
    """向管理员 Telegram chat_id 发送崩溃告警（同步 HTTP，不依赖事件循环）。"""
    from config import ADMIN_CHAT_ID, BOT_TOKEN

    if not ADMIN_CHAT_ID:
        return
    token = BOT_TOKEN
    if not token:
        return

    exc_type = type(exc).__name__
    exc_summary = str(exc)[:300]
    tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
    # 只保留最后 5 行 traceback，避免消息过长
    tb_short = "".join(tb_lines[-5:]).strip()[:600]
    text = (
        f"⚠️ 防撞客机器人已崩溃\n"
        f"类型：{exc_type}\n"
        f"原因：{exc_summary}\n\n"
        f"堆栈（末尾）：\n```\n{tb_short}\n```\n\n"
        f"请检查 Railway Deploy Logs。"
    )

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": ADMIN_CHAT_ID,
        "text": text,
        "parse_mode": "Markdown",
    }

    import urllib.request
    import json as _json
    import time

    data = _json.dumps(payload).encode()
    max_retries = 3
    # 指数退避：每次重试前等待 1s、2s、4s，总退避等待 ≤ 7s
    backoff_delays = [1, 2, 4]

    last_err: Exception | None = None
    for attempt in range(1, max_retries + 2):  # 1 次初始 + 最多 3 次重试 = 4 次
        try:
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10):
                pass
            log.info("崩溃告警已发送给管理员 chat_id=%s", ADMIN_CHAT_ID)
            return
        except Exception as send_err:
            last_err = send_err
            retry_num = attempt - 1  # 已完成的重试次数
            if retry_num < max_retries:
                delay = backoff_delays[retry_num]
                log.info(
                    "发送崩溃告警失败（第 %d 次尝试），%ds 后重试：%s",
                    attempt, delay, send_err,
                )
                time.sleep(delay)

    log.error(
        "发送崩溃告警失败（共尝试 %d 次，重试 %d 次，放弃）：%s",
        max_retries + 1, max_retries, last_err,
    )


if __name__ == "__main__":
    _check_singleton()
    try:
        init_db()
    except Exception as exc:
        # 数据库是防撞客系统的核心状态。尤其在 Railway 上，如果 /data Volume
        # 丢失，继续启动会让所有老客户看起来像“新客户”，因此必须失败关闭。
        log.critical("数据库初始化失败，Bot 拒绝启动：%s", exc, exc_info=True)
        _send_crash_alert(exc)
        sys.exit(1)
    try:
        start()
    except (KeyboardInterrupt, SystemExit):
        # 正常停止，不发告警
        log.info("Bot 已正常停止。")
    except Exception as exc:
        log.critical("Bot 因未捕获异常崩溃：%s", exc, exc_info=True)
        # 写入崩溃历史（无论告警是否配置都执行；失败不阻断退出）
        try:
            from core.database import save_crash_log
            tb_full = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            save_crash_log(type(exc).__name__, str(exc)[:500], tb_full)
        except Exception as _db_err:
            log.error("写入崩溃日志失败：%s", _db_err)
        _send_crash_alert(exc)
        sys.exit(1)
