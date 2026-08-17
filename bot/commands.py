"""
/start /help /stats 命令处理
"""
import logging
from telegram import Update
from telegram.ext import ContextTypes
from core.database import get_conn
from core.image_match import count_copy_index_status
from config import SSCD_MODEL_PATH, SSCD_ENABLED

log = logging.getLogger(__name__)


_HELP = """
🤖 *TG防撞客机器人 V1.9.1 SAFE-MATCH SSCD-AI*

⚠️ *仅在群聊 / 超级群中工作，私聊模式已关闭。*

*基本用法*
• 发一张图片（无说明）→ 查询是否撞客，不入库
• 发图片 \\+ 客户资料 caption → 检测后正式入库

*客户资料格式（繁简均可）*
```
姓名：张三
年龄：28
职业：文员
收入：5k
引流软件：小红书
接粉人员：阿林
```

*批量导入历史记录*
把 Telegram Desktop 导出的 `.zip`、`result.json` 或 `messages.html` 直接发到群里，机器人自动扫描导入。

*管理员命令*
• `/stats` — 查看客户库统计
• `/recover_images` — 恢复发布前已入库但图片丢失的历史记录
• `/aistatus` — 查看 SSCD 同图 AI 索引进度
• `/help` — 显示本帮助
""".strip()


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 我是*防撞客机器人*，发送客户图片即可检测是否撞客。\n\n"
        "发送 /help 查看详细使用说明。",
        parse_mode="Markdown",
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(_HELP, parse_mode="Markdown")


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        conn = get_conn()
        customers = conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
        images = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
        collisions = conn.execute(
            "SELECT COUNT(*) FROM collision_records WHERE status='confirmed'"
        ).fetchone()[0]
        pending = conn.execute(
            "SELECT COUNT(*) FROM collision_records WHERE status='pending'"
        ).fetchone()[0]
        conn.close()
        try:
            _, ai_indexed, ai_missing = count_copy_index_status()
        except Exception:
            ai_indexed, ai_missing = 0, images

        text = (
            "📊 *客户库统计*\n\n"
            f"👥 客户总数：{customers:,}\n"
            f"🖼 图片总数：{images:,}\n"
            f"🧠 SSCD AI已索引：{ai_indexed:,}（待补：{ai_missing:,}）\n"
            f"🔴 已确认撞客：{collisions:,}\n"
            f"🟠 待确认撞客：{pending:,}"
        )
        await update.message.reply_text(text, parse_mode="Markdown")
    except Exception:
        log.exception("stats 查询失败")
        await update.message.reply_text("统计查询失败，请稍后重试")


async def ai_status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        total,indexed,missing=count_copy_index_status()
        model_ok=bool(SSCD_ENABLED and SSCD_MODEL_PATH.exists() and SSCD_MODEL_PATH.stat().st_size>80_000_000)
        pct=(indexed/total*100.0) if total else 100.0
        text=(
            "🧠 *SSCD 同图 AI 状态*\n\n"
            f"模型：{'✅ 已就绪' if model_ok else '⏳ 未就绪/使用备用算法'}\n"
            f"图库图片：{total:,}\n"
            f"AI已索引：{indexed:,}\n"
            f"待补索引：{missing:,}\n"
            f"完成度：{pct:.1f}%"
        )
        await update.message.reply_text(text,parse_mode="Markdown")
    except Exception:
        log.exception("aistatus 查询失败")
        await update.message.reply_text("AI状态查询失败，请稍后重试")
