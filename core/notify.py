"""
后台实时通知 — Bot 写入客户数据后，立即通知 API Server 推送 SSE 事件。

设计原则：fire-and-forget。通知失败绝不影响主流程（覆盖/编辑照常完成），
只记 warning 日志。

API Server 地址解析：
1. API_SERVER_URL 环境变量（显式指定，优先）
2. 生产环境：Bot 由 API Server 以子进程启动，继承其 PORT → 127.0.0.1:$PORT
3. 开发环境：走本地共享代理 127.0.0.1:80（/api/* 路由到 API Server）
"""
import asyncio
import json
import logging
import os
import urllib.request

log = logging.getLogger(__name__)

_TIMEOUT = 3  # 秒


def _api_base() -> str:
    explicit = os.getenv("API_SERVER_URL")
    if explicit:
        return explicit.rstrip("/")
    port = os.getenv("PORT")
    if port:
        return f"http://127.0.0.1:{port}"
    return "http://127.0.0.1:80"


def _post_notify(customer_id=None) -> None:
    """同步 HTTP POST（在线程池中执行，不阻塞事件循环）。"""
    secret = os.getenv("SESSION_SECRET")
    if not secret:
        log.debug("SESSION_SECRET 未设置，跳过后台通知")
        return
    url = f"{_api_base()}/api/events/notify"
    body = json.dumps(
        {"event": "customers_updated", "customer_id": customer_id}
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {secret}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        resp.read()


async def notify_dashboard_update(customer_id=None) -> None:
    """异步通知后台客户数据已变更。失败只记日志，不抛异常。"""
    try:
        await asyncio.get_running_loop().run_in_executor(
            None, _post_notify, customer_id
        )
    except Exception:
        log.warning("通知后台刷新失败（不影响主流程）", exc_info=True)
