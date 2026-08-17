"""
启动前语法检查脚本

在执行 main.py 之前，对所有 bot 源文件进行 Python 语法检查。
任何 SyntaxError 都会立即打印出错文件名和行号，并拒绝启动，
而不是等到消息到来时才在运行时崩溃。
"""
import glob
import os
import py_compile
import sys

# ── 进程级单例锁 ──────────────────────────────────────────────────────────────
# 同一容器只允许一个 Bot 轮询同一个 token，避免 Telegram 409 Conflict。
# Railway 服务请保持 1 个 replica；此 PID 锁再做一层容器内保护。
_PID_FILE = "/tmp/tg_bot_singleton.pid"


def _acquire_singleton() -> None:
    if os.path.exists(_PID_FILE):
        try:
            with open(_PID_FILE) as _f:
                _old_pid = int(_f.read().strip())
            os.kill(_old_pid, 0)  # 进程存在则不抛异常
            print(
                f"[singleton] Bot 已在运行（pid={_old_pid}），本次启动退出。",
                flush=True,
            )
            sys.exit(0)
        except (ProcessLookupError, ValueError, OSError):
            pass  # PID 文件过期，忽略

    with open(_PID_FILE, "w") as _f:
        _f.write(str(os.getpid()))

    import atexit

    def _release():
        try:
            if os.path.exists(_PID_FILE):
                with open(_PID_FILE) as _f:
                    if _f.read().strip() == str(os.getpid()):
                        os.unlink(_PID_FILE)
        except OSError:
            pass

    atexit.register(_release)


_acquire_singleton()
# ─────────────────────────────────────────────────────────────────────────────


def check_syntax():
    """检查所有 bot 相关 Python 源文件的语法，发现错误时打印中文提示并退出。"""
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 需要检查的目录/文件模式（相对于 tg-bot/）
    patterns = [
        "*.py",
        "bot/*.py",
        "core/*.py",
        "ai/*.py",
        "collision/*.py",
        "customer/*.py",
        "importer/*.py",
        "history_ai/*.py",
        "ocr/*.py",
    ]

    files = []
    for pattern in patterns:
        full_pattern = os.path.join(base_dir, pattern)
        files.extend(sorted(glob.glob(full_pattern)))

    # 去重（保持顺序）
    seen = set()
    unique_files = []
    for f in files:
        if f not in seen:
            seen.add(f)
            unique_files.append(f)

    errors = []
    for filepath in unique_files:
        try:
            py_compile.compile(filepath, doraise=True)
        except py_compile.PyCompileError as exc:
            errors.append((filepath, exc))

    if errors:
        print("=" * 60, file=sys.stderr)
        print("❌  语法检查失败，Bot 拒绝启动！", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        for filepath, exc in errors:
            rel = os.path.relpath(filepath, base_dir)
            print(f"\n  文件：{rel}", file=sys.stderr)
            # PyCompileError.msg 已包含文件名和行号
            print(f"  错误：{exc.msg}", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        print("请修复以上语法错误后重新启动。", file=sys.stderr)
        sys.exit(1)

    print(f"✅  语法检查通过（共 {len(unique_files)} 个文件），正在启动 Bot…")


if __name__ == "__main__":
    check_syntax()

    # 以 __main__ 身份执行 main.py，保证 `if __name__ == "__main__":` 分支正常运行
    import runpy
    runpy.run_path(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.py"),
        run_name="__main__",
    )
