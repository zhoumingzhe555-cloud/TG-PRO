"""Railway persistent SQLite backend.

Railway deployment:
  - mount one Railway Volume at /data
  - default DB: /data/customers.db

All callers use sqlite-style `?` placeholders and Row access, so no PostgreSQL
compatibility layer is needed on Railway.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

from config import DATABASE_PATH

log = logging.getLogger(__name__)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_db_path() -> Path:
    raw = str(DATABASE_PATH or "").strip()
    path = Path(raw) if raw else Path("data/customers.db")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        probe = path.parent / ".db_write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return path
    except Exception as exc:
        fallback = Path("data/customers.db")
        fallback.parent.mkdir(parents=True, exist_ok=True)
        log.warning("数据库目录不可写 %s，自动回退到 %s：%s", path.parent, fallback, exc)
        return fallback


def get_conn() -> sqlite3.Connection:
    path = _resolve_db_path()
    conn = sqlite3.connect(str(path), timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA temp_store=MEMORY")
    except Exception:
        pass
    return conn


def _existing_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _ensure_columns(conn: sqlite3.Connection, table: str, mapping: dict[str, str]) -> None:
    cols = _existing_columns(conn, table)
    for name, ddl in mapping.items():
        if name not in cols:
            conn.execute(f'ALTER TABLE {table} ADD COLUMN {name} {ddl}')


def init_db() -> None:
    path = _resolve_db_path()
    conn = get_conn()
    try:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS images(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id TEXT,
            file_unique_id TEXT,
            file_path TEXT,
            sha256 TEXT,
            md5 TEXT,
            phash TEXT,
            dhash TEXT,
            roi_phash TEXT,
            p0 TEXT, p1 TEXT, p2 TEXT, p3 TEXT,
            p4 TEXT, p5 TEXT, p6 TEXT, p7 TEXT,
            color_hist BLOB,
            orb_desc BLOB,
            orb_rows INTEGER DEFAULT 0,
            ai_feature BLOB,
            ai_hash TEXT,
            a0 TEXT, a1 TEXT, a2 TEXT, a3 TEXT,
            a4 TEXT, a5 TEXT, a6 TEXT, a7 TEXT,
            customer_id INTEGER,
            submitter TEXT,
            submitter_id TEXT,
            chat_id TEXT,
            source TEXT DEFAULT 'live',
            created_time TEXT
        )
        """)
        _ensure_columns(conn, "images", {
            "file_id":"TEXT", "file_unique_id":"TEXT", "file_path":"TEXT",
            "sha256":"TEXT", "md5":"TEXT", "phash":"TEXT", "dhash":"TEXT", "roi_phash":"TEXT",
            "p0":"TEXT", "p1":"TEXT", "p2":"TEXT", "p3":"TEXT", "p4":"TEXT", "p5":"TEXT", "p6":"TEXT", "p7":"TEXT",
            "color_hist":"BLOB", "orb_desc":"BLOB", "orb_rows":"INTEGER DEFAULT 0",
            "ai_feature":"BLOB", "ai_hash":"TEXT",
            "a0":"TEXT", "a1":"TEXT", "a2":"TEXT", "a3":"TEXT", "a4":"TEXT", "a5":"TEXT", "a6":"TEXT", "a7":"TEXT",
            "customer_id":"INTEGER", "submitter":"TEXT", "submitter_id":"TEXT", "chat_id":"TEXT",
            "source":"TEXT DEFAULT 'live'", "created_time":"TEXT",
        })

        conn.execute("""
        CREATE TABLE IF NOT EXISTS customers(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT, age TEXT, job TEXT, income TEXT, work_year TEXT,
            software TEXT, receiver TEXT, raw_text TEXT, submitter TEXT,
            submitter_id TEXT, chat_id TEXT, source TEXT DEFAULT 'live',
            source_message_id TEXT, created_time TEXT,
            updated_time TEXT, last_updated_by TEXT
        )
        """)
        _ensure_columns(conn, "customers", {
            "age":"TEXT", "job":"TEXT", "income":"TEXT", "work_year":"TEXT", "software":"TEXT", "receiver":"TEXT",
            "raw_text":"TEXT", "submitter":"TEXT", "submitter_id":"TEXT", "chat_id":"TEXT", "source":"TEXT DEFAULT 'live'",
            "source_message_id":"TEXT", "created_time":"TEXT", "updated_time":"TEXT", "last_updated_by":"TEXT",
        })

        conn.execute("""
        CREATE TABLE IF NOT EXISTS collision_records(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_sha256 TEXT,
            matched_image_id INTEGER,
            query_submitter TEXT,
            query_submitter_id TEXT,
            chat_id TEXT,
            match_type TEXT,
            score REAL,
            status TEXT DEFAULT 'pending',
            confirmer TEXT,
            confirmer_id TEXT,
            created_time TEXT,
            confirmed_time TEXT
        )
        """)

        conn.execute("""
        CREATE TABLE IF NOT EXISTS imports(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT,
            scanned_messages INTEGER DEFAULT 0,
            customers_found INTEGER DEFAULT 0,
            images_imported INTEGER DEFAULT 0,
            skipped INTEGER DEFAULT 0,
            status TEXT,
            created_time TEXT
        )
        """)

        conn.execute("""
        CREATE TABLE IF NOT EXISTS customer_edit_log(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id INTEGER NOT NULL,
            field_name TEXT NOT NULL,
            old_value TEXT,
            new_value TEXT,
            operator TEXT,
            operator_id TEXT,
            changed_time TEXT NOT NULL
        )
        """)

        conn.execute("""
        CREATE TABLE IF NOT EXISTS pending_buffer(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            buffer_type TEXT NOT NULL,
            chat_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            ts REAL NOT NULL,
            file_id TEXT,
            file_unique_id TEXT,
            file_path TEXT,
            obj_key TEXT,
            message_id TEXT,
            user_name TEXT,
            raw_text TEXT,
            info_json TEXT,
            created_time TEXT
        )
        """)

        conn.execute("""
        CREATE TABLE IF NOT EXISTS crash_logs(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            exc_type TEXT,
            summary TEXT,
            traceback TEXT,
            created_time TEXT
        )
        """)

        index_sql = [
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_images_sha256 ON images(sha256) WHERE sha256 IS NOT NULL",
            "CREATE INDEX IF NOT EXISTS idx_images_md5 ON images(md5)",
            "CREATE INDEX IF NOT EXISTS idx_images_customer ON images(customer_id)",
            "CREATE INDEX IF NOT EXISTS idx_images_created ON images(created_time)",
            "CREATE INDEX IF NOT EXISTS idx_collision_status ON collision_records(status)",
            "CREATE INDEX IF NOT EXISTS idx_edit_log_customer ON customer_edit_log(customer_id)",
            "CREATE INDEX IF NOT EXISTS idx_edit_log_changed_time ON customer_edit_log(changed_time)",
            "CREATE INDEX IF NOT EXISTS idx_customers_created_time ON customers(created_time)",
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_pending_buffer_key ON pending_buffer(buffer_type, chat_id, user_id)",
            "CREATE INDEX IF NOT EXISTS idx_crash_logs_created ON crash_logs(created_time)",
        ]
        for i in range(8):
            index_sql.append(f"CREATE INDEX IF NOT EXISTS idx_images_p{i} ON images(p{i})")
            index_sql.append(f"CREATE INDEX IF NOT EXISTS idx_images_a{i} ON images(a{i})")
        for sql in index_sql:
            conn.execute(sql)

        conn.commit()
        print(f"数据库初始化完成（Railway SQLite）：{path}", flush=True)
    finally:
        conn.close()


def save_crash_log(exc_type: str, summary: str, traceback_text: str) -> None:
    try:
        conn = get_conn()
        try:
            conn.execute(
                "INSERT INTO crash_logs(exc_type,summary,traceback,created_time) VALUES(?,?,?,?)",
                (exc_type, summary[:500], traceback_text[:4000], now_iso()),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception as exc:
        log.error("写入崩溃日志失败（不影响主程序）：%s", exc)


def save_pending_buffer(
    buffer_type: str,
    chat_id: str,
    user_id: str,
    ts: float,
    *,
    file_id: str | None = None,
    file_unique_id: str | None = None,
    file_path: str | None = None,
    obj_key: str | None = None,
    message_id=None,
    user_name: str | None = None,
    raw_text: str | None = None,
    info: dict | None = None,
) -> None:
    info_json = json.dumps(info, ensure_ascii=False) if info is not None else None
    conn = get_conn()
    try:
        conn.execute(
            "DELETE FROM pending_buffer WHERE buffer_type=? AND chat_id=? AND user_id=?",
            (str(buffer_type), str(chat_id), str(user_id)),
        )
        conn.execute(
            """INSERT INTO pending_buffer
               (buffer_type,chat_id,user_id,ts,file_id,file_unique_id,file_path,obj_key,message_id,user_name,raw_text,info_json,created_time)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                str(buffer_type), str(chat_id), str(user_id), float(ts), file_id, file_unique_id,
                file_path, obj_key, str(message_id) if message_id is not None else None,
                user_name, raw_text, info_json, now_iso(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def delete_pending_buffer(buffer_type: str, chat_id: str, user_id: str) -> None:
    conn = get_conn()
    try:
        conn.execute(
            "DELETE FROM pending_buffer WHERE buffer_type=? AND chat_id=? AND user_id=?",
            (str(buffer_type), str(chat_id), str(user_id)),
        )
        conn.commit()
    finally:
        conn.close()


def cleanup_expired_pending_buffers(max_age_sec: float) -> int:
    cutoff = time.time() - float(max_age_sec)
    conn = get_conn()
    try:
        cur = conn.execute("DELETE FROM pending_buffer WHERE ts < ?", (cutoff,))
        conn.commit()
        return int(cur.rowcount or 0)
    finally:
        conn.close()


def load_pending_buffers() -> list:
    conn = get_conn()
    try:
        rows = conn.execute("SELECT * FROM pending_buffer ORDER BY ts ASC").fetchall()
    finally:
        conn.close()
    result = []
    for row in rows:
        entry = dict(row)
        raw = entry.get("info_json")
        if raw:
            try:
                entry["info"] = json.loads(raw)
            except Exception:
                entry["info"] = {}
        else:
            entry["info"] = {}
        result.append(entry)
    return result
