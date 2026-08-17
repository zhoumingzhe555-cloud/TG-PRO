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
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

from config import DATABASE_PATH, DATA_DIR, REQUIRE_PERSISTENT_STORAGE

log = logging.getLogger(__name__)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_db_path() -> Path:
    raw = str(DATABASE_PATH or "").strip()
    path = Path(raw) if raw else Path("data/customers.db")

    # Railway production safety: never silently fall back to an ephemeral DB.
    # If the /data Volume is missing/mis-mounted, refusing to start is safer than
    # telling staff every existing customer is "new".
    if REQUIRE_PERSISTENT_STORAGE:
        mount_raw = os.getenv("RAILWAY_VOLUME_MOUNT_PATH", "").strip()
        if os.getenv("RAILWAY_ENVIRONMENT") and not mount_raw:
            raise RuntimeError(
                "Railway Volume 未挂载：缺少 RAILWAY_VOLUME_MOUNT_PATH。"
                "请把 Volume Mount Path 设置为 /data 后再启动。"
            )
        if mount_raw:
            mount = Path(mount_raw).resolve()
            try:
                target = path.resolve()
                if mount != target and mount not in target.parents:
                    raise RuntimeError(
                        f"数据库路径 {target} 不在 Railway Volume {mount} 内。"
                        "请设置 DATA_DIR=/data。"
                    )
            except FileNotFoundError:
                pass

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        probe = path.parent / ".db_write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return path
    except Exception as exc:
        if REQUIRE_PERSISTENT_STORAGE:
            raise RuntimeError(f"持久化数据库目录不可写：{path.parent}: {exc}") from exc
        fallback = Path("data/customers.db")
        fallback.parent.mkdir(parents=True, exist_ok=True)
        log.warning("数据库目录不可写 %s，开发环境回退到 %s：%s", path.parent, fallback, exc)
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
            created_time TEXT,
            trusted INTEGER DEFAULT 1,
            parent_image_id INTEGER,
            match_evidence TEXT
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
            "source":"TEXT DEFAULT 'live'", "created_time":"TEXT", "trusted":"INTEGER DEFAULT 1",
            "parent_image_id":"INTEGER", "match_evidence":"TEXT",
        })
        # V1.9.1 SAFE: old automatically learned aliases may have been created
        # by permissive AUTO90 logic. Quarantine them instead of deleting data.
        # New aliases created by V1.9.1 are explicitly re-trusted after strict verification.
        conn.execute("UPDATE images SET trusted=0 WHERE source='auto_alias' AND COALESCE(parent_image_id,0)=0")

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

        _ensure_columns(conn, "collision_records", {
            "query_phash":"TEXT", "query_file_path":"TEXT", "query_file_id":"TEXT", "query_file_unique_id":"TEXT",
            "query_copy_feature":"BLOB", "query_copy_dim":"INTEGER DEFAULT 0",
            "query_customer_json":"TEXT", "query_raw_text":"TEXT", "query_source_message_id":"TEXT",
        })

        conn.execute("""
        CREATE TABLE IF NOT EXISTS image_signatures(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER NOT NULL,
            kind TEXT NOT NULL,
            phash TEXT, dhash TEXT,
            b0 TEXT, b1 TEXT, b2 TEXT, b3 TEXT, b4 TEXT, b5 TEXT, b6 TEXT, b7 TEXT,
            created_time TEXT,
            UNIQUE(image_id, kind)
        )
        """)

        conn.execute("""
        CREATE TABLE IF NOT EXISTS image_local_features(
            image_id INTEGER PRIMARY KEY,
            sift_kp BLOB, sift_desc BLOB, sift_rows INTEGER DEFAULT 0, sift_cols INTEGER DEFAULT 0, sift_dtype TEXT,
            akaze_kp BLOB, akaze_desc BLOB, akaze_rows INTEGER DEFAULT 0, akaze_cols INTEGER DEFAULT 0, akaze_dtype TEXT,
            updated_time TEXT
        )
        """)

        conn.execute("""
        CREATE TABLE IF NOT EXISTS image_copy_features(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER NOT NULL,
            kind TEXT NOT NULL,
            feature BLOB NOT NULL,
            dim INTEGER NOT NULL,
            model_name TEXT,
            simhash TEXT,
            created_time TEXT,
            UNIQUE(image_id, kind)
        )
        """)

        conn.execute("""
        CREATE TABLE IF NOT EXISTS image_copy_lsh(
            image_id INTEGER NOT NULL,
            kind TEXT NOT NULL,
            band INTEGER NOT NULL,
            value TEXT NOT NULL,
            PRIMARY KEY(image_id, kind, band)
        )
        """)

        conn.execute("""
        CREATE TABLE IF NOT EXISTS false_positive_pairs(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_sha256 TEXT NOT NULL,
            query_phash TEXT,
            matched_image_id INTEGER NOT NULL,
            confirmer_id TEXT,
            created_time TEXT,
            UNIQUE(query_sha256, matched_image_id)
        )
        """)
        _ensure_columns(conn, "false_positive_pairs", {
            "matched_customer_id":"INTEGER",
            "query_copy_feature":"BLOB",
            "query_copy_dim":"INTEGER DEFAULT 0",
        })

        conn.execute("""
        CREATE TABLE IF NOT EXISTS customer_profile_conflicts(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id INTEGER NOT NULL,
            image_id INTEGER,
            incoming_raw_text TEXT,
            conflict_json TEXT,
            source TEXT,
            created_time TEXT
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

        # V1.9.3: 图片缓冲允许同一员工连续发送多张图片。旧版按 user_id 唯一会覆盖前一张，
        # 导致稍后回复较早图片时重启后无法配对。迁移为 message_id 唯一。
        conn.execute("DROP INDEX IF EXISTS idx_pending_buffer_key")

        index_sql = [
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_images_sha256 ON images(sha256) WHERE sha256 IS NOT NULL",
            "CREATE INDEX IF NOT EXISTS idx_images_md5 ON images(md5)",
            "CREATE INDEX IF NOT EXISTS idx_images_customer ON images(customer_id)",
            "CREATE INDEX IF NOT EXISTS idx_images_trusted ON images(trusted)",
            "CREATE INDEX IF NOT EXISTS idx_images_created ON images(created_time)",
            "CREATE INDEX IF NOT EXISTS idx_collision_status ON collision_records(status)",
            "CREATE INDEX IF NOT EXISTS idx_edit_log_customer ON customer_edit_log(customer_id)",
            "CREATE INDEX IF NOT EXISTS idx_edit_log_changed_time ON customer_edit_log(changed_time)",
            "CREATE INDEX IF NOT EXISTS idx_customers_created_time ON customers(created_time)",
            "CREATE INDEX IF NOT EXISTS idx_pending_buffer_user ON pending_buffer(buffer_type, chat_id, user_id, ts)",
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_pending_buffer_message ON pending_buffer(buffer_type, chat_id, message_id) WHERE message_id IS NOT NULL",
            "CREATE INDEX IF NOT EXISTS idx_crash_logs_created ON crash_logs(created_time)",
            "CREATE INDEX IF NOT EXISTS idx_sig_image ON image_signatures(image_id)",
            "CREATE INDEX IF NOT EXISTS idx_false_query ON false_positive_pairs(query_sha256)",
            "CREATE INDEX IF NOT EXISTS idx_false_match ON false_positive_pairs(matched_image_id)",
            "CREATE INDEX IF NOT EXISTS idx_profile_conflict_customer ON customer_profile_conflicts(customer_id)",
            "CREATE INDEX IF NOT EXISTS idx_copy_features_image ON image_copy_features(image_id)",
            "CREATE INDEX IF NOT EXISTS idx_copy_features_simhash ON image_copy_features(simhash)",
            "CREATE INDEX IF NOT EXISTS idx_copy_lsh_band_value ON image_copy_lsh(band,value)",
            "CREATE INDEX IF NOT EXISTS idx_false_customer ON false_positive_pairs(matched_customer_id)",
        ]
        for i in range(8):
            index_sql.append(f"CREATE INDEX IF NOT EXISTS idx_sig_b{i} ON image_signatures(b{i})")
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
    info_json = json.dumps(
        info, ensure_ascii=False,
        default=lambda o: sorted(o) if isinstance(o, set) else str(o),
    ) if info is not None else None
    conn = get_conn()
    try:
        # 文字资料只保留同一员工最新一份；图片则按 message_id 分别保留，
        # 这样同一员工连续发多张图后，仍可通过回复某张旧图精确补资料。
        if str(buffer_type) == "text" or message_id is None:
            conn.execute(
                "DELETE FROM pending_buffer WHERE buffer_type=? AND chat_id=? AND user_id=?",
                (str(buffer_type), str(chat_id), str(user_id)),
            )
        else:
            conn.execute(
                "DELETE FROM pending_buffer WHERE buffer_type=? AND chat_id=? AND message_id=?",
                (str(buffer_type), str(chat_id), str(message_id)),
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


def delete_pending_buffer(buffer_type: str, chat_id: str, user_id: str, message_id=None) -> None:
    conn = get_conn()
    try:
        if message_id is not None:
            conn.execute(
                "DELETE FROM pending_buffer WHERE buffer_type=? AND chat_id=? AND message_id=?",
                (str(buffer_type), str(chat_id), str(message_id)),
            )
        else:
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
