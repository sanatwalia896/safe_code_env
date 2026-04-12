from __future__ import annotations

import sqlite3
from pathlib import Path

DB_PATH = Path("app.db")


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                display_name TEXT NOT NULL
            )
            """
        )
        conn.commit()


def sqlite_ready() -> bool:
    try:
        with get_conn() as conn:
            conn.execute("SELECT 1")
        return True
    except sqlite3.Error:
        return False
