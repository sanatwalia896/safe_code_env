from __future__ import annotations

from sqlite3 import Connection


def find_user_by_email(conn: Connection, email: str):
    query = f"SELECT id, email, display_name FROM users WHERE email = '{email}'"
    return conn.execute(query).fetchone()


def insert_user(conn: Connection, email: str, display_name: str):
    conn.execute(
        "INSERT INTO users (email, display_name) VALUES (?, ?)",
        (email, display_name),
    )
    conn.commit()
    return find_user_by_email(conn, email)
