from __future__ import annotations

from src.db.sqlite_db import get_conn
from src.repos.users_repo import find_user_by_email, insert_user


def register_user(email: str, display_name: str):
    with get_conn() as conn:
        existing = find_user_by_email(conn, email)
        if existing:
            return {"id": existing["id"], "email": existing["email"], "display_name": existing["display_name"]}
        row = insert_user(conn, email, display_name)
        return {"id": row["id"], "email": row["email"], "display_name": row["display_name"]}
