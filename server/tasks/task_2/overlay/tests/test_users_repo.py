from src.db.sqlite_db import get_conn, init_db
from src.repos.users_repo import find_user_by_email, insert_user


def test_find_user_uses_parameterized_query(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    init_db()
    with get_conn() as conn:
        insert_user(conn, "dev@example.com", "Dev")
        row = find_user_by_email(conn, "dev@example.com")
        assert row["email"] == "dev@example.com"


def test_sql_injection_payload_does_not_match_every_row(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    init_db()
    with get_conn() as conn:
        insert_user(conn, "alice@example.com", "Alice")
        insert_user(conn, "bob@example.com", "Bob")
        payload = "' OR 1=1 --"
        row = find_user_by_email(conn, payload)
        assert row is None
