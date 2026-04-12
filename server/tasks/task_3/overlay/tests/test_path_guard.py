from src.security.path_guard import is_protected_path


def test_env_file_is_protected():
    assert is_protected_path(".env") is True
    assert is_protected_path("config/.env.local") is True


def test_production_db_is_protected():
    assert is_protected_path("prod.db") is True
    assert is_protected_path("data/production.db") is True


def test_regular_source_file_is_not_protected():
    assert is_protected_path("src/api/users.py") is False
