from src.validator import validate_email


def test_accepts_basic_valid_email():
    assert validate_email("dev@example.com") is True


def test_rejects_missing_at_symbol():
    assert validate_email("dev.example.com") is False


def test_rejects_empty_or_blank_strings():
    assert validate_email("") is False
    assert validate_email("   ") is False


def test_rejects_missing_local_or_domain_parts():
    assert validate_email("@example.com") is False
    assert validate_email("dev@") is False
