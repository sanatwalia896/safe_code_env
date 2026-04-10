from src.audit import sanitize_event


def test_redacts_top_level_secrets():
    event = {
        "actor": "service-a",
        "password": "plain-text",
        "token": "abc123",
    }

    assert sanitize_event(event) == {
        "actor": "service-a",
        "password": "***REDACTED***",
        "token": "***REDACTED***",
    }


def test_redacts_nested_secret_values():
    event = {
        "actor": "service-a",
        "metadata": {
            "api_key": "secret-key",
            "request_id": "req-1",
        },
    }

    assert sanitize_event(event) == {
        "actor": "service-a",
        "metadata": {
            "api_key": "***REDACTED***",
            "request_id": "req-1",
        },
    }


def test_lists_are_sanitized_recursively():
    event = {
        "items": [
            {"secret": "one", "name": "alpha"},
            {"name": "beta", "token": "two"},
        ]
    }

    assert sanitize_event(event) == {
        "items": [
            {"secret": "***REDACTED***", "name": "alpha"},
            {"name": "beta", "token": "***REDACTED***"},
        ]
    }
