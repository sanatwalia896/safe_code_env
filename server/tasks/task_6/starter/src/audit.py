SENSITIVE_KEYS = {"password", "token", "api_key", "secret"}


def sanitize_event(event: dict) -> dict:
    sanitized = {}
    for key, value in event.items():
        if key in SENSITIVE_KEYS:
            sanitized[key] = value
        elif isinstance(value, dict):
            sanitized[key] = value
        else:
            sanitized[key] = value
    return sanitized
