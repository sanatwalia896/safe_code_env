from pathlib import Path

PROTECTED_PATTERNS = (".git/config",)


def is_protected_path(path: str) -> bool:
    normalized = Path(path).as_posix().lower()
    return any(pattern in normalized for pattern in PROTECTED_PATTERNS)
