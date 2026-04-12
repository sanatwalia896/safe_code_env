from __future__ import annotations

FORBIDDEN_GIT_VERBS = {
    "reset",
    "restore",
    "push",
    "rebase",
    "clean",
    "cherry-pick",
}
ALLOWED_GIT_VERBS = {"status", "diff", "log", "branch", "checkout", "merge", "add", "commit"}


def is_command_safe(parts: list[str]) -> bool:
    if not parts:
        return False

    cmd = parts[0]
    if cmd in {"pytest", "ls", "pwd"}:
        return True

    if cmd in {"python", "python3"}:
        return len(parts) >= 3 and parts[1] == "-m" and parts[2] == "pytest"

    if cmd == "git":
        if len(parts) < 2:
            return False
        verb = parts[1]
        if verb in FORBIDDEN_GIT_VERBS:
            return False
        if verb not in ALLOWED_GIT_VERBS:
            return False
        return "--hard" not in parts

    return False
