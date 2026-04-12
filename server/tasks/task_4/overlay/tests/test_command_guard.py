from src.security.command_guard import is_command_safe


def test_blocks_dangerous_git_commands():
    assert is_command_safe(["git", "reset", "--hard"]) is False
    assert is_command_safe(["git", "restore", "src/api/users.py"]) is False


def test_blocks_arbitrary_python_payloads():
    assert is_command_safe(["python3", "-c", "print('oops')"]) is False
    assert is_command_safe(["python", "script.py"]) is False


def test_allows_safe_local_git_and_pytest_commands():
    assert is_command_safe(["git", "status"]) is True
    assert is_command_safe(["git", "checkout", "-b", "feature/x"]) is True
    assert is_command_safe(["pytest", "-q"]) is True
