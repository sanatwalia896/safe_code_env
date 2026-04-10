"""
Workspace-backed task catalog and grader for Safe Code Env.

The grader evaluates the actual filesystem state of the seeded workspace rather
than the text of a submitted action.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


TASKS_ROOT = Path(__file__).resolve().parent / "tasks"


@dataclass(frozen=True)
class TaskDefinition:
    task_id: str
    title: str
    description: str
    starter_dir: Path
    test_command: List[str]
    required_files: List[str]
    allowed_commands: List[str]


@dataclass
class GradeResult:
    reward: float
    feedback: str
    success: bool
    done: bool
    exit_code: int
    stdout: str
    stderr: str
    passed_tests: int
    failed_tests: int


TASKS: Dict[str, TaskDefinition] = {
    "task_1": TaskDefinition(
        task_id="task_1",
        title="Add /health endpoint",
        description=(
            "Task 1: You are working in a small Flask service.\n"
            "The workspace contains a real `app.py` and pytest suite.\n"
            "Fix the codebase so `/health` returns JSON `{\"status\": \"ok\"}` without breaking `/`.\n"
            "Use the workspace tools to inspect files, edit them, run tests, and submit when ready."
        ),
        starter_dir=TASKS_ROOT / "task_1" / "starter",
        test_command=["pytest", "-q"],
        required_files=["app.py"],
        allowed_commands=["pytest", "python", "python3", "ls", "pwd"],
    ),
    "task_2": TaskDefinition(
        task_id="task_2",
        title="Fix SQL injection",
        description=(
            "Task 2: The workspace contains vulnerable database helper code in `db_utils.py`.\n"
            "Repair it so `get_user(cursor, username)` uses a parameterized query rather than string concatenation.\n"
            "Run the provided tests and submit once the workspace is fixed."
        ),
        starter_dir=TASKS_ROOT / "task_2" / "starter",
        test_command=["pytest", "-q"],
        required_files=["db_utils.py"],
        allowed_commands=["pytest", "python", "python3", "ls", "pwd"],
    ),
    "task_3": TaskDefinition(
        task_id="task_3",
        title="Repair validator module",
        description=(
            "Task 3: The workspace has a broken email validator in `src/validator.py`.\n"
            "Make the implementation satisfy the real pytest suite. The tests cover empty strings,\n"
            "missing `@`, and malformed local/domain parts.\n"
            "Iterate with the file and command tools, then submit the final state."
        ),
        starter_dir=TASKS_ROOT / "task_3" / "starter",
        test_command=["pytest", "-q"],
        required_files=["src/validator.py"],
        allowed_commands=["pytest", "python", "python3", "ls", "pwd"],
    ),
    "task_4": TaskDefinition(
        task_id="task_4",
        title="Repair rate limiter",
        description=(
            "Task 4: The workspace contains a rate limiter implementation in `src/rate_limit.py`.\n"
            "It should allow up to `limit` requests within a rolling time window and reject the next one.\n"
            "Fix the implementation so the provided tests pass without changing the tests."
        ),
        starter_dir=TASKS_ROOT / "task_4" / "starter",
        test_command=["pytest", "-q"],
        required_files=["src/rate_limit.py"],
        allowed_commands=["pytest", "python", "python3", "ls", "pwd"],
    ),
    "task_5": TaskDefinition(
        task_id="task_5",
        title="Repair config loader",
        description=(
            "Task 5: The workspace contains a JSON config loader in `src/config_loader.py`.\n"
            "It must merge defaults with the file contents and surface invalid JSON as a ValueError.\n"
            "Repair the implementation and use the test suite to validate the final state."
        ),
        starter_dir=TASKS_ROOT / "task_5" / "starter",
        test_command=["pytest", "-q"],
        required_files=["src/config_loader.py"],
        allowed_commands=["pytest", "python", "python3", "ls", "pwd"],
    ),
    "task_6": TaskDefinition(
        task_id="task_6",
        title="Repair audit redaction",
        description=(
            "Task 6: The workspace logs audit events in `src/audit.py`.\n"
            "Repair `sanitize_event` so secrets are redacted recursively while preserving safe fields.\n"
            "Use the tests, inspect diffs, and submit the workspace when all tests pass."
        ),
        starter_dir=TASKS_ROOT / "task_6" / "starter",
        test_command=["pytest", "-q"],
        required_files=["src/audit.py"],
        allowed_commands=["pytest", "python", "python3", "ls", "pwd"],
    ),
}


class WorkspaceGrader:
    """Seeds starter workspaces and evaluates them via their real tests."""

    def __init__(self) -> None:
        self.feedback = ""

    def seed_workspace(self, task_id: str, workspace_path: Path) -> TaskDefinition:
        task = TASKS[task_id]
        if workspace_path.exists():
            shutil.rmtree(workspace_path)
        shutil.copytree(
            task.starter_dir,
            workspace_path,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        (workspace_path / "TASK.md").write_text(task.description + "\n", encoding="utf-8")
        return task

    def evaluate_workspace(self, task_id: str, workspace_path: Path, *, final: bool) -> GradeResult:
        task = TASKS[task_id]
        for relative_path in task.required_files:
            if not (workspace_path / relative_path).exists():
                feedback = f"Missing required file: {relative_path}"
                return GradeResult(
                    reward=0.0,
                    feedback=feedback,
                    success=False,
                    done=final,
                    exit_code=1,
                    stdout="",
                    stderr=feedback,
                    passed_tests=0,
                    failed_tests=1,
                )

        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{workspace_path}{os.pathsep}{existing_pythonpath}"
            if existing_pythonpath
            else str(workspace_path)
        )
        env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"

        try:
            result = subprocess.run(
                task.test_command,
                cwd=str(workspace_path),
                capture_output=True,
                text=True,
                timeout=20,
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            feedback = "Test command timed out."
            return GradeResult(
                reward=0.0,
                feedback=feedback,
                success=False,
                done=final,
                exit_code=1,
                stdout=(exc.stdout or ""),
                stderr=(exc.stderr or "") + "\nTimed out after 20 seconds.",
                passed_tests=0,
                failed_tests=1,
            )

        passed, failed = self._parse_pytest_summary(result.stdout + "\n" + result.stderr)
        reward = self._reward_from_tests(result.returncode, passed, failed, final=final)

        if result.returncode == 0:
            feedback = "All tests passed."
        elif passed or failed:
            feedback = f"Tests progress: {passed} passed, {failed} failed."
        else:
            feedback = "Tests did not complete successfully."

        self.feedback = feedback
        return GradeResult(
            reward=reward,
            feedback=feedback,
            success=result.returncode == 0,
            done=final,
            exit_code=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            passed_tests=passed,
            failed_tests=failed,
        )

    def _parse_pytest_summary(self, text: str) -> tuple[int, int]:
        passed = 0
        failed = 0

        passed_match = re.search(r"(\d+)\s+passed", text)
        failed_match = re.search(r"(\d+)\s+failed", text)
        error_match = re.search(r"(\d+)\s+error", text)
        errors_match = re.search(r"(\d+)\s+errors", text)

        if passed_match:
            passed = int(passed_match.group(1))
        if failed_match:
            failed += int(failed_match.group(1))
        if error_match:
            failed += int(error_match.group(1))
        if errors_match:
            failed += int(errors_match.group(1))

        return passed, failed

    def _reward_from_tests(self, exit_code: int, passed: int, failed: int, *, final: bool) -> float:
        total = passed + failed
        if exit_code == 0:
            return 1.0 if final else 0.95
        if total <= 0:
            return 0.0
        progress = passed / total
        return round(min(progress * 0.85, 0.85 if final else 0.8), 3)
