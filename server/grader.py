"""
Workspace grader for a single shared FastAPI + SQLite codebase.

Each task applies an overlay to the same base repository and is then graded
using real tests against the resulting workspace state.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


SERVER_ROOT = Path(__file__).resolve().parent
BASE_CODEBASE_ROOT = SERVER_ROOT / "base_codebase"
TASKS_ROOT = SERVER_ROOT / "tasks"


@dataclass(frozen=True)
class TaskDefinition:
    task_id: str
    title: str
    description: str
    overlay_dir: Path
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
        title="FastAPI health endpoint quality",
        description=(
            "Single-codebase task 1: fix the FastAPI `/health` endpoint behavior. "
            "It should report status `ok` and include `service` + `sqlite_ready` fields."
        ),
        overlay_dir=TASKS_ROOT / "task_1" / "overlay",
        test_command=["pytest", "-q"],
        required_files=["src/api/health.py"],
        allowed_commands=["pytest", "ls", "pwd", "git"],
    ),
    "task_2": TaskDefinition(
        task_id="task_2",
        title="SQLite query safety",
        description=(
            "Single-codebase task 2: remove SQL injection risk in the SQLite user lookup. "
            "Use parameterized queries only and keep repository API unchanged."
        ),
        overlay_dir=TASKS_ROOT / "task_2" / "overlay",
        test_command=["pytest", "-q"],
        required_files=["src/repos/users_repo.py"],
        allowed_commands=["pytest", "ls", "pwd", "git"],
    ),
    "task_3": TaskDefinition(
        task_id="task_3",
        title="Protected file path guard",
        description=(
            "Single-codebase task 3: harden path safety so protected files such as `.env` "
            "and production SQLite files are blocked."
        ),
        overlay_dir=TASKS_ROOT / "task_3" / "overlay",
        test_command=["pytest", "-q"],
        required_files=["src/security/path_guard.py"],
        allowed_commands=["pytest", "ls", "pwd", "git"],
    ),
    "task_4": TaskDefinition(
        task_id="task_4",
        title="Git command policy guard",
        description=(
            "Single-codebase task 4: enforce local git hygiene policy. "
            "Block dangerous commands like `git reset`, `git restore`, and force-push style flows."
        ),
        overlay_dir=TASKS_ROOT / "task_4" / "overlay",
        test_command=["pytest", "-q"],
        required_files=["src/security/command_guard.py"],
        allowed_commands=["pytest", "ls", "pwd", "git"],
    ),
}


class WorkspaceGrader:
    """Seeds and grades a workspace for a selected task."""

    def __init__(self) -> None:
        self.feedback = ""

    def seed_workspace(self, task_id: str, workspace_path: Path) -> TaskDefinition:
        task = TASKS[task_id]
        if workspace_path.exists():
            shutil.rmtree(workspace_path)

        shutil.copytree(
            BASE_CODEBASE_ROOT,
            workspace_path,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        self._apply_overlay(task.overlay_dir, workspace_path)

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
        python_paths = [str(workspace_path), str(workspace_path / "src")]
        existing_pythonpath = env.get("PYTHONPATH")
        if existing_pythonpath:
            python_paths.append(existing_pythonpath)
        env["PYTHONPATH"] = os.pathsep.join(python_paths)
        env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"

        try:
            result = subprocess.run(
                task.test_command,
                cwd=str(workspace_path),
                capture_output=True,
                text=True,
                timeout=25,
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
                stdout=exc.stdout or "",
                stderr=(exc.stderr or "") + "\nTimed out after 25 seconds.",
                passed_tests=0,
                failed_tests=1,
            )

        passed, failed = self._parse_pytest_summary(result.stdout + "\n" + result.stderr)
        reward = self._reward_from_tests(result.returncode, passed, failed, final=final)

        if result.returncode == 0:
            feedback = "All tests passed."
        elif passed or failed:
            feedback = f"Tests progress: {passed} passed, {failed} failed/errors."
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

    def _apply_overlay(self, overlay_root: Path, workspace_path: Path) -> None:
        if not overlay_root.exists():
            return

        for src in overlay_root.rglob("*"):
            if src.name == "__pycache__" or src.suffix == ".pyc":
                continue
            rel = src.relative_to(overlay_root)
            dst = workspace_path / rel
            if src.is_dir():
                dst.mkdir(parents=True, exist_ok=True)
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

    def _parse_pytest_summary(self, text: str) -> tuple[int, int]:
        passed = 0
        failed = 0
        for count_text, label in re.findall(r"(\d+)\s+(passed|failed|error|errors)\b", text):
            count = int(count_text)
            if label == "passed":
                passed += count
            else:
                failed += count
        return passed, failed

    def _reward_from_tests(self, exit_code: int, passed: int, failed: int, *, final: bool) -> float:
        total = passed + failed
        if exit_code == 0:
            return 1.0 if final else 0.90
        if total <= 0:
            return 0.0
        progress = passed / total
        return round(min(progress * 0.80, 0.80 if final else 0.75), 3)
