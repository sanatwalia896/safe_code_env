# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Safe Code Env Environment Implementation.

This version exposes a realistic workspace with file and command tools instead
of accepting a single-shot code submission.
"""

from __future__ import annotations

import difflib
import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models import SafeCodeAction, SafeCodeObservation, SafeCodeState
from server.grader import TASKS, WorkspaceGrader


TOOL_NAMES = [
    "list_files",
    "read_file",
    "read_files",
    "write_file",
    "search",
    "diff",
    "run_command",
    "submit",
]
MAX_STEPS = 25


class SafeCodeEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._grader = WorkspaceGrader()
        self._tasks = list(TASKS.keys())
        self._task_idx = 0
        self._workspace_path: Path | None = None
        self._starter_snapshot: dict[str, str] = {}
        self._state = SafeCodeState(
            episode_id=str(uuid4()),
            step_count=0,
            task_id="",
            workspace_path="",
            available_tools=TOOL_NAMES.copy(),
        )

    def reset(self) -> SafeCodeObservation:
        self._cleanup_workspace()

        task_id = self._tasks[self._task_idx % len(self._tasks)]
        self._task_idx += 1

        workspace_path = Path(tempfile.mkdtemp(prefix=f"safe_code_env_{task_id}_")).resolve()
        self._workspace_path = workspace_path
        task = self._grader.seed_workspace(task_id, workspace_path)
        self._starter_snapshot = self._snapshot_workspace()

        self._state = SafeCodeState(
            episode_id=str(uuid4()),
            step_count=0,
            task_id=task_id,
            workspace_path=str(workspace_path),
            available_tools=TOOL_NAMES.copy(),
            changed_files=[],
            last_command="",
            last_exit_code=0,
        )

        return SafeCodeObservation(
            success=True,
            output="\n".join(self._list_files(".")),
            error="",
            exit_code=0,
            reward=0.0,
            done=False,
            feedback="Workspace seeded. Inspect the files, run tests, and submit when ready.",
            task_id=task_id,
            task_description=task.description,
            workspace_path=str(workspace_path),
            current_path=".",
            files=self._list_files("."),
            changed_files=[],
            available_tools=TOOL_NAMES.copy(),
            metadata={"task_title": task.title},
        )

    def step(self, action: SafeCodeAction) -> SafeCodeObservation:
        self._ensure_workspace()
        self._state.step_count += 1

        observation = self._dispatch(action)
        observation.changed_files = self._compute_changed_files()
        observation.available_tools = TOOL_NAMES.copy()
        observation.task_id = self._state.task_id
        observation.task_description = TASKS[self._state.task_id].description
        observation.workspace_path = self._state.workspace_path
        observation.metadata = {
            **observation.metadata,
            "step_count": self._state.step_count,
        }

        self._state.changed_files = observation.changed_files
        self._state.last_exit_code = observation.exit_code

        if not observation.done and self._state.step_count >= MAX_STEPS:
            final_result = self._grader.evaluate_workspace(
                self._state.task_id,
                self._workspace_path,
                final=True,
            )
            observation.reward = final_result.reward
            observation.done = True
            observation.feedback = (
                "Step limit reached. Final evaluation executed. "
                + final_result.feedback
            )
            observation.output = final_result.stdout[:4000]
            observation.error = final_result.stderr[:2000]
            observation.exit_code = final_result.exit_code

        return observation

    def _dispatch(self, action: SafeCodeAction) -> SafeCodeObservation:
        if action.action_type == "list_files":
            try:
                files = self._list_files(action.path)
            except (FileNotFoundError, ValueError) as exc:
                return self._error_observation(str(exc), action.path)
            return SafeCodeObservation(
                success=True,
                output="\n".join(files),
                feedback=f"Listed {len(files)} paths under {action.path}.",
                current_path=action.path,
                files=files,
            )

        if action.action_type == "read_file":
            try:
                path = self._resolve_path(action.path)
            except ValueError as exc:
                return self._error_observation(str(exc), action.path)
            if not path.is_file():
                return self._error_observation(f"File not found: {action.path}", action.path)
            content = path.read_text(encoding="utf-8")
            return SafeCodeObservation(
                success=True,
                output=content[:12000],
                feedback=f"Read {action.path}.",
                current_path=action.path,
            )

        if action.action_type == "read_files":
            paths = action.paths or ([] if not action.path else [action.path])
            if not paths:
                return self._error_observation("read_files requires paths", ".")
            chunks = []
            resolved_files = []
            for relative_path in paths[:10]:
                try:
                    path = self._resolve_path(relative_path)
                except ValueError as exc:
                    return self._error_observation(str(exc), relative_path)
                if not path.is_file():
                    return self._error_observation(f"File not found: {relative_path}", relative_path)
                content = path.read_text(encoding="utf-8")
                chunks.append(f"FILE: {relative_path}\n{content}")
                resolved_files.append(relative_path)
            return SafeCodeObservation(
                success=True,
                output="\n\n".join(chunks)[:12000],
                feedback=f"Read {len(resolved_files)} files.",
                current_path=".",
                files=resolved_files,
            )

        if action.action_type == "write_file":
            if action.content is None:
                return self._error_observation("write_file requires content", action.path)
            try:
                path = self._resolve_path(action.path)
            except ValueError as exc:
                return self._error_observation(str(exc), action.path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(action.content, encoding="utf-8")
            return SafeCodeObservation(
                success=True,
                output=f"Wrote {len(action.content)} bytes to {action.path}",
                reward=0.05,
                feedback=f"Updated {action.path}.",
                current_path=action.path,
            )

        if action.action_type == "search":
            if not action.pattern:
                return self._error_observation("search requires pattern", action.path)
            return self._search_workspace(action.pattern, action.path)

        if action.action_type == "diff":
            return self._workspace_diff(action.path)

        if action.action_type == "run_command":
            if not action.command:
                return self._error_observation("run_command requires command", action.path)
            return self._run_command(action.command)

        if action.action_type == "submit":
            result = self._grader.evaluate_workspace(
                self._state.task_id,
                self._workspace_path,
                final=True,
            )
            return SafeCodeObservation(
                success=result.success,
                output=result.stdout[:4000],
                error=result.stderr[:2000],
                exit_code=result.exit_code,
                reward=result.reward,
                done=True,
                feedback=result.feedback,
                current_path=".",
            )

        return self._error_observation(f"Unsupported action_type: {action.action_type}", action.path)

    def _run_command(self, command: str) -> SafeCodeObservation:
        try:
            parts = shlex.split(command)
        except ValueError as exc:
            return self._error_observation(f"Invalid command syntax: {exc}", ".")

        if not parts:
            return self._error_observation("run_command requires a non-empty command", ".")
        task_commands = set(TASKS[self._state.task_id].allowed_commands)
        if parts[0] not in task_commands:
            return self._error_observation(
                f"Command '{parts[0]}' is not allowed. Allowed commands: {sorted(task_commands)}",
                ".",
            )

        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{self._workspace_path}{os.pathsep}{existing_pythonpath}"
            if existing_pythonpath
            else str(self._workspace_path)
        )

        result = subprocess.run(
            parts,
            cwd=str(self._workspace_path),
            capture_output=True,
            text=True,
            timeout=20,
            env=env,
        )
        self._state.last_command = command

        reward = 0.0
        feedback = f"Command exited with code {result.returncode}."
        if parts[0] == "pytest":
            grade = self._grader.evaluate_workspace(
                self._state.task_id,
                self._workspace_path,
                final=False,
            )
            reward = grade.reward
            feedback = grade.feedback

        return SafeCodeObservation(
            success=result.returncode == 0,
            output=result.stdout[:4000],
            error=result.stderr[:2000],
            exit_code=result.returncode,
            reward=reward,
            done=False,
            feedback=feedback,
            current_path=".",
            metadata={"command": command},
        )

    def _search_workspace(self, pattern: str, path: str) -> SafeCodeObservation:
        try:
            target = self._resolve_path(path)
        except ValueError as exc:
            return self._error_observation(str(exc), path)
        if not target.exists():
            return self._error_observation(f"Path not found: {path}", path)

        rg_binary = shutil.which("rg")
        if rg_binary:
            result = subprocess.run(
                [rg_binary, "-n", pattern, str(target)],
                cwd=str(self._workspace_path),
                capture_output=True,
                text=True,
            )
            output = result.stdout if result.returncode in (0, 1) else ""
            error = result.stderr if result.returncode not in (0, 1) else ""
            return SafeCodeObservation(
                success=result.returncode in (0, 1),
                output=output[:4000],
                error=error[:2000],
                exit_code=0 if result.returncode in (0, 1) else result.returncode,
                feedback="Search completed.",
                current_path=path,
                files=self._extract_search_files(output),
            )

        matches = []
        if target.is_file():
            files = [target]
        else:
            files = [file_path for file_path in target.rglob("*") if file_path.is_file()]
        for file_path in files:
            if "__pycache__" in file_path.parts or file_path.suffix == ".pyc":
                continue
            for line_no, line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), start=1):
                if pattern in line:
                    rel_path = file_path.resolve().relative_to(self._workspace_path.resolve()).as_posix()
                    matches.append(f"{rel_path}:{line_no}:{line}")
        return SafeCodeObservation(
            success=True,
            output="\n".join(matches[:200]),
            feedback="Search completed.",
            current_path=path,
            files=sorted({item.split(":", 1)[0] for item in matches}),
        )

    def _workspace_diff(self, path: str) -> SafeCodeObservation:
        target = path if path not in ("", ".") else ""
        changed_files = self._compute_changed_files()
        if target:
            normalized = target.rstrip("/")
            changed_files = [
                file_path
                for file_path in changed_files
                if file_path == normalized or file_path.startswith(normalized + "/")
            ]
        if not changed_files:
            return SafeCodeObservation(
                success=True,
                output="",
                feedback="No workspace changes yet.",
                current_path=path or ".",
                files=[],
            )

        diff_chunks = []
        for relative_path in changed_files[:20]:
            current_path = self._workspace_path / relative_path
            original = self._starter_snapshot.get(relative_path, "").splitlines(keepends=True)
            current = []
            if current_path.exists():
                try:
                    current = current_path.read_text(encoding="utf-8").splitlines(keepends=True)
                except UnicodeDecodeError:
                    continue
            diff = difflib.unified_diff(
                original,
                current,
                fromfile=f"a/{relative_path}",
                tofile=f"b/{relative_path}",
            )
            diff_chunks.append("".join(diff))

        return SafeCodeObservation(
            success=True,
            output="\n".join(chunk for chunk in diff_chunks if chunk)[:12000],
            feedback=f"Generated diff for {len(changed_files)} changed files.",
            current_path=path or ".",
            files=changed_files,
        )

    def _list_files(self, path: str) -> list[str]:
        target = self._resolve_path(path)
        workspace_root = self._workspace_path.resolve()
        if not target.exists():
            raise FileNotFoundError(path)

        if target.is_file():
            return [target.relative_to(workspace_root).as_posix()]

        entries = []
        for child in sorted(target.rglob("*")):
            if "__pycache__" in child.parts or child.suffix == ".pyc":
                continue
            rel_path = child.resolve().relative_to(workspace_root).as_posix()
            entries.append(rel_path + ("/" if child.is_dir() else ""))
        return entries

    def _snapshot_workspace(self) -> dict[str, str]:
        snapshot: dict[str, str] = {}
        if self._workspace_path is None:
            return snapshot
        workspace_root = self._workspace_path.resolve()
        for path in self._workspace_path.rglob("*"):
            if path.is_file():
                if "__pycache__" in path.parts or path.suffix == ".pyc":
                    continue
                try:
                    content = path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    continue
                snapshot[path.resolve().relative_to(workspace_root).as_posix()] = content
        return snapshot

    def _compute_changed_files(self) -> list[str]:
        if self._workspace_path is None:
            return []
        changed = []
        current = self._snapshot_workspace()
        for rel_path, content in current.items():
            if self._starter_snapshot.get(rel_path) != content:
                changed.append(rel_path)
        for rel_path in self._starter_snapshot:
            if rel_path not in current:
                changed.append(rel_path)
        return sorted(set(changed))

    def _resolve_path(self, relative_path: str) -> Path:
        self._ensure_workspace()
        normalized = Path(relative_path or ".")
        candidate = (self._workspace_path / normalized).resolve()
        workspace_root = self._workspace_path.resolve()
        if workspace_root not in (candidate, *candidate.parents):
            raise ValueError(f"Path escapes workspace: {relative_path}")
        return candidate

    def _extract_search_files(self, output: str) -> list[str]:
        files = []
        for line in output.splitlines():
            if ":" in line:
                files.append(line.split(":", 1)[0])
        return sorted(set(files))

    def _error_observation(self, message: str, path: str) -> SafeCodeObservation:
        return SafeCodeObservation(
            success=False,
            output="",
            error=message,
            exit_code=1,
            reward=0.0,
            done=False,
            feedback=message,
            current_path=path,
        )

    def _ensure_workspace(self) -> None:
        if self._workspace_path is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

    def _cleanup_workspace(self) -> None:
        if self._workspace_path and self._workspace_path.exists():
            shutil.rmtree(self._workspace_path, ignore_errors=True)
        self._workspace_path = None

    @property
    def state(self) -> SafeCodeState:
        return self._state
