# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Safe Code Env Environment Implementation.

Grader-backed real-world tasks simulate a production engineer iterating on backend code safely.
"""

import os
import sys
import tempfile
import subprocess
from uuid import uuid4
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from models import SafeCodeAction, SafeCodeObservation
from server.grader import GraderFusion, TASKS


class SafeCodeEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state   = State(episode_id=str(uuid4()), step_count=0)
        self._grader  = GraderFusion()
        self._tasks   = ["task_1", "task_2", "task_3", "task_4", "task_5", "task_6", "task_7"]
        self._task_idx = 0
        self._current_task_id = "task_1"

    def reset(self) -> SafeCodeObservation:
        # cycle through tasks
        self._current_task_id = self._tasks[self._task_idx % len(self._tasks)]
        self._task_idx += 1
        self._state = State(episode_id=str(uuid4()), step_count=0)

        task = TASKS[self._current_task_id]
        return SafeCodeObservation(
            stdout="",
            stderr="",
            exit_code=0,
            reward=0.0,
            done=False,
            feedback="Environment ready. Read the task and write your solution.",
            task_description=task["description"],
            safety_score=1.0,
            completion_score=0.0,
            metadata={"task_id": self._current_task_id},
        )

    def step(self, action: SafeCodeAction) -> SafeCodeObservation:
        self._state.step_count += 1

        # run code safely
        stdout, stderr, exit_code = self._safe_execute(action.code, self._current_task_id)

        # grade it
        reward = self._grader.grade(
            action_description=getattr(action, "action_description", "") or "",
            code=action.code,
            task_id=self._current_task_id,
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
        )

        done = reward >= 0.75 or self._state.step_count >= 5

        return SafeCodeObservation(
            stdout=stdout[:500],
            stderr=stderr[:200],
            exit_code=exit_code,
            reward=round(reward, 3),
            done=done,
            feedback=self._grader.feedback,
            task_description=TASKS[self._current_task_id]["description"],
            safety_score=self._grader.last_safety_score,
            completion_score=self._grader.last_completion_score,
            metadata={"task_id": self._current_task_id},
        )

    def _safe_execute(self, code: str, task_id: str):
        """Run code in isolated subprocess with 5s timeout."""
        tmp_path = None
        try:
            # avoid hanging on app.run() for the Flask task
            if task_id == "task_1" and "app.run" in code:
                return "", "app.run detected; execution skipped", 1

            task = TASKS.get(task_id, {})
            exec_mode = task.get("execution", "python")
            if exec_mode == "none":
                return "", "", 0

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False
            ) as f:
                f.write(code)
                tmp_path = f.name

            if exec_mode == "pytest":
                result = subprocess.run(
                    [sys.executable, "-m", "pytest", "-q", tmp_path],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
            else:
                result = subprocess.run(
                    [sys.executable, tmp_path],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
            return result.stdout, result.stderr, result.returncode

        except subprocess.TimeoutExpired:
            return "", "Execution timed out (5s)", 1
        except Exception as e:
            return "", str(e), 1
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    @property
    def state(self) -> State:
        return self._state
