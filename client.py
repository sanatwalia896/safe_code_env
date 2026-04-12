# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Safe Code Env Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

try:
    from .models import SafeCodeAction, SafeCodeObservation, SafeCodeState
except ImportError:
    from models import SafeCodeAction, SafeCodeObservation, SafeCodeState


class SafeCodeEnv(
    EnvClient[SafeCodeAction, SafeCodeObservation, SafeCodeState]
):
    """Client for the workspace-based Safe Code Env environment."""

    def _step_payload(self, action: SafeCodeAction) -> Dict:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict) -> StepResult[SafeCodeObservation]:
        obs_data = payload.get("observation") or payload
        observation = SafeCodeObservation(
            success=obs_data.get("success", True),
            output=obs_data.get("output", ""),
            error=obs_data.get("error", ""),
            exit_code=obs_data.get("exit_code", 0),
            reward=obs_data.get("reward", payload.get("reward", 0.0)),
            done=payload.get("done", obs_data.get("done", False)),
            passed_tests=obs_data.get("passed_tests", 0),
            failed_tests=obs_data.get("failed_tests", 0),
            feedback=obs_data.get("feedback", ""),
            task_id=obs_data.get("task_id", ""),
            task_description=obs_data.get("task_description", ""),
            workspace_path=obs_data.get("workspace_path", ""),
            current_path=obs_data.get("current_path", "."),
            files=obs_data.get("files", []),
            changed_files=obs_data.get("changed_files", []),
            available_tools=obs_data.get("available_tools", []),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", observation.reward),
            done=payload.get("done", observation.done),
        )

    def _parse_state(self, payload: Dict) -> SafeCodeState:
        return SafeCodeState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", ""),
            workspace_path=payload.get("workspace_path", ""),
            changed_files=payload.get("changed_files", []),
            available_tools=payload.get("available_tools", []),
            last_command=payload.get("last_command", ""),
            last_exit_code=payload.get("last_exit_code", 0),
        )
