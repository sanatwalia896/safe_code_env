# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Safe Code Env Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import SafeCodeAction, SafeCodeObservation
except ImportError:
    # Support direct script execution from repo root where package context is absent.
    from models import SafeCodeAction, SafeCodeObservation


class SafeCodeEnv(
    EnvClient[SafeCodeAction, SafeCodeObservation, State]
):
    """
    Client for the Safe Code Env Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> with SafeCodeEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.task_description.splitlines()[0])
        ...     action = SafeCodeAction(code="def health():\\n    return {'status': 'ok'}", task_id="task_1")
        ...     step_result = client.step(action)
        ...     print(step_result.observation.feedback)

    Example with Docker:
        >>> client = SafeCodeEnv.from_docker_image("safe_code_env-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     action = SafeCodeAction(code="def multiply(a,b):\\n    return a*b", task_id="task_3")
        ...     client.step(action)
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: SafeCodeAction) -> Dict:
        """Serialize the coding action into the HTTP/WebSocket payload."""

        return {
            "code": action.code,
            "task_id": action.task_id,
        }

    def _parse_result(self, payload: Dict) -> StepResult[SafeCodeObservation]:
        """Rebuild the observation from the server payload."""

        obs_data = payload.get("observation")
        if not obs_data:
            obs_data = payload
        observation = SafeCodeObservation(
            stdout=obs_data.get("stdout", ""),
            stderr=obs_data.get("stderr", ""),
            exit_code=obs_data.get("exit_code", 0),
            reward=obs_data.get("reward", payload.get("reward", 0.0)),
            done=payload.get("done", False),
            feedback=obs_data.get("feedback", ""),
            task_description=obs_data.get("task_description", ""),
            safety_score=obs_data.get("safety_score", 1.0),
            completion_score=obs_data.get("completion_score", 0.0),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", observation.reward),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
