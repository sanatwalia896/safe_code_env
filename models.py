# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Safe Code Env Environment.

The safe_code_env environment grades agent-submitted Python code across real-world engineering tasks.
"""
from openenv.core.env_server.types import Action, Observation
from pydantic import Field, ConfigDict

class SafeCodeAction(Action):
    model_config = ConfigDict(extra="allow")
    
    code: str = Field(
        ...,
        description="Python code written by the agent to solve the task"
    )
    task_id: str = Field(
        default="task_1",
        description="Which task: task_1, task_2, or task_3"
    )

class SafeCodeObservation(Observation):
    model_config = ConfigDict(extra="allow")
    
    stdout:           str   = Field(default="",    description="stdout from execution")
    stderr:           str   = Field(default="",    description="stderr from execution")
    exit_code:        int   = Field(default=0,     description="0=success 1=error")
    reward:           float = Field(default=0.0,   description="reward 0.0-1.0")
    done:             bool  = Field(default=False, description="episode complete")
    feedback:         str   = Field(default="",    description="grader feedback")
    task_description: str   = Field(default="",    description="task the agent must solve")
    safety_score:     float = Field(default=1.0,   description="safety score 0.0-1.0")
    completion_score: float = Field(default=0.0,   description="task completion 0.0-1.0")
