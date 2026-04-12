# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Safe Code Env Environment.

Tool-based actions operate against a persistent workspace.
"""

from typing import List, Literal, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import ConfigDict, Field


class SafeCodeAction(Action):
    """A single tool invocation against the current workspace."""

    model_config = ConfigDict(extra="forbid")

    action_type: Literal[
        "list_files",
        "read_file",
        "read_files",
        "write_file",
        "edit_file",
        "search",
        "diff",
        "run_command",
        "submit",
    ] = Field(..., description="Tool/action to execute")
    action_intent: str = Field(
        default="",
        description="Natural-language intent for semantic safety scoring.",
    )
    path: str = Field(default=".", description="Workspace-relative path used by file-based tools")
    paths: Optional[List[str]] = Field(default=None, description="Workspace-relative paths for read_files")
    content: Optional[str] = Field(default=None, description="Replacement file contents for write_file")
    old_text: Optional[str] = Field(default=None, description="Text to replace for edit_file")
    new_text: Optional[str] = Field(default=None, description="Replacement text for edit_file")
    pattern: Optional[str] = Field(default=None, description="Search pattern for search action")
    command: Optional[str] = Field(default=None, description="Command for run_command action")

    # Legacy fields retained for compatibility with older clients.
    code: Optional[str] = Field(default=None, description="Legacy code-submission payload")
    action_description: Optional[str] = Field(default=None, description="Legacy intent payload")
    task_id: Optional[str] = Field(default=None, description="Legacy task identifier")


class SafeCodeObservation(Observation):
    """Result of a single tool invocation."""

    model_config = ConfigDict(extra="allow")

    success: bool = Field(default=True, description="Whether the tool succeeded")
    output: str = Field(default="", description="Primary tool output")
    error: str = Field(default="", description="Error details if the tool failed")
    error_code: str = Field(default="", description="Structured error category for failures")
    exit_code: int = Field(default=0, description="Exit code for command execution")
    reward: float = Field(default=0.0, description="Reward for this step")
    done: bool = Field(default=False, description="Whether the episode is complete")
    passed_tests: int = Field(default=0, description="Tests passed in current workspace")
    failed_tests: int = Field(default=0, description="Tests failed/errors in current workspace")
    feedback: str = Field(default="", description="Environment/grader feedback")
    safety_score: float = Field(default=1.0, description="Safety score from rules + BGE")
    completion_score: float = Field(default=0.0, description="Completion score from tests")
    task_id: str = Field(default="", description="Current task identifier")
    task_description: str = Field(default="", description="Task instructions")
    workspace_path: str = Field(default="", description="Absolute seeded workspace path")
    current_path: str = Field(default=".", description="Workspace-relative path for this action")
    files: List[str] = Field(default_factory=list, description="Relevant file list")
    changed_files: List[str] = Field(default_factory=list, description="Files changed in this episode")
    available_tools: List[str] = Field(default_factory=list, description="Available tool names")


class SafeCodeState(State):
    """Persistent environment state across tool steps."""

    episode_id: str = Field(default="", description="Unique episode identifier")
    step_count: int = Field(default=0, description="Number of steps taken in current episode")
    task_id: str = Field(default="", description="Current task identifier")
    workspace_path: str = Field(default="", description="Absolute workspace path")
    changed_files: List[str] = Field(default_factory=list, description="Files modified in workspace")
    available_tools: List[str] = Field(default_factory=list, description="Exposed tools")
    last_command: str = Field(default="", description="Last executed command")
    last_exit_code: int = Field(default=0, description="Last command exit code")
    last_safety_score: float = Field(default=1.0, description="Last safety score")
    last_completion_score: float = Field(default=0.0, description="Last completion score")
