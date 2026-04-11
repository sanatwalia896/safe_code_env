# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Safe Code Env Environment.

The realistic version of Safe Code Env exposes explicit filesystem and command
tools over a seeded workspace instead of accepting a single code blob.
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
    path: str = Field(
        default=".",
        description="Workspace-relative path used by file-based tools",
    )
    paths: Optional[List[str]] = Field(
        default=None,
        description="Workspace-relative paths used by read_files",
    )
    content: Optional[str] = Field(
        default=None,
        description="Replacement file contents for write_file",
    )
    old_text: Optional[str] = Field(
        default=None,
        description="Text to be replaced for edit_file",
    )
    new_text: Optional[str] = Field(
        default=None,
        description="Replacement text for edit_file",
    )
    pattern: Optional[str] = Field(
        default=None,
        description="Search pattern for the search action",
    )
    command: Optional[str] = Field(
        default=None,
        description="Command to run inside the workspace for run_command",
    )


class SafeCodeObservation(Observation):
    """Result of a single tool invocation."""

    model_config = ConfigDict(extra="allow")

    success: bool = Field(default=True, description="Whether the tool succeeded")
    output: str = Field(default="", description="Primary tool output")
    error: str = Field(default="", description="Error details if the tool failed")
    exit_code: int = Field(default=0, description="Exit code for command execution")
    reward: float = Field(default=0.0, description="Reward for this step")
    done: bool = Field(default=False, description="Whether the episode is complete")
    passed_tests: int = Field(default=0, description="Number of tests passed in the current state")
    failed_tests: int = Field(default=0, description="Number of tests failed in the current state")
    feedback: str = Field(
        default="",
        description="Environment or grader feedback for the agent",
    )
    task_id: str = Field(default="", description="Current task identifier")
    task_description: str = Field(
        default="",
        description="Task instructions shown to the agent",
    )
    workspace_path: str = Field(
        default="",
        description="Absolute path to the seeded workspace for this episode",
    )
    current_path: str = Field(
        default=".",
        description="Workspace-relative path the action operated on",
    )
    files: List[str] = Field(
        default_factory=list,
        description="Relevant file listing for list/search-style actions",
    )
    changed_files: List[str] = Field(
        default_factory=list,
        description="Workspace-relative files changed so far in the episode",
    )
    available_tools: List[str] = Field(
        default_factory=list,
        description="Available tool names for the current environment",
    )


class SafeCodeState(State):
    """Persistent environment state across tool steps."""

    task_id: str = Field(default="", description="Current task identifier")
    workspace_path: str = Field(default="", description="Absolute workspace path")
    changed_files: List[str] = Field(
        default_factory=list,
        description="Files modified in the current workspace",
    )
    available_tools: List[str] = Field(
        default_factory=list,
        description="Tool names exposed by the environment",
    )
    last_command: str = Field(default="", description="Last command executed")
    last_exit_code: int = Field(default=0, description="Last command exit code")
