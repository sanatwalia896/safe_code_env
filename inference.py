"""
LLM-driven tool-use inference loop for the workspace-based Safe Code env.

The agent interacts with the environment through explicit tool actions:
list_files, read_file, write_file, search, run_command, and submit.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import Any

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

from client import SafeCodeAction, SafeCodeEnv

load_dotenv(find_dotenv())


MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Meta-Llama-3.1-8B-Instruct")
API_KEY = os.environ.get("HF_TOKEN", "")
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
ENV_URL = os.environ.get("ENV_URL", "http://localhost:8000")
MAX_AGENT_STEPS = int(os.environ.get("MAX_AGENT_STEPS", "14"))
NUM_EPISODES = int(os.environ.get("NUM_EPISODES", "7"))

SYSTEM_PROMPT = """You are a careful coding agent operating inside a workspace environment.

You do not output code directly unless you are using the `write_file` or `edit_file` actions.
You must choose exactly one environment action at a time.

Available action schema:
{
  "action_type": "list_files" | "read_file" | "read_files" | "write_file" | "edit_file" | "search" | "diff" | "run_command" | "submit",
  "path": "workspace-relative path",
  "paths": ["workspace-relative path", "..."],
  "content": "full replacement file contents for write_file",
  "old_text": "exact text to be replaced for edit_file",
  "new_text": "replacement text for edit_file",
  "pattern": "substring or regex-lite text for search",
  "command": "command string for run_command"
}

Rules:
- Always return exactly one JSON object, with no markdown fences and no explanation.
- Prefer inspecting files and tests before editing.
- Use `read_files` when you need to inspect a source file and its test together.
- Use `edit_file` for precise modifications; it requires that `old_text` be unique in the file.
- Use `write_file` only when a complete rewrite is necessary.
- Mandatory: Run `pytest -q` using `run_command` after every modification to verify progress.
- Mandatory: Use `diff` to check changes before using `submit`.
- Use `submit` only when you believe the workspace is fixed or when you are out of ideas.
- Never reference files outside the workspace.
"""


def build_client() -> OpenAI:
    if not API_KEY:
        raise RuntimeError("HF_TOKEN environment variable not set")
    return OpenAI(api_key=API_KEY, base_url=API_BASE_URL)


def call_llm(llm: OpenAI, messages: list[dict[str, str]]) -> str:
    response = llm.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=2000,
        temperature=0.1,
    )
    return (response.choices[0].message.content or "").strip()


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_]*\n?", "", stripped)
        stripped = re.sub(r"\n?```$", "", stripped).strip()

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def observation_to_user_message(obs) -> str:
    payload = {
        "task_id": obs.task_id,
        "task_description": obs.task_description,
        "success": obs.success,
        "feedback": obs.feedback,
        "reward": obs.reward,
        "done": obs.done,
        "current_path": obs.current_path,
        "files": obs.files[:100],
        "changed_files": obs.changed_files[:100],
        "output": obs.output[:4000],
        "error": obs.error[:2000],
        "exit_code": obs.exit_code,
        "available_tools": obs.available_tools,
    }
    return json.dumps(payload, indent=2)


def extract_candidate_paths(text: str) -> list[str]:
    if not text:
        return []
    matches = re.findall(r"([A-Za-z0-9_./-]+\.(?:py|json|md))", text)
    cleaned = []
    for match in matches:
        if match.startswith("a/") or match.startswith("b/"):
            match = match[2:]
        cleaned.append(match)
    seen = []
    for item in cleaned:
        if item not in seen:
            seen.append(item)
    return seen


def normalize_action(raw_action: dict[str, Any]) -> SafeCodeAction:
    if "action" in raw_action and isinstance(raw_action["action"], dict):
        raw_action = raw_action["action"]
    action_type = raw_action["action_type"]
    normalized: dict[str, Any] = {"action_type": action_type, "path": raw_action.get("path", ".")}

    if action_type == "read_files":
        normalized["paths"] = raw_action.get("paths") or []
        normalized["path"] = raw_action.get("path", ".")
    elif action_type == "write_file":
        normalized["content"] = raw_action.get("content")
    elif action_type == "edit_file":
        normalized["old_text"] = raw_action.get("old_text")
        normalized["new_text"] = raw_action.get("new_text")
    elif action_type == "search":
        normalized["pattern"] = raw_action.get("pattern")
    elif action_type == "run_command":
        normalized["command"] = raw_action.get("command")
    elif action_type in {"submit", "diff", "list_files", "read_file"}:
        pass

    return SafeCodeAction(**normalized)


def choose_action(llm: OpenAI, messages: list[dict[str, str]], step: int) -> SafeCodeAction:
    raw_response = call_llm(llm, messages)
    try:
        raw_action = extract_json_object(raw_response)
        return normalize_action(raw_action)
    except Exception as exc:
        messages.append(
            {
                "role": "assistant",
                "content": raw_response[:4000],
            }
        )
        messages.append(
            {
                "role": "user",
                "content": (
                    "Your previous response was invalid JSON for the action schema.\n"
                    f"Parser error: {exc}\n"
                    "Reply again with exactly one valid JSON action object."
                ),
            }
        )
        retry_response = call_llm(llm, messages)
        try:
            raw_action = extract_json_object(retry_response)
            return normalize_action(raw_action)
        except Exception:
            fallback = fallback_action(messages, step)
            messages.append(
                {
                    "role": "assistant",
                    "content": json.dumps(fallback.model_dump(exclude_none=True)),
                }
            )
            return fallback


def fallback_action(messages: list[dict[str, str]], step: int) -> SafeCodeAction:
    transcript = "\n".join(message.get("content", "") for message in messages[-6:])
    candidate_paths = extract_candidate_paths(transcript)

    if "failed" in transcript or "error" in transcript:
        focused = [
            path
            for path in candidate_paths
            if path.startswith("src/") or path.startswith("tests/") or path.endswith(".py")
        ]
        if focused:
            return SafeCodeAction(action_type="read_files", paths=focused[:2], path=".")
        if "pytest" not in transcript:
            return SafeCodeAction(action_type="run_command", command="pytest -q")

    if step >= MAX_AGENT_STEPS - 1:
        if "pytest" not in transcript:
            return SafeCodeAction(action_type="run_command", command="pytest -q")
        return SafeCodeAction(action_type="submit")

    if "run_command" not in transcript and "pytest" not in transcript:
        return SafeCodeAction(action_type="run_command", command="pytest -q")

    if candidate_paths:
        return SafeCodeAction(action_type="read_files", paths=candidate_paths[:2], path=".")

    return SafeCodeAction(action_type="list_files", path=".")


def short_action_log(action: SafeCodeAction) -> str:
    fields = {
        "action_type": action.action_type,
        "path": action.path,
        "paths": action.paths,
        "command": action.command,
        "pattern": action.pattern,
    }
    if action.action_type == "write_file" and action.content is not None:
        fields["content_preview"] = action.content[:80].replace("\n", "\\n")
    return json.dumps({k: v for k, v in fields.items() if v not in (None, "", ".")})


def run_episode(llm: OpenAI, env: SafeCodeEnv) -> float:
    reset = env.reset()
    obs = reset.observation
    print(f"[START] task={obs.task_id}")
    sys.stdout.flush()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "You are starting a new task in the Safe Code workspace environment.\n"
                "Current observation:\n"
                f"{observation_to_user_message(obs)}"
            ),
        },
    ]

    final_reward = 0.0

    for step in range(1, MAX_AGENT_STEPS + 1):
        action = choose_action(llm, messages, step)
        result = env.step(action)
        obs = result.observation
        final_reward = result.reward or obs.reward

        print(
            f"[STEP] step={step} action={short_action_log(action)} "
            f"reward={final_reward:.2f} done={obs.done}"
        )
        sys.stdout.flush()

        messages.append(
            {
                "role": "assistant",
                "content": json.dumps(action.model_dump(exclude_none=True)),
            }
        )
        messages.append(
            {
                "role": "user",
                "content": (
                    "Environment response:\n"
                    f"{observation_to_user_message(obs)}"
                ),
            }
        )

        if not obs.done and step >= MAX_AGENT_STEPS - 2 and action.action_type != "diff":
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "You are near the step limit. Prefer validating with pytest, inspecting a diff, "
                        "and then submitting if the workspace looks correct."
                    ),
                }
            )

        if obs.done:
            break

    print(f"[END] success={str(obs.success).lower()} score={final_reward:.2f}")
    sys.stdout.flush()
    return final_reward


def main() -> None:
    llm = build_client()
    rewards = []

    with SafeCodeEnv(base_url=ENV_URL).sync() as env:
        for _ in range(NUM_EPISODES):
            reward = run_episode(llm, env)
            rewards.append(reward)
            time.sleep(1)


if __name__ == "__main__":
    main()
