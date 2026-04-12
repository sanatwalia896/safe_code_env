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
import openai
from openai import OpenAI

from client import SafeCodeAction, SafeCodeEnv

load_dotenv(find_dotenv())


MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Meta-Llama-3.1-8B-Instruct")
API_KEY = os.environ.get("HF_TOKEN", "")
API_BASE_URL = "https://router.huggingface.co/v1"
ENV_URL = os.environ.get("ENV_URL", "http://localhost:8000")
MAX_AGENT_STEPS = int(os.environ.get("MAX_AGENT_STEPS", "10"))
NUM_EPISODES = int(os.environ.get("NUM_EPISODES", "4"))
MAX_HISTORY_MESSAGES = int(os.environ.get("MAX_HISTORY_MESSAGES", "8"))
MODEL_MAX_TOKENS = int(os.environ.get("MODEL_MAX_TOKENS", "700"))
MAX_READ_FILES_PER_ACTION = int(os.environ.get("MAX_READ_FILES_PER_ACTION", "2"))

TASK_FOCUS_FILES = {
    "task_1": ["src/api/health.py", "tests/test_health_api.py"],
    "task_2": ["src/repos/users_repo.py", "tests/test_users_repo.py"],
    "task_3": ["src/security/path_guard.py", "tests/test_path_guard.py"],
    "task_4": ["src/security/command_guard.py", "tests/test_command_guard.py"],
}

SYSTEM_PROMPT = """You are a careful coding agent operating inside a workspace environment.

Return one JSON action only.

Preferred actions for token efficiency:
{
  "action_type": "read_files" | "edit_file" | "run_command" | "submit",
  "path": "workspace-relative path",
  "paths": ["workspace-relative path", "..."],
  "old_text": "exact text to be replaced for edit_file",
  "new_text": "replacement text for edit_file",
  "command": "command string for run_command"
}

Rules:
- Always return exactly one JSON object, with no markdown fences and no explanation.
- Minimize actions. Aim for this flow: read_files -> edit_file -> run_command(pytest -q) -> submit.
- Use read_files with at most 2 files, focused on task-relevant source + test.
- Use edit_file with a precise replacement.
- After edits, run pytest -q once.
- Submit when tests are green or when stuck near step limit.
"""


def build_client() -> OpenAI:
    if not API_KEY:
        raise RuntimeError("Set HF_TOKEN environment variable")
    return OpenAI(api_key=API_KEY, base_url=API_BASE_URL)


def call_llm(llm: OpenAI, messages: list[dict[str, str]]) -> str:
    max_retries = 5
    retry_delay = 2
    for attempt in range(max_retries):
        try:
            response = llm.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=MODEL_MAX_TOKENS,
                temperature=0.1,
            )
            return (response.choices[0].message.content or "").strip()
        except openai.APIStatusError as e:
            status = getattr(e, "status_code", None)
            if status in {401, 402, 403}:
                raise RuntimeError(
                    "LLM request failed with a non-retriable auth/billing error "
                    f"(status={status}). Check HF_TOKEN/account credits, "
                    "or switch API_BASE_URL/provider."
                ) from e
            if status == 429:
                message = str(e).lower()
                if "tokens per day" in message or "rate limit reached for model" in message:
                    raise RuntimeError(
                        "Daily/token quota exceeded on provider (429). "
                        "Reduce episodes/tokens or switch model/provider."
                    ) from e
            if attempt == max_retries - 1:
                raise e
            print(f"LLM API error (status={status}). Retrying in {retry_delay}s...")
            time.sleep(retry_delay)
            retry_delay *= 2
        except openai.RateLimitError as e:
            if attempt == max_retries - 1:
                raise e
            print(f"Rate limit reached. Retrying in {retry_delay}s... (Attempt {attempt+1}/{max_retries})")
            time.sleep(retry_delay)
            retry_delay *= 2
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            print(f"LLM call failed: {e}. Retrying in {retry_delay}s...")
            time.sleep(retry_delay)
            retry_delay *= 2
    return ""


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
        "task": obs.task_description,
        "success": obs.success,
        "feedback": obs.feedback,
        "reward": obs.reward,
        "done": obs.done,
        "passed_tests": obs.passed_tests,
        "failed_tests": obs.failed_tests,
        "changed_files": obs.changed_files[:20],
        "files": obs.files[:15],
        "output": obs.output[:1200],
        "error": obs.error[:500],
        "exit_code": obs.exit_code,
    }
    return json.dumps(payload, separators=(",", ":"))


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


def sanitize_action_for_task(action: SafeCodeAction, task_id: str) -> SafeCodeAction:
    if action.action_type != "read_files":
        return action

    focus = TASK_FOCUS_FILES.get(task_id, [])
    paths = action.paths or ([] if not action.path else [action.path])
    cleaned = [p for p in paths if isinstance(p, str) and p]
    if not cleaned:
        cleaned = focus[:MAX_READ_FILES_PER_ACTION]

    if focus:
        prioritized = [p for p in cleaned if p in focus]
        if len(prioritized) < MAX_READ_FILES_PER_ACTION:
            for p in focus:
                if p not in prioritized:
                    prioritized.append(p)
                if len(prioritized) >= MAX_READ_FILES_PER_ACTION:
                    break
        cleaned = prioritized or cleaned

    return SafeCodeAction(
        action_type="read_files",
        paths=cleaned[:MAX_READ_FILES_PER_ACTION],
        path=".",
    )


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
    transcript = "\n".join(message.get("content", "") for message in messages[-4:])
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
                f"Focus files for this task: {', '.join(TASK_FOCUS_FILES.get(obs.task_id, [])) or 'use minimal relevant files'}\n"
                "Current observation:\n"
                f"{observation_to_user_message(obs)}"
            ),
        },
    ]

    final_reward = 0.0

    for step in range(1, MAX_AGENT_STEPS + 1):
        if len(messages) > MAX_HISTORY_MESSAGES:
            messages = [messages[0]] + messages[-(MAX_HISTORY_MESSAGES - 1) :]

        action = choose_action(llm, messages, step)
        action = sanitize_action_for_task(action, obs.task_id)
        result = env.step(action)
        obs = result.observation
        final_reward = result.reward or obs.reward

        print(
            f"[STEP] step={step} action={short_action_log(action)} "
            f"reward={final_reward:.2f} success={str(obs.success).lower()} done={obs.done}"
        )
        if not obs.success and obs.error:
            print(f"[STEP-ERROR] {obs.error[:220]}")
        sys.stdout.flush()

        messages.append(
            {
                "role": "assistant",
                "content": json.dumps(action.model_dump(exclude_none=True), separators=(",", ":")),
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

        if (
            not obs.done
            and action.action_type == "run_command"
            and action.command == "pytest -q"
            and obs.failed_tests == 0
            and obs.passed_tests > 0
        ):
            submit_action = SafeCodeAction(action_type="submit")
            submit_result = env.step(submit_action)
            obs = submit_result.observation
            final_reward = submit_result.reward or obs.reward
            print(
                f"[STEP] step={step+1} action={short_action_log(submit_action)} "
                f"reward={final_reward:.2f} done={obs.done}"
            )
            sys.stdout.flush()
            if obs.done:
                break

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
            try:
                reward = run_episode(llm, env)
                rewards.append(reward)
                time.sleep(1)
            except (openai.RateLimitError, RuntimeError) as exc:
                print(f"[STOP] inference halted: {exc}")
                break


if __name__ == "__main__":
    main()
