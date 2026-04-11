---
title: Safe Code Env Environment Server
emoji: 🛠️
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Safe Code Env Environment

Safe Code Env now behaves like a real coding workspace instead of a single-shot code grader. Each episode seeds a broken mini-repository into a temp directory, exposes explicit tools over that workspace, and grades the final filesystem state by running the real test suite.

## How It Works

- `reset()` creates a fresh workspace from `server/tasks/<task_id>/starter`
- the agent uses one tool per `step()`
- the workspace persists across steps
- `submit` runs the final grader against the actual files and tests
- rewards come from real progress, especially pytest results

## Tool Actions

`SafeCodeAction` supports:

- `list_files`
- `read_file`
- `read_files`
- `write_file`
- `search`
- `diff`
- `run_command`
- `submit`

Example:

```python
from safe_code_env import SafeCodeAction, SafeCodeEnv

with SafeCodeEnv(base_url="http://localhost:8000").sync() as env:
    reset = env.reset()
    print(reset.observation.task_description)

    result = env.step(SafeCodeAction(action_type="list_files", path="."))
    print(result.observation.output)

    result = env.step(SafeCodeAction(action_type="read_file", path="app.py"))
    print(result.observation.output)
```

## Observation Shape

Each observation includes:

- `success`
- `output`
- `error`
- `exit_code`
- `reward`
- `done`
- `feedback`
- `task_id`
- `task_description`
- `workspace_path`
- `current_path`
- `files`
- `changed_files`
- `available_tools`

## Task Layout

The environment currently seeds seven realistic starter repos:

- `task_1`: Flask `/health` endpoint repair
- `task_2`: SQL injection fix in `db_utils.py`
- `task_3`: broken validator module under `src/`
- `task_4`: broken rolling-window rate limiter
- `task_5`: broken JSON config loader/default merge
- `task_6`: broken recursive audit secret redaction
- `task_7`: larger multi-module orders service with a pricing bug under `src/orders/`

The source for those tasks lives under [server/tasks](/Users/sanat/Documents/PROJECTS/scaler_hackathon/safe_code_env/server/tasks).

## Grading

The grader reads the final workspace state and runs each task’s real pytest suite. `run_command` with `pytest -q` gives partial progress, while `submit` performs final evaluation and ends the episode. `diff` shows the current workspace delta relative to the seeded starter, and `read_files` lets the agent inspect multiple files in one step. The largest task (`task_7`) forces the agent to navigate a broader service-style codebase rather than a single file.

## Local Run

```bash
uv run uvicorn server.app:app --reload
```

In another shell:

```bash
python inference.py
```
