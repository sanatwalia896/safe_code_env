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

Safe Code Env uses a single shared FastAPI + SQLite codebase for all tasks. Each episode seeds a workspace from `server/base_codebase`, applies a task-specific overlay, and then grades the final filesystem state by running real tests.

## How It Works

- `reset()` creates a fresh workspace from `server/base_codebase` plus `server/tasks/<task_id>/overlay`
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

The environment currently seeds five task overlays on the same shared codebase:

- `task_1`: FastAPI `/health` response contract fix
- `task_2`: SQLite query parameterization / SQL injection fix
- `task_3`: protected file path policy (`.env`, production db files)
- `task_4`: local git safety policy (`git reset` / `git restore` forbidden)
- `task_5`: users endpoint + service integration fix over SQLite

The shared base repo lives in [server/base_codebase](/Users/sanat/Documents/PROJECTS/scaler_hackathon/safe_code_env/server/base_codebase), and task overlays live in [server/tasks](/Users/sanat/Documents/PROJECTS/scaler_hackathon/safe_code_env/server/tasks).

## Grading

The grader reads the final workspace state and runs each task’s real pytest suite. `run_command` with `pytest -q` gives partial progress, while `submit` performs final evaluation and ends the episode. `diff` shows the workspace delta relative to seeded baseline.

Safety constraints include:

- blocked protected paths (`.env`, production SQLite files)
- blocked dangerous git commands (`reset`, `restore`, and hard resets)
- blocked arbitrary python payload execution via `run_command`

## Local Run

```bash
uv run uvicorn server.app:app --reload
```

In another shell:

```bash
python inference.py
```
