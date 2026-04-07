---
title: CodeNav Environment Server
emoji: 🧑‍💻
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - coding
  - developer-workflow
---

# CodeNav — Developer Workflow RL Environment

CodeNav is an RL environment where an agent acts as a software developer dropped into an unfamiliar Python codebase. The agent must read, diagnose, fix, verify, and submit — rewarded for the quality of the **entire reasoning process**, not just whether the final code is correct.

Most coding benchmarks ask one question: did the tests pass? CodeNav asks a harder one: did you reason well to get there?

---

## Quick Start

```python
from codenav import CodeNavAction, CodeNavEnv

env = CodeNavEnv.from_docker_image("codenav-env:latest")

obs = env.reset()
print(obs.message)  # Task description + available files

# Read a file
obs = env.step(CodeNavAction(action_type="read_file", filename="processor.py"))

# Submit a diagnosis
obs = env.step(CodeNavAction(
    action_type="submit_diagnosis",
    diagnosis="The bug is an off-by-one error — + 1 is incorrectly added to every average"
))

# Fix it
obs = env.step(CodeNavAction(
    action_type="edit_code",
    filename="processor.py",
    old_code="        averages[cid] = totals[cid] / counts[cid] + 1",
    new_code="        averages[cid] = totals[cid] / counts[cid]"
))

# Verify
obs = env.step(CodeNavAction(action_type="run_tests"))

# Submit
obs = env.step(CodeNavAction(action_type="submit"))
print(obs.final_score)       # e.g. 0.800
print(obs.score_breakdown)   # per-dimension breakdown
```

---

## Three Tasks

The `CODENAV_TASK` environment variable controls difficulty. Each episode picks randomly from a pool of scenarios — preventing memorization.

### Easy
Single-file bugs, 2 distractor files, no cross-file dependencies. 25 steps, pool of 5 scenarios.

Bug types: off-by-one errors, integer division, wrong variable, missing return statement, wrong comparison operator.

### Medium
Cross-file bugs — cause is in one file, symptom in another. Two-bug episodes: bug 2 surfaces automatically after bug 1 is fixed. 35 steps, pool of 6 scenarios.

Bug types: None check missing after sanitization, empty string validation, case sensitivity, wrong dictionary key in API response, missing password strip, sessions not invalidated on lockout.

### Hard
Silent bugs — no error raised, wrong output produced under specific conditions. Requires understanding design intent across 3+ files. Two-bug episodes. 45 steps, pool of 8 scenarios.

Bug types: shallow merge corrupting nested config, mutable default argument, cache invalidation order, wrong merge direction, default config mutation, double timezone conversion, key collision between modules, validator silently dropping nested keys.

---

## Reward Function

Total episode score ranges from **-1.0 to +1.0** across five independent dimensions.

### Reading — max +0.20, max penalty -0.30
- `+0.05` per relevant file read
- `-0.05` per irrelevant file read
- `-0.15` if agent edits a file it never read
- `-0.10` if agent reads more than 70% of all files

### Diagnosis — max +0.20, max penalty -0.25
- `+0.20` correct diagnosis submitted before any edit
- `+0.10` correct diagnosis submitted after some edits
- `+0.08` partial diagnosis (on the right track)
- `-0.10` no diagnosis submitted
- `-0.15` wrong diagnosis submitted

### Edit — max +0.30, max penalty -0.25
- `+0.30` correct fix, minimal lines changed (5 or fewer)
- `+0.20` correct fix, but touched unnecessary code
- `+0.10` partially correct fix
- `-0.15` fix broke previously passing tests

Edit correctness is determined by silently running the test suite — not string matching. If tests pass, the fix was correct.

### Verification — max +0.20, max penalty -0.10
- `+0.05` ran tests at all
- `+0.10` tests passed after fix
- `+0.05` used trace_execution to verify behavior
- `-0.10` submitted without any verification

### Efficiency — max +0.10, max penalty -0.10
- `+0.10` solved within 60% of max steps
- `+0.05` solved within 80% of max steps
- `-0.10` hit max steps without submitting

### Two-Bug Episodes
For medium and hard tasks each bug contributes 50% of the total score. Bug 2 is revealed automatically when bug 1's tests all pass.

---

## Action Space

```python
# Read actions
{"action_type": "read_file", "filename": "processor.py"}
{"action_type": "read_function", "filename": "processor.py", "function_name": "calculate_avg"}
{"action_type": "search_codebase", "query": "sanitize_username"}

# Diagnose actions
{"action_type": "submit_diagnosis", "diagnosis": "The bug is X because Y"}
{"action_type": "identify_location", "filename": "processor.py", "line_start": 22, "line_end": 22}

# Edit actions
{"action_type": "edit_code", "filename": "processor.py", "old_code": "exact code", "new_code": "fix"}
{"action_type": "add_code", "filename": "processor.py", "new_code": "new code", "location": "bottom"}
{"action_type": "delete_code", "filename": "processor.py", "target_code": "code to remove"}

# Verify actions
{"action_type": "run_tests"}
{"action_type": "trace_execution", "function_name": "calculate_avg", "input_args": {"data": [1,2,3]}}

# Terminal
{"action_type": "submit"}
```

---

## Observation Space

Every step returns a `CodeNavObservation`:

```python
# Always present
obs.success              # bool
obs.message              # str — human-readable result
obs.current_step         # int
obs.max_steps            # int
obs.available_files      # list[str]
obs.files_read           # list[str]
obs.edits_made           # int
obs.diagnosis_submitted  # bool
obs.tests_run            # bool

# Populated by the relevant action
obs.file_content         # str — from read_file, read_function
obs.search_results       # list — from search_codebase
obs.diagnosis_feedback   # str — from submit_diagnosis
obs.diff                 # str — from any edit action
obs.test_results         # dict — from run_tests
obs.execution_trace      # dict — from trace_execution
obs.final_score          # float — from submit
obs.score_breakdown      # dict — per-dimension scores on submit
```

---

## Baseline Results

Evaluated using Llama 3.3 70B (via Groq API), temperature=0, ReAct-style agent loop with full conversation history per step.

| Task | Score | Steps | Notes |
|------|-------|-------|-------|
| Easy | 0.800 | 5 | Perfect workflow — read, diagnose, fix, test, submit |
| Medium | 0.750 | 10 | Both bugs fixed cleanly in one pass |
| Hard | 0.725 | 12 | Both bugs fixed, one re-read needed on bug 2 |

The gap between the scripted ideal agent (0.800 / 0.750 / 0.750) and Llama 3.3 70B is small on easy and medium. Hard tasks expose the model's weakness on exact string construction for edits — the model understands the bug but sometimes needs a second read to get the exact `old_code` right.

---

## Running the Baseline Script

```bash
pip install groq

export OPENAI_API_KEY=gsk_your_groq_key_here

# Single task, pinned scenario
python baseline.py --task easy --scenario 0

# All tasks
python baseline.py --task all

# Multiple episodes for averaged scores
python baseline.py --task easy --episodes 5

# Smaller/faster model to save tokens
python baseline.py --model llama-3.1-8b-instant
```

Results are saved to `baseline_results.json`.

---

## Deploying with Docker

```bash
# Build
docker build -t codenav-env:latest -f server/Dockerfile .

# Run easy task
docker run -e CODENAV_TASK=easy -p 8000:8000 codenav-env:latest

# Run medium task
docker run -e CODENAV_TASK=medium -p 8000:8000 codenav-env:latest

# Run hard task
docker run -e CODENAV_TASK=hard -p 8000:8000 codenav-env:latest
```

---

## Deploying to Hugging Face Spaces

```bash
# From the codenav directory
openenv push

# Push to a specific repo
openenv push --repo-id your-org/codenav

# Push as private
openenv push --private
```

The deployed space exposes:
- `/web` — interactive UI for exploring the environment
- `/docs` — full OpenAPI documentation
- `/health` — health check endpoint
- `/ws` — WebSocket endpoint for low-latency agent loops

---

## Project Structure

```
codenav/
├── models.py                    — CodeNavAction, CodeNavObservation, CodeNavState
├── client.py                    — Typed WebSocket client
├── scenarios.py                 — 19 scenario pool (5 easy, 6 medium, 8 hard)
├── baseline.py                  — Groq/Llama ReAct agent baseline
├── test_all_tasks.py            — Ideal agent regression test
├── openenv.yaml                 — OpenEnv manifest
├── pyproject.toml               — Dependencies
└── server/
    ├── codenav_environment.py   — Full environment logic and reward computation
    ├── app.py                   — FastAPI server (HTTP + WebSocket)
    └── Dockerfile               — Container definition
```

---

## Concurrent Sessions

The server supports multiple concurrent WebSocket connections. To enable factory mode in `server/app.py`:

```python
app = create_app(
    CodeNavEnvironment,   # Pass class, not instance
    CodeNavAction,
    CodeNavObservation,
    max_concurrent_envs=4,
)
```

---

## Design Notes

**Test-based edit correctness.** We do not compare fixes against a template. After every edit, the environment silently runs the full test suite. If all tests pass, the fix was correct. We do not care how you fixed it, only whether it works.

**Distractor files.** Every episode includes files that look relevant but contain no bug. An agent that reads them all gets penalized. An agent that reads selectively gets rewarded. This trains targeted exploration over brute-force reading.

**Silent bugs on hard.** Hard scenarios produce wrong output without raising any error. The agent cannot rely on tracebacks. It must understand the code well enough to predict incorrect behavior from reading alone.

**Two-bug episodes.** Medium and hard tasks have two independent bugs. Bug 2 is always in a different file than bug 1, requiring the agent to shift its mental model mid-episode without a reset.

**Scenario pools.** Each difficulty level draws from a pool of scenarios chosen uniformly at random on each `reset()`. This prevents memorization and requires genuine generalization across different codebases and bug patterns.

---

## License

BSD License. See LICENSE for details.