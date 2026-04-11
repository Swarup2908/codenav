"""
CodeNav Inference Script — OpenEnv Hackathon Submission

Environment variables:
    API_BASE_URL      LLM endpoint (default: https://router.huggingface.co/v1)
    MODEL_NAME        Model identifier (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN          Hugging Face API key (required)
    CODENAV_SPACE_URL CodeNav HF Space URL (default: https://swarup29-codenav.hf.space)
"""

import json
import os
import sys
import textwrap
from typing import List, Optional

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")
SPACE_URL    = os.getenv("CODENAV_SPACE_URL", "https://swarup29-codenav.hf.space").rstrip("/")
BENCHMARK    = "codenav"

MAX_PARSE_ERRORS = 3
TEMPERATURE      = 0.0
MAX_TOKENS       = 1024

# ---------------------------------------------------------------------------
# Mandatory stdout loggers
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.3f} "
        f"done={str(done).lower()} error={error if error else 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"rewards={','.join(f'{r:.3f}' for r in rewards)}",
        flush=True,
    )

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert software developer working inside a Python codebase.
    Your job is to find and fix bugs by following a careful workflow:

    1. READ — Explore relevant files before acting. Never edit without reading first.
    2. DIAGNOSE — Submit a diagnosis explaining the bug before making any edit.
    3. ACT — Make the minimal targeted fix. Change only what needs to change.
    4. VERIFY — Run tests after your fix to confirm it works.
    5. SUBMIT — Only submit when tests pass.

    Respond with ONLY a valid JSON object — no explanation, no markdown, no code blocks.

    Available actions:
      {"action_type": "read_file", "filename": "example.py"}
      {"action_type": "read_function", "filename": "example.py", "function_name": "my_func"}
      {"action_type": "search_codebase", "query": "search term"}
      {"action_type": "submit_diagnosis", "diagnosis": "The bug is X because Y"}
      {"action_type": "edit_code", "filename": "example.py", "old_code": "exact code", "new_code": "fix"}
      {"action_type": "add_code", "filename": "example.py", "new_code": "code", "location": "bottom"}
      {"action_type": "delete_code", "filename": "example.py", "target_code": "code to remove"}
      {"action_type": "run_tests"}
      {"action_type": "trace_execution", "function_name": "my_func", "input_args": {"x": 1}}
      {"action_type": "submit"}

    CRITICAL: old_code must match file content EXACTLY. Read the file before editing.
    Respond with ONLY the JSON object, nothing else.
""").strip()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_prompt(obs: dict, step: int) -> str:
    parts = [f"[Step {step}] {obs.get('message', '')}"]
    if obs.get("available_files"):
        parts.append(f"Available files: {obs['available_files']}")
    parts.append(
        f"Status: step {obs.get('current_step', step)}/{obs.get('max_steps', '?')} | "
        f"files_read={obs.get('files_read', [])} | edits={obs.get('edits_made', 0)} | "
        f"diagnosis_submitted={obs.get('diagnosis_submitted', False)} | "
        f"tests_run={obs.get('tests_run', False)}"
    )
    if obs.get("file_content"):
        content = obs["file_content"]
        if len(content) > 3000:
            content = content[:3000] + "\n...[truncated]"
        parts.append(f"File content:\n{content}")
    if obs.get("search_results"):
        lines = [f"  {r['file']}:{r['line']} — {r['content']}" for r in obs["search_results"][:15]]
        parts.append("Search results:\n" + "\n".join(lines))
    if obs.get("diagnosis_feedback"):
        parts.append(f"Diagnosis feedback: {obs['diagnosis_feedback']}")
    if obs.get("diff"):
        parts.append(f"Diff applied:\n{obs['diff']}")
    if obs.get("test_results"):
        r = obs["test_results"]
        parts.append(f"Tests: {r.get('total_passed', 0)} passed, {r.get('total_failed', 0)} failed")
        for fname, fr in (r.get("files") or {}).items():
            for tname, result in (fr.get("tests") or {}).items():
                if result.get("status") != "PASS":
                    parts.append(f"  FAIL {tname}: {result.get('error', '')}")
    if obs.get("final_score") is not None:
        parts.append(f"\nFINAL SCORE: {obs['final_score']}\nEpisode complete.")
    else:
        parts.append("\nWhat is your next action? Respond with JSON only.")
    return "\n".join(parts)


def get_action(client: OpenAI, messages: list) -> str:
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME, messages=messages,
            temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"[DEBUG] LLM error: {e}", file=sys.stderr, flush=True)
        return ""


def trim(messages: list, keep: int = 6) -> list:
    if len(messages) <= 2 + keep * 2:
        return messages
    return messages[:2] + [{"role": "user", "content": "[Earlier conversation trimmed.]"}] + messages[-(keep * 2):]


def parse(text: str) -> Optional[dict]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
    s, e = text.find("{"), text.rfind("}") + 1
    if s == -1 or e == 0:
        return None
    try:
        return json.loads(text[s:e])
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(client: OpenAI, task_id: str) -> dict:
    rewards: List[float] = []
    final_score = 0.001
    steps_taken = 0
    success = False
    parse_errors = 0

    # Reset
    try:
        resp = requests.post(f"{SPACE_URL}/reset", json={"task_id": task_id}, timeout=60)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[DEBUG] Reset failed: {e}", file=sys.stderr, flush=True)
        log_end(False, 0, [])
        return {"final_score": 0.001, "steps_taken": 0, "rewards": [], "success": False}

    obs = data.get("observation", {})
    max_steps = obs.get("max_steps", 25)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(obs, 0)},
    ]

    while True:
        response_text = get_action(client, trim(messages))
        action_dict = parse(response_text)

        if action_dict is None:
            parse_errors += 1
            messages.append({"role": "assistant", "content": response_text})
            messages.append({"role": "user", "content": "ERROR: Invalid JSON. Respond with ONLY a JSON object."})
            log_step(steps_taken + 1, "parse_error", 0.001, False, "invalid_json")
            if parse_errors >= MAX_PARSE_ERRORS:
                break
            continue

        parse_errors = 0
        steps_taken += 1

        try:
            resp = requests.post(f"{SPACE_URL}/step", json={"action": action_dict}, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            log_step(steps_taken, action_dict.get("action_type", "unknown"), 0.001, True, str(e)[:80])
            break

        obs = data.get("observation", {})
        reward = max(0.001, float(data.get("reward", 0.001)))
        done = bool(data.get("done", False))

        if obs.get("final_score") is not None:
            reward = float(obs["final_score"])
            final_score = reward
            success = final_score > 0

        rewards.append(reward)

        action_str = action_dict.get("action_type", "unknown")
        if action_dict.get("filename"):
            action_str += f"({action_dict['filename']})"

        error_msg = None
        if not obs.get("success", True) and action_dict.get("action_type") != "submit":
            error_msg = str(obs.get("message", ""))[:80].replace("\n", " ")

        log_step(steps_taken, action_str, reward, done, error_msg)

        messages.append({"role": "assistant", "content": response_text})
        messages.append({"role": "user", "content": build_prompt(obs, steps_taken)})

        if done or action_dict.get("action_type") == "submit":
            break

        max_steps = obs.get("max_steps", max_steps)
        if steps_taken >= max_steps:
            log_step(steps_taken + 1, "timeout", 0.001, True, "max_steps_reached")
            break

    return {"final_score": final_score, "steps_taken": steps_taken, "rewards": rewards, "success": success}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN not set.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    for task_id in ["easy", "medium", "hard"]:
        log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
        result = run_episode(client=client, task_id=task_id)
        clamped_rewards = [max(0.001, min(0.999, r)) for r in result["rewards"]] if result["rewards"] else [0.001]
        log_end(success=result["success"], steps=result["steps_taken"], rewards=clamped_rewards)


if __name__ == "__main__":
    main()