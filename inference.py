"""
CodeNav Inference Script — OpenEnv Hackathon Submission

Connects to the deployed CodeNav HF Space via OpenEnv async client.

Mandatory stdout format:
    [START] task=<task> env=codenav model=<model>
    [STEP]  step=<n> action=<action> reward=0.00 done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...>

Environment variables:
    API_BASE_URL      LLM endpoint (default: https://router.huggingface.co/v1)
    MODEL_NAME        Model identifier (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN          Hugging Face / API key
    CODENAV_SPACE_URL CodeNav HF Space URL (default: https://swarup29-codenav.hf.space)
    CODENAV_TASK      Task difficulty: easy, medium, hard (default: easy)
"""

import asyncio
import json
import os
import sys
import textwrap
from typing import List, Optional

from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration from environment variables
# ---------------------------------------------------------------------------

API_BASE_URL  = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME    = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
SPACE_URL     = os.getenv("CODENAV_SPACE_URL", "https://swarup29-codenav.hf.space")
TASK_NAME     = os.getenv("CODENAV_TASK", "easy")
BENCHMARK     = "codenav"

MAX_PARSE_ERRORS = 3
TEMPERATURE      = 0.0
MAX_TOKENS       = 1024


# ---------------------------------------------------------------------------
# Mandatory stdout loggers
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
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

def build_user_prompt(obs_message: str, obs_data: dict, step: int) -> str:
    parts = [f"[Step {step}] {obs_message}"]

    if obs_data.get("available_files"):
        parts.append(f"Available files: {obs_data['available_files']}")

    status = (
        f"Status: step {obs_data.get('current_step', step)}/{obs_data.get('max_steps', '?')} | "
        f"files_read={obs_data.get('files_read', [])} | "
        f"edits={obs_data.get('edits_made', 0)} | "
        f"diagnosis_submitted={obs_data.get('diagnosis_submitted', False)} | "
        f"tests_run={obs_data.get('tests_run', False)}"
    )
    parts.append(status)

    if obs_data.get("file_content"):
        content = obs_data["file_content"]
        if len(content) > 3000:
            content = content[:3000] + "\n...[truncated]"
        parts.append(f"File content:\n{content}")

    if obs_data.get("search_results"):
        results = obs_data["search_results"][:15]
        lines = [f"  {r['file']}:{r['line']} — {r['content']}" for r in results]
        parts.append("Search results:\n" + "\n".join(lines))

    if obs_data.get("diagnosis_feedback"):
        parts.append(f"Diagnosis feedback: {obs_data['diagnosis_feedback']}")

    if obs_data.get("diff"):
        parts.append(f"Diff applied:\n{obs_data['diff']}")

    if obs_data.get("test_results"):
        r = obs_data["test_results"]
        parts.append(f"Tests: {r.get('total_passed', 0)} passed, {r.get('total_failed', 0)} failed")
        for fname, fr in (r.get("files") or {}).items():
            for tname, result in (fr.get("tests") or {}).items():
                if result.get("status") != "PASS":
                    parts.append(f"  FAIL {tname}: {result.get('error', '')}")

    if obs_data.get("final_score") is not None:
        parts.append(f"\nFINAL SCORE: {obs_data['final_score']}")
        parts.append("Episode complete.")
    else:
        parts.append("\nWhat is your next action? Respond with JSON only.")

    return "\n".join(parts)


def get_llm_action(client: OpenAI, messages: list) -> str:
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"[DEBUG] LLM error: {e}", file=sys.stderr, flush=True)
        return ""


def trim_messages(messages: list, keep_turns: int = 6) -> list:
    if len(messages) <= 2 + keep_turns * 2:
        return messages
    fixed  = messages[:2]
    recent = messages[-(keep_turns * 2):]
    note   = {"role": "user", "content": "[Earlier conversation trimmed. Continue from current state.]"}
    return fixed + [note] + recent


def parse_action_str(response_text: str) -> Optional[dict]:
    text = response_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Main async episode runner
# ---------------------------------------------------------------------------

async def run_episode(client: OpenAI, task_id: str) -> dict:
    """Run one full CodeNav episode via the deployed HF Space."""

    # Import the typed client — installed from our HF Space
    try:
        from codenav import CodeNavAction, CodeNavEnv
    except ImportError:
        print(
            "[ERROR] codenav client not installed.\n"
            f"Run: pip install git+{SPACE_URL}",
            file=sys.stderr,
        )
        sys.exit(1)

    rewards:     List[float] = []
    final_score: float       = 0.0
    steps_taken: int         = 0
    success:     bool        = False
    parse_errors: int        = 0

    async with CodeNavEnv(base_url=SPACE_URL) as env:

        # Reset episode
        result     = await env.reset()
        obs        = result.observation
        obs_dict   = obs.model_dump() if hasattr(obs, "model_dump") else obs.__dict__

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_user_prompt(obs_dict.get("message", "Episode started."), obs_dict, 0),
            },
        ]

        while True:
            trimmed      = trim_messages(messages)
            response_text = get_llm_action(client, trimmed)

            action_dict = parse_action_str(response_text)

            if action_dict is None:
                parse_errors += 1
                messages.append({"role": "assistant", "content": response_text})
                messages.append({
                    "role": "user",
                    "content": (
                        "ERROR: Invalid JSON. Respond with ONLY a JSON object.\n"
                        'Example: {"action_type": "read_file", "filename": "processor.py"}'
                    ),
                })
                action_str = "parse_error"
                log_step(steps_taken + 1, action_str, 0.0, False, "invalid_json")

                if parse_errors >= MAX_PARSE_ERRORS:
                    break
                continue

            parse_errors = 0

            # Build typed action
            try:
                action = CodeNavAction(**action_dict)
            except Exception as e:
                log_step(steps_taken + 1, "invalid_action", 0.0, True, str(e)[:80])
                break

            steps_taken += 1

            # Step environment
            result  = await env.step(action)
            obs     = result.observation
            obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else obs.__dict__

            reward  = float(result.reward or 0.0)
            done    = bool(result.done)

            if obs_dict.get("final_score") is not None:
                reward      = float(obs_dict["final_score"])
                final_score = reward
                success     = final_score > 0

            rewards.append(reward)

            # Compact action string for log
            action_str = action_dict.get("action_type", "unknown")
            if action_dict.get("filename"):
                action_str += f"({action_dict['filename']})"
            if action_dict.get("diagnosis"):
                safe = action_dict["diagnosis"][:40].replace(" ", "_")
                action_str += f"[{safe}]"

            error_msg = None
            if not obs_dict.get("success", True) and action_dict.get("action_type") != "submit":
                error_msg = str(obs_dict.get("message", ""))[:80].replace("\n", " ")

            log_step(step=steps_taken, action=action_str, reward=reward, done=done, error=error_msg)

            messages.append({"role": "assistant", "content": response_text})
            messages.append({
                "role": "user",
                "content": build_user_prompt(
                    obs_dict.get("message", ""), obs_dict, steps_taken
                ),
            })

            if done or action_dict.get("action_type") == "submit":
                break

            max_steps = obs_dict.get("max_steps", 25)
            if steps_taken >= max_steps:
                log_step(steps_taken + 1, "timeout", 0.0, True, "max_steps_reached")
                break

    return {
        "final_score": final_score,
        "steps_taken": steps_taken,
        "rewards":     rewards,
        "success":     success,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN not set. Run: export HF_TOKEN=your_token", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    tasks  = ["easy", "medium", "hard"]

    for task_id in tasks:
        log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

        result = await run_episode(client=client, task_id=task_id)

        log_end(
            success=result["success"],
            steps=result["steps_taken"],
            score=result["final_score"],
            rewards=result["rewards"],
        )

    # Summary to stderr only — does not pollute mandatory stdout format
    print(f"\n[DONE] Ran {len(tasks)} tasks against {SPACE_URL}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    asyncio.run(main())