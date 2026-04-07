"""
CodeNav Inference Script — OpenEnv Hackathon Submission

Mandatory stdout format:
    [START] task=<task> env=codenav model=<model>
    [STEP]  step=<n> action=<action> reward=0.00 done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...>

Environment variables:
    API_BASE_URL      LLM endpoint (default: https://router.huggingface.co/v1)
    MODEL_NAME        Model identifier (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN          Hugging Face / API key
    CODENAV_TASK      Task difficulty: easy, medium, hard (default: easy)
    CODENAV_SPACE_URL CodeNav HF Space URL (default: http://localhost:8000)
"""

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import CodeNavAction, CodeNavObservation
from server.codenav_environment import CodeNavEnvironment


# ---------------------------------------------------------------------------
# Configuration from environment variables
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY")
TASK_NAME    = os.getenv("CODENAV_TASK", "easy")
BENCHMARK    = "codenav"


# ---------------------------------------------------------------------------
# Structured stdout loggers — mandatory format
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert software developer working inside a Python codebase.
Your job is to find and fix bugs by following a careful workflow:

1. READ — Explore relevant files before acting. Never edit a file you haven't read.
2. DIAGNOSE — Submit a diagnosis explaining what is wrong and why before making any edit.
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

CRITICAL RULES:
- Always read a file before editing it
- Always submit_diagnosis before edit_code
- Always run_tests after edit_code
- Only submit when tests pass
- old_code must match file content EXACTLY including whitespace
- If edit_code fails, re-read the file to get exact content before retrying
- Respond with ONLY the JSON object, nothing else
"""


# ---------------------------------------------------------------------------
# Conversation trimmer
# ---------------------------------------------------------------------------

def trim_messages(messages: List[Dict], keep_turns: int = 6) -> List[Dict]:
    """Keep system + opening message + last N turns to control token usage."""
    if len(messages) <= 2 + keep_turns * 2:
        return messages
    fixed = messages[:2]
    recent = messages[-(keep_turns * 2):]
    trim_note = {
        "role": "user",
        "content": "[Earlier conversation trimmed. Continue from current state.]"
    }
    return fixed + [trim_note] + recent


# ---------------------------------------------------------------------------
# Action parser
# ---------------------------------------------------------------------------

def parse_action(response_text: str) -> Optional[CodeNavAction]:
    text = response_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    try:
        data = json.loads(text[start:end])
        return CodeNavAction(**data)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Observation formatter
# ---------------------------------------------------------------------------

def format_observation(obs: CodeNavObservation, step: int) -> str:
    parts = [f"[Step {step} result] {obs.message}"]
    parts.append(
        f"Status: step {obs.current_step}/{obs.max_steps} | "
        f"files_read={obs.files_read} | "
        f"edits={obs.edits_made} | "
        f"diagnosis_submitted={obs.diagnosis_submitted} | "
        f"tests_run={obs.tests_run}"
    )
    if obs.file_content:
        content = obs.file_content
        if len(content) > 3000:
            content = content[:3000] + "\n... [truncated]"
        parts.append(f"--- File content ---\n{content}\n--- End ---")
    if obs.search_results:
        results_text = "\n".join(
            f"  {r['file']}:{r['line']} — {r['content']}"
            for r in obs.search_results[:15]
        )
        parts.append(f"Search results:\n{results_text}")
    if obs.diagnosis_feedback:
        parts.append(f"Diagnosis feedback: {obs.diagnosis_feedback}")
    if obs.diff:
        parts.append(f"Diff applied:\n{obs.diff}")
    if obs.syntax_valid is not None:
        parts.append(f"Syntax valid: {obs.syntax_valid}")
    if obs.scope_warning:
        parts.append(f"WARNING: {obs.scope_warning}")
    if obs.test_results:
        r = obs.test_results
        parts.append(f"Tests: {r['total_passed']} passed, {r['total_failed']} failed")
        if r.get("files"):
            for fname, file_results in r["files"].items():
                if file_results.get("tests"):
                    for test_name, result in file_results["tests"].items():
                        if result["status"] != "PASS":
                            parts.append(
                                f"  FAIL {test_name}: {result.get('error', '')}"
                            )
    if obs.execution_trace:
        parts.append(f"Execution trace: {obs.execution_trace}")
    if obs.final_score is not None:
        parts.append(f"\nFINAL SCORE: {obs.final_score}")
        parts.append(f"Breakdown: {obs.score_breakdown}")
        parts.append("Episode complete.")
    else:
        parts.append("\nWhat is your next action? Respond with JSON only.")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Single episode runner
# ---------------------------------------------------------------------------

def run_episode(
    client: Any,
    task_id: str,
    model: str,
    scenario_index: Optional[int] = 0,
) -> Dict[str, Any]:
    """Run one full episode and return results."""
    env = CodeNavEnvironment(task_id=task_id, scenario_index=scenario_index)
    reset_obs = env.reset()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"TASK: {reset_obs.message}\n\n"
                f"Available files: {reset_obs.available_files}\n"
                f"Max steps: {reset_obs.max_steps}\n\n"
                "Start by reading the relevant files. "
                "Respond with your first action as a JSON object."
            )
        }
    ]

    rewards: List[float] = []
    final_score = 0.0
    steps_taken = 0
    parse_errors = 0
    success = False

    while True:
        trimmed = trim_messages(messages, keep_turns=6)

        try:
            response = client.chat.completions.create(
                model=model,
                messages=trimmed,
                temperature=0.0,
                max_tokens=1024,
            )
            response_text = response.choices[0].message.content.strip()
        except Exception as e:
            log_step(
                step=steps_taken + 1,
                action="api_error",
                reward=0.0,
                done=True,
                error=str(e)[:100],
            )
            break

        action = parse_action(response_text)

        if action is None:
            parse_errors += 1
            messages.append({"role": "assistant", "content": response_text})
            messages.append({
                "role": "user",
                "content": (
                    "ERROR: Invalid JSON. Respond with ONLY a JSON object. Example:\n"
                    '{"action_type": "read_file", "filename": "processor.py"}'
                )
            })
            if parse_errors >= 3:
                log_step(
                    step=steps_taken + 1,
                    action="parse_error",
                    reward=0.0,
                    done=True,
                    error="too_many_parse_errors",
                )
                break
            continue

        parse_errors = 0
        steps_taken += 1

        obs = env.step(action)

        # Reward per step — intermediate steps get 0, final submit gets the score
        step_reward = 0.0
        done = obs.done or action.action_type == "submit"

        if obs.final_score is not None:
            step_reward = obs.final_score
            final_score = obs.final_score
            success = final_score > 0

        rewards.append(step_reward)

        # Action string for log — compact single-line representation
        action_str = action.action_type
        if action.filename:
            action_str += f"({action.filename})"
        if action.diagnosis:
            safe_diag = action.diagnosis[:40].replace(" ", "_")
            action_str += f"[{safe_diag}]"

        error_msg = None
        if not obs.success and action.action_type not in ("submit",):
            error_msg = obs.message[:80].replace("\n", " ")

        log_step(
            step=steps_taken,
            action=action_str,
            reward=step_reward,
            done=done,
            error=error_msg,
        )

        messages.append({"role": "assistant", "content": response_text})
        messages.append({
            "role": "user",
            "content": format_observation(obs, steps_taken)
        })

        if done:
            break

        if steps_taken >= env._task["max_steps"]:
            log_step(
                step=steps_taken + 1,
                action="timeout",
                reward=0.0,
                done=True,
                error="max_steps_reached",
            )
            break

        time.sleep(0.3)

    return {
        "final_score": final_score,
        "steps_taken": steps_taken,
        "rewards": rewards,
        "success": success,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not HF_TOKEN:
        print(
            "ERROR: HF_TOKEN not set. Run: export HF_TOKEN=your_token",
            file=sys.stderr
        )
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    # Run all three tasks — one episode each, scenario 0 pinned for reproducibility
    tasks = ["easy", "medium", "hard"]
    all_scores = []

    for task_id in tasks:
        log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

        result = run_episode(
            client=client,
            task_id=task_id,
            model=MODEL_NAME,
            scenario_index=0,
        )

        all_scores.append(result["final_score"])

        log_end(
            success=result["success"],
            steps=result["steps_taken"],
            score=result["final_score"],
            rewards=result["rewards"],
        )

    # Print summary to stderr so it doesn't pollute the mandatory stdout format
    avg = sum(all_scores) / len(all_scores) if all_scores else 0.0
    print(
        f"\nSummary: easy={all_scores[0]:.3f} medium={all_scores[1]:.3f} "
        f"hard={all_scores[2]:.3f} avg={avg:.3f}",
        file=sys.stderr,
        flush=True,
    )


if __name__ == "__main__":
    main()