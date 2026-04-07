"""
Quick end-to-end test of the CodeNav environment.
Runs a full easy episode directly (no server needed).

Usage:
    python test_episode.py
"""

import sys
import os

# Make sure we can import from the current directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import CodeNavAction, CodeNavObservation
from server.codenav_environment import CodeNavEnvironment


def print_obs(label: str, obs: CodeNavObservation):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  success          : {obs.success}")
    print(f"  message          : {obs.message[:120]}")
    print(f"  step             : {obs.current_step}/{obs.max_steps}")
    print(f"  files_read       : {obs.files_read}")
    print(f"  edits_made       : {obs.edits_made}")
    print(f"  diagnosis_sub    : {obs.diagnosis_submitted}")
    print(f"  tests_run        : {obs.tests_run}")
    if obs.file_content:
        print(f"  file_content     :\n{obs.file_content[:300]}")
    if obs.diagnosis_feedback:
        print(f"  diagnosis_feed   : {obs.diagnosis_feedback}")
    if obs.diff:
        print(f"  diff             :\n{obs.diff}")
    if obs.syntax_valid is not None:
        print(f"  syntax_valid     : {obs.syntax_valid}")
    if obs.test_results:
        print(f"  test_results     : {obs.test_results}")
    if obs.final_score is not None:
        print(f"  FINAL SCORE      : {obs.final_score}")
        print(f"  score_breakdown  : {obs.score_breakdown}")
    print(f"  done             : {obs.done}")
    print(f"  reward           : {obs.reward}")


def main():
    print("\n🚀 CodeNav — Easy Task Episode Test")
    print("Agent plays the role of a developer fixing a bug.\n")

    env = CodeNavEnvironment(task_id="easy")

    # --- Reset ---
    obs = env.reset()
    print_obs("RESET", obs)

    # --- Step 1: Read the main file ---
    obs = env.step(CodeNavAction(
        action_type="read_file",
        filename="processor.py"
    ))
    print_obs("READ processor.py", obs)

    # --- Step 2: Submit diagnosis ---
    obs = env.step(CodeNavAction(
        action_type="submit_diagnosis",
        diagnosis="The bug is an off-by-one error — the averages calculation adds 1 incorrectly: totals[cid] / counts[cid] + 1 should just be totals[cid] / counts[cid]"
    ))
    print_obs("SUBMIT DIAGNOSIS", obs)

    # --- Step 3: Fix the bug ---
    obs = env.step(CodeNavAction(
        action_type="edit_code",
        filename="processor.py",
        old_code="        averages[cid] = totals[cid] / counts[cid] + 1  # BUG: off-by-one error",
        new_code="        averages[cid] = totals[cid] / counts[cid]"
    ))
    print_obs("EDIT CODE — fix the bug", obs)

    # --- Step 4: Run tests ---
    obs = env.step(CodeNavAction(
        action_type="run_tests"
    ))
    print_obs("RUN TESTS", obs)

    # --- Step 5: Submit ---
    obs = env.step(CodeNavAction(
        action_type="submit"
    ))
    print_obs("SUBMIT", obs)

    print("\n✅ Episode complete.\n")


if __name__ == "__main__":
    main()