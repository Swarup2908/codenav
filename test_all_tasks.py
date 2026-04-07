"""
Full three-task end-to-end test for CodeNav.
Tests easy, medium, and hard tasks with an ideal agent.
Medium and hard now include two-bug episodes.

scenario_index=0 is pinned so the test always runs the known scenarios
(easy_1, medium_1, hard_1) that the ideal agent is scripted for.
The baseline.py script will use random picks to test the full pool.

Usage:
    python test_all_tasks.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import CodeNavAction, CodeNavObservation
from server.codenav_environment import CodeNavEnvironment


def print_obs(label: str, obs: CodeNavObservation):
    print(f"\n  [{label}]")
    print(f"    success   : {obs.success}")
    print(f"    message   : {obs.message[:120]}")
    print(f"    step      : {obs.current_step}/{obs.max_steps}")
    print(f"    files_read: {obs.files_read}")
    print(f"    edits     : {obs.edits_made}")
    if obs.diagnosis_feedback:
        print(f"    diag_feed : {obs.diagnosis_feedback}")
    if obs.test_results:
        r = obs.test_results
        print(f"    tests     : {r['total_passed']} passed, {r['total_failed']} failed")
    if obs.final_score is not None:
        print(f"\n    ★ FINAL SCORE : {obs.final_score}")
        print(f"    breakdown     : {obs.score_breakdown}")


def run_easy():
    print("\n" + "="*65)
    print("  TASK 1 — EASY")
    print("  Richer codebase, distractor files, single bug")
    print("="*65)

    env = CodeNavEnvironment(task_id="easy", scenario_index=0)
    env.reset()

    obs = env.step(CodeNavAction(
        action_type="read_file",
        filename="processor.py"
    ))
    print_obs("read_file processor.py", obs)

    obs = env.step(CodeNavAction(
        action_type="submit_diagnosis",
        diagnosis="The bug is an off-by-one error — averages[cid] = totals[cid] / counts[cid] + 1 incorrectly adds 1 to every average"
    ))
    print_obs("submit_diagnosis", obs)

    obs = env.step(CodeNavAction(
        action_type="edit_code",
        filename="processor.py",
        old_code="        averages[cid] = totals[cid] / counts[cid] + 1  # BUG: off-by-one error",
        new_code="        averages[cid] = totals[cid] / counts[cid]"
    ))
    print_obs("edit_code", obs)

    obs = env.step(CodeNavAction(action_type="run_tests"))
    print_obs("run_tests", obs)

    obs = env.step(CodeNavAction(action_type="submit"))
    print_obs("SUBMIT", obs)

    return obs.final_score


def run_medium():
    print("\n" + "="*65)
    print("  TASK 2 — MEDIUM")
    print("  Two bugs, cross-file dependency, distractor files")
    print("="*65)

    env = CodeNavEnvironment(task_id="medium", scenario_index=0)
    env.reset()

    obs = env.step(CodeNavAction(action_type="read_file", filename="api_handler.py"))
    print_obs("read_file api_handler.py", obs)

    obs = env.step(CodeNavAction(action_type="read_file", filename="utils.py"))
    print_obs("read_file utils.py", obs)

    # BUG 1
    obs = env.step(CodeNavAction(
        action_type="submit_diagnosis",
        diagnosis="sanitize_username returns None for usernames with special characters but api_handler does not check for None before calling clean_username.lower() causing an AttributeError"
    ))
    print_obs("submit_diagnosis bug1", obs)

    obs = env.step(CodeNavAction(
        action_type="edit_code",
        filename="api_handler.py",
        old_code=(
            "    # BUG: no None check — if sanitize_username returns None,\n"
            "    # the next line raises AttributeError (500 error)\n"
            "    if clean_username.lower() not in USER_DB:"
        ),
        new_code=(
            "    if clean_username is None:\n"
            "        return {\"success\": False, \"message\": \"Invalid username\"}\n"
            "    if clean_username.lower() not in USER_DB:"
        )
    ))
    print_obs("edit_code bug1", obs)

    obs = env.step(CodeNavAction(action_type="run_tests"))
    print_obs("run_tests bug1", obs)

    # BUG 2
    obs = env.step(CodeNavAction(
        action_type="submit_diagnosis",
        diagnosis="sanitize_username does not check username length — usernames longer than 20 characters should be rejected to prevent downstream failures"
    ))
    print_obs("submit_diagnosis bug2", obs)

    obs = env.step(CodeNavAction(
        action_type="edit_code",
        filename="utils.py",
        old_code='    if not re.match(r"^[a-zA-Z0-9_]+$", username):',
        new_code=(
            "    if len(username) > 20:\n"
            "        return None\n"
            '    if not re.match(r"^[a-zA-Z0-9_]+$", username):'
        )
    ))
    print_obs("edit_code bug2", obs)

    obs = env.step(CodeNavAction(action_type="run_tests"))
    print_obs("run_tests bug2", obs)

    obs = env.step(CodeNavAction(action_type="submit"))
    print_obs("SUBMIT", obs)

    return obs.final_score


def run_hard():
    print("\n" + "="*65)
    print("  TASK 3 — HARD")
    print("  Two bugs, six files, two distractors, cascading deps")
    print("="*65)

    env = CodeNavEnvironment(task_id="hard", scenario_index=0)
    env.reset()

    for fname in ["config_manager.py", "loader.py", "defaults.py"]:
        obs = env.step(CodeNavAction(action_type="read_file", filename=fname))
        print_obs(f"read_file {fname}", obs)

    obs = env.step(CodeNavAction(
        action_type="search_codebase",
        query="notification_preferences"
    ))
    print_obs("search_codebase", obs)

    # BUG 1
    obs = env.step(CodeNavAction(
        action_type="submit_diagnosis",
        diagnosis=(
            "The bug is in loader.py — the post-process block re-merges theme "
            "using merged['theme'] which has already lost user values from the "
            "shallow merge loop when notification_preferences is also present."
        )
    ))
    print_obs("submit_diagnosis bug1", obs)

    obs = env.step(CodeNavAction(
        action_type="edit_code",
        filename="loader.py",
        old_code=(
            '    # BUG: post-process block re-merges theme using merged["theme"] which has\n'
            '    # already lost user values when notification_preferences is also present\n'
            '    if "notification_preferences" in merged:\n'
            '        merged["notification_preferences"] = {\n'
            '            **DEFAULT_CONFIG.get("notification_preferences", {}),\n'
            '            **merged.get("notification_preferences", {}),\n'
            '        }\n'
            '        if "theme" in DEFAULT_CONFIG:\n'
            '            merged["theme"] = {\n'
            '                **DEFAULT_CONFIG["theme"],\n'
            '                **merged.get("theme", {}),\n'
            '            }'
        ),
        new_code=(
            '    if "notification_preferences" in user_prefs:\n'
            '        merged["notification_preferences"] = {\n'
            '            **DEFAULT_CONFIG.get("notification_preferences", {}),\n'
            '            **user_prefs.get("notification_preferences", {}),\n'
            '        }'
        )
    ))
    print_obs("edit_code bug1", obs)

    obs = env.step(CodeNavAction(action_type="run_tests"))
    print_obs("run_tests bug1", obs)

    # BUG 2
    obs = env.step(CodeNavAction(
        action_type="read_file",
        filename="validator.py"
    ))
    print_obs("read_file validator.py", obs)

    obs = env.step(CodeNavAction(
        action_type="submit_diagnosis",
        diagnosis="validator.py is missing validation for display_density — invalid values like ultra-compact pass through silently instead of falling back to the default comfortable"
    ))
    print_obs("submit_diagnosis bug2", obs)

    obs = env.step(CodeNavAction(
        action_type="edit_code",
        filename="validator.py",
        old_code=(
            '        elif key == "display_density":\n'
            '            # BUG 2: missing validation — invalid densities pass through silently\n'
            '            cleaned[key] = value'
        ),
        new_code=(
            '        elif key == "display_density":\n'
            '            if value not in ALLOWED_DENSITIES:\n'
            '                value = "comfortable"\n'
            '            cleaned[key] = value'
        )
    ))
    print_obs("edit_code bug2", obs)

    obs = env.step(CodeNavAction(action_type="run_tests"))
    print_obs("run_tests bug2", obs)

    obs = env.step(CodeNavAction(action_type="submit"))
    print_obs("SUBMIT", obs)

    return obs.final_score


def main():
    print("\n🚀 CodeNav — Full Three-Task Test")
    print("Ideal agent against all three tasks (medium + hard have two bugs).\n")

    easy_score = run_easy()
    medium_score = run_medium()
    hard_score = run_hard()

    print("\n" + "="*65)
    print("  FINAL RESULTS")
    print("="*65)
    print(f"  Easy   : {easy_score:.3f}")
    print(f"  Medium : {medium_score:.3f}")
    print(f"  Hard   : {hard_score:.3f}")
    print(f"  Average: {(easy_score + medium_score + hard_score) / 3:.3f}")
    print("="*65)
    print()


if __name__ == "__main__":
    main()