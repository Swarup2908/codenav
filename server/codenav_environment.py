# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
CodeNav Environment Implementation.

An RL environment where an agent acts as a software developer dropped into
an unfamiliar Python codebase. The agent must read, diagnose, act, verify,
and submit — rewarded for the quality of the entire process, not just the
final output.

Three tasks of increasing difficulty:
    easy   — single file, obvious bug, no cross-file dependencies
    medium — two/three files, non-obvious bug, one dependency to reason about
    hard   — multiple files, subtle bug, cascading dependencies
"""

import ast
import difflib
import re
from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import CodeNavAction, CodeNavObservation, CodeNavState
except ImportError:
    from models import CodeNavAction, CodeNavObservation, CodeNavState


# ---------------------------------------------------------------------------
# Scenario Pool Import
# ---------------------------------------------------------------------------

try:
    from ..scenarios import get_scenario, get_pool_size
except ImportError:
    from scenarios import get_scenario, get_pool_size
# ---------------------------------------------------------------------------
# Reward Computer
# ---------------------------------------------------------------------------

class RewardComputer:
    """Computes reward signals based on agent behavior tracked in state."""

    # Reading intelligently
    RELEVANT_FILE_READ = 0.05
    IRRELEVANT_FILE_READ = -0.05
    EDIT_WITHOUT_READING = -0.15
    BRUTE_FORCE_READ_PENALTY = -0.10   # read >70% of files

    # Diagnosis quality
    CORRECT_DIAGNOSIS_EARLY = 0.20     # before any edit
    CORRECT_DIAGNOSIS_LATE = 0.10      # after some edits
    PARTIAL_DIAGNOSIS = 0.08
    NO_DIAGNOSIS = -0.10
    WRONG_DIAGNOSIS = -0.15
    REPEATED_WRONG_DIAGNOSIS = -0.05   # per extra wrong attempt

    # Edit quality
    CORRECT_MINIMAL_FIX = 0.30
    CORRECT_NONMINIMAL_FIX = 0.20
    PARTIAL_FIX = 0.10
    FIX_INTRODUCED_BUG = -0.10
    FIX_BROKE_TESTS = -0.15

    # Verification quality
    RAN_TESTS = 0.05
    TESTS_PASSED = 0.10
    TRACED_EXECUTION = 0.05
    SUBMIT_WITHOUT_VERIFY = -0.10

    # Efficiency
    EFFICIENCY_HIGH = 0.10    # solved within 60% of max steps
    EFFICIENCY_MED = 0.05     # solved within 80% of max steps
    HIT_MAX_STEPS = -0.10

    def compute_final_reward(self, state: CodeNavState, task: dict) -> Dict[str, float]:
        """
        Compute the full reward breakdown at episode end.

        Returns a dict with per-dimension scores and total.
        """
        breakdown = {}

        # --- Reading ---
        read_score = 0.0
        for f in state.relevant_files_read:
            read_score += self.RELEVANT_FILE_READ
        for f in state.irrelevant_files_read:
            read_score += self.IRRELEVANT_FILE_READ
        if state.read_more_than_threshold:
            read_score += self.BRUTE_FORCE_READ_PENALTY
        breakdown["reading"] = round(read_score, 3)

        # --- Diagnosis ---
        diag_score = 0.0
        if not state.diagnosis_submitted:
            diag_score = self.NO_DIAGNOSIS
        elif state.diagnosis_correct:
            if state.diagnosis_before_edit:
                diag_score = self.CORRECT_DIAGNOSIS_EARLY
            else:
                diag_score = self.CORRECT_DIAGNOSIS_LATE
        elif state.diagnosis_partial:
            diag_score = self.PARTIAL_DIAGNOSIS
        else:
            diag_score = self.WRONG_DIAGNOSIS
        breakdown["diagnosis"] = round(diag_score, 3)

        # --- Edit ---
        edit_score = 0.0
        if state.edit_correct is not None:
            if state.edit_correct and state.edit_minimal:
                edit_score = self.CORRECT_MINIMAL_FIX
            elif state.edit_correct:
                edit_score = self.CORRECT_NONMINIMAL_FIX
            else:
                edit_score = self.PARTIAL_FIX
            if state.tests_passed is False and state.tests_run:
                edit_score += self.FIX_BROKE_TESTS
        breakdown["edit"] = round(edit_score, 3)

        # --- Verification ---
        verify_score = 0.0
        if state.tests_run:
            verify_score += self.RAN_TESTS
            if state.tests_passed:
                verify_score += self.TESTS_PASSED
        if not state.verified_before_submit:
            verify_score += self.SUBMIT_WITHOUT_VERIFY
        breakdown["verification"] = round(verify_score, 3)

        # --- Efficiency ---
        eff_score = 0.0
        ratio = state.step_count / state.max_steps
        if state.done and state.edit_correct:
            if ratio <= 0.60:
                eff_score = self.EFFICIENCY_HIGH
            elif ratio <= 0.80:
                eff_score = self.EFFICIENCY_MED
        if state.step_count >= state.max_steps:
            eff_score += self.HIT_MAX_STEPS
        breakdown["efficiency"] = round(eff_score, 3)

        raw_total = round(sum(breakdown.values()), 3)

        # For two-bug episodes, scale bug_1 contribution to 50%
        # Bug_2 reward will be added when bug_2 is submitted
        if state.bug_2_exists:
            raw_total = round(raw_total * 0.5, 3)

        breakdown["total"] = raw_total
        return breakdown


    def compute_bug_2_reward(self, state: "CodeNavState") -> "Dict[str, float]":
        """
        Compute reward for bug_2 performance.
        Same dimensions as bug_1 but based on bug_2 state.
        Contributes the other 50% of total episode reward.
        """
        breakdown = {}

        # Diagnosis for bug_2
        if not state.diagnosis_submitted:
            breakdown["diagnosis"] = self.NO_DIAGNOSIS
        elif state.diagnosis_correct:
            breakdown["diagnosis"] = (
                self.CORRECT_DIAGNOSIS_EARLY
                if state.diagnosis_before_edit
                else self.CORRECT_DIAGNOSIS_LATE
            )
        elif state.diagnosis_partial:
            breakdown["diagnosis"] = self.PARTIAL_DIAGNOSIS
        else:
            breakdown["diagnosis"] = self.WRONG_DIAGNOSIS

        # Edit for bug_2
        if state.edit_correct:
            breakdown["edit"] = (
                self.CORRECT_MINIMAL_FIX
                if state.edit_minimal
                else self.CORRECT_NONMINIMAL_FIX
            )
        else:
            breakdown["edit"] = self.PARTIAL_FIX

        # Verification for bug_2
        verify = 0.0
        if state.tests_run:
            verify += self.RAN_TESTS
            if state.tests_passed:
                verify += self.TESTS_PASSED
        if not state.verified_before_submit:
            verify += self.SUBMIT_WITHOUT_VERIFY
        breakdown["verification"] = round(verify, 3)

        # Scale to 50% of total
        raw = round(sum(breakdown.values()), 3)
        breakdown["total"] = round(raw * 0.5, 3)
        return breakdown


# ---------------------------------------------------------------------------
# Main Environment
# ---------------------------------------------------------------------------

class CodeNavEnvironment(Environment):
    """
    CodeNav: Developer Workflow RL Environment.

    Places an agent in the role of a Python developer dropped into an
    unfamiliar synthetic codebase. Rewards quality of the entire reasoning
    process — not just final output correctness.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, task_id: str = "easy", scenario_index: int = None):
        """
        Initialize CodeNav for a specific task.

        Args:
            task_id: one of 'easy', 'medium', 'hard'
            scenario_index: specific scenario to use (None = random each reset)
        """
        if task_id not in ("easy", "medium", "hard"):
            raise ValueError(f"task_id must be one of ['easy', 'medium', 'hard']")

        self._task_id = task_id
        self._scenario_index = scenario_index
        # Load initial scenario — will be refreshed on each reset() call
        self._task = get_scenario(task_id, scenario_index)
        self._active_task = self._task
        self._reward_computer = RewardComputer()
        self._state: Optional[CodeNavState] = None
        self._files: Dict[str, str] = {}
        self._wrong_diagnosis_count: int = 0

    # -----------------------------------------------------------------------
    # OpenEnv Interface
    # -----------------------------------------------------------------------

    def reset(self) -> CodeNavObservation:
        """Reset the environment and return the initial observation."""
        # Pick a fresh scenario each episode (random unless index is pinned)
        self._task = get_scenario(self._task_id, self._scenario_index)
        self._active_task = self._task

        self._files = dict(self._task["files"])
        self._wrong_diagnosis_count = 0

        self._state = CodeNavState(
            episode_id=str(uuid4()),
            task_id=self._task_id,
            step_count=0,
            max_steps=self._task["max_steps"],
            done=False,
            bug_2_exists=self._task.get("bug_2") is not None,
        )

        return CodeNavObservation(
            success=True,
            message=(
                f"Episode started. Task: {self._task['description']}\n"
                f"Available files: {list(self._files.keys())}\n"
                "Begin by reading the relevant files."
            ),
            current_step=0,
            max_steps=self._task["max_steps"],
            available_files=list(self._files.keys()),
            files_read=[],
            edits_made=0,
            diagnosis_submitted=False,
            tests_run=False,
            done=False,
            reward=0.0,
        )

    def step(self, action: CodeNavAction) -> CodeNavObservation:  # type: ignore[override]
        """Execute one action and return the resulting observation."""
        if self._state is None:
            raise RuntimeError("Call reset() before step()")

        if self._state.done:
            return self._make_obs(
                success=False,
                message="Episode is already done. Call reset() to start a new episode.",
            )

        self._state.step_count += 1

        # Route to the correct handler
        handlers = {
            "read_file": self._handle_read_file,
            "read_function": self._handle_read_function,
            "search_codebase": self._handle_search_codebase,
            "submit_diagnosis": self._handle_submit_diagnosis,
            "identify_location": self._handle_identify_location,
            "edit_code": self._handle_edit_code,
            "add_code": self._handle_add_code,
            "delete_code": self._handle_delete_code,
            "run_tests": self._handle_run_tests,
            "trace_execution": self._handle_trace_execution,
            "submit": self._handle_submit,
        }

        handler = handlers.get(action.action_type)
        if handler is None:
            return self._make_obs(
                success=False,
                message=f"Unknown action_type: {action.action_type}",
            )

        obs = handler(action)

        # Check step limit
        if self._state.step_count >= self._state.max_steps and not self._state.done:
            self._state.done = True
            obs.message += "\n[MAX STEPS REACHED — episode ending]"
            obs.done = True

        return obs

    @property
    def state(self) -> CodeNavState:
        """Return current episode state."""
        if self._state is None:
            raise RuntimeError("Call reset() before accessing state")
        return self._state

    # -----------------------------------------------------------------------
    # Action Handlers
    # -----------------------------------------------------------------------

    def _handle_read_file(self, action: CodeNavAction) -> CodeNavObservation:
        fname = action.filename
        if not fname or fname not in self._files:
            return self._make_obs(
                success=False,
                message=f"File '{fname}' not found. Available: {list(self._files.keys())}",
            )

        content = self._files[fname]
        line_count = len(content.splitlines())

        # Track reading
        if fname not in self._state.files_read:
            self._state.files_read.append(fname)
            if fname in self._task["relevant_files"]:
                self._state.relevant_files_read.append(fname)
            else:
                self._state.irrelevant_files_read.append(fname)

        # Brute force check
        total = len(self._files)
        if len(self._state.files_read) / total > 0.70:
            self._state.read_more_than_threshold = True

        return self._make_obs(
            success=True,
            message=f"Read file '{fname}' ({line_count} lines)",
            file_content=content,
            file_line_count=line_count,
        )

    def _handle_read_function(self, action: CodeNavAction) -> CodeNavObservation:
        fname = action.filename
        func_name = action.function_name

        if not fname or fname not in self._files:
            return self._make_obs(
                success=False,
                message=f"File '{fname}' not found.",
            )
        if not func_name:
            return self._make_obs(
                success=False,
                message="function_name is required for read_function action.",
            )

        content = self._files[fname]
        extracted = self._extract_function(content, func_name)

        if extracted is None:
            return self._make_obs(
                success=False,
                message=f"Function '{func_name}' not found in '{fname}'.",
            )

        # Count as reading the file
        if fname not in self._state.files_read:
            self._state.files_read.append(fname)
            if fname in self._task["relevant_files"]:
                self._state.relevant_files_read.append(fname)
            else:
                self._state.irrelevant_files_read.append(fname)

        return self._make_obs(
            success=True,
            message=f"Read function '{func_name}' from '{fname}'",
            file_content=extracted,
            file_line_count=len(extracted.splitlines()),
        )

    def _handle_search_codebase(self, action: CodeNavAction) -> CodeNavObservation:
        query = action.query
        if not query:
            return self._make_obs(
                success=False,
                message="query is required for search_codebase action.",
            )

        results = []
        for fname, content in self._files.items():
            for i, line in enumerate(content.splitlines(), 1):
                if query.lower() in line.lower():
                    results.append({
                        "file": fname,
                        "line": i,
                        "content": line.strip(),
                    })

        return self._make_obs(
            success=True,
            message=f"Search for '{query}' returned {len(results)} results.",
            search_results=results,
        )

    def _handle_submit_diagnosis(self, action: CodeNavAction) -> CodeNavObservation:
        diagnosis = action.diagnosis
        if not diagnosis:
            return self._make_obs(
                success=False,
                message="diagnosis is required for submit_diagnosis action.",
            )

        keywords = self._active_task["correct_diagnosis_keywords"]
        diagnosis_lower = diagnosis.lower()

        is_correct = any(kw.lower() in diagnosis_lower for kw in keywords)
        is_partial = not is_correct and any(
            kw.split()[0].lower() in diagnosis_lower for kw in keywords
        )

        if is_correct:
            self._state.diagnosis_correct = True
            self._state.diagnosis_partial = False
            if self._state.edits_made == 0:
                self._state.diagnosis_before_edit = True
            feedback = "Your diagnosis matches the root cause. Proceed with the fix."
        elif is_partial:
            self._state.diagnosis_partial = True
            feedback = "You are on the right track but the diagnosis is incomplete."
            self._wrong_diagnosis_count += 1
        else:
            self._state.diagnosis_correct = False
            self._state.diagnosis_partial = False
            feedback = "Diagnosis does not match the root cause. Keep investigating."
            self._wrong_diagnosis_count += 1

        self._state.diagnosis_submitted = True

        return self._make_obs(
            success=True,
            message=f"Diagnosis submitted.",
            diagnosis_feedback=feedback,
        )

    def _handle_identify_location(self, action: CodeNavAction) -> CodeNavObservation:
        fname = action.filename
        line_start = action.line_start
        line_end = action.line_end

        bug_loc = self._task["bug_location"]
        correct_file = bug_loc["file"]
        correct_start = bug_loc["line_start"]
        correct_end = bug_loc["line_end"]

        if fname == correct_file and line_start is not None and line_end is not None:
            overlap = not (line_end < correct_start or line_start > correct_end)
            if overlap:
                feedback = "Location identified correctly — the bug is in this region."
            else:
                feedback = "Wrong line range in the right file. Look more carefully."
        else:
            feedback = f"Wrong file. The bug is not in '{fname}'."

        return self._make_obs(
            success=True,
            message="Location submitted.",
            diagnosis_feedback=feedback,
        )

    def _handle_edit_code(self, action: CodeNavAction) -> CodeNavObservation:
        fname = action.filename
        old_code = action.old_code
        new_code = action.new_code

        if not fname or fname not in self._files:
            return self._make_obs(
                success=False,
                message=f"File '{fname}' not found.",
            )
        if old_code is None or new_code is None:
            return self._make_obs(
                success=False,
                message="old_code and new_code are required for edit_code.",
            )

        # Penalty if agent edits without reading
        step_penalty = 0.0
        if fname not in self._state.files_read:
            step_penalty = RewardComputer.EDIT_WITHOUT_READING
            self._state.cumulative_reward += step_penalty

        content = self._files[fname]
        if old_code not in content:
            return self._make_obs(
                success=False,
                message="old_code not found in file. Check exact whitespace and indentation.",
            )

        new_content = content.replace(old_code, new_code, 1)
        diff = "\n".join(difflib.unified_diff(
            content.splitlines(),
            new_content.splitlines(),
            fromfile=f"a/{fname}",
            tofile=f"b/{fname}",
            lineterm="",
        ))

        # Syntax check
        syntax_valid = self._check_syntax(new_content)
        if not syntax_valid:
            return self._make_obs(
                success=False,
                message="Edit rejected — result is not valid Python. Check your syntax.",
                diff=diff,
                syntax_valid=False,
            )

        self._files[fname] = new_content
        self._state.edits_made += 1

        # Check edit correctness by silently running tests.
        # Most honest signal — did the fix actually work?
        # We do NOT update state.tests_run or state.tests_passed here —
        # those only change when the agent explicitly calls run_tests.
        silent_result = self._silent_test_run()
        self._state.edit_correct = silent_result

        if self._state.edit_correct:
            # Minimal = agent changed 5 or fewer lines
            lines_changed = max(
                len(old_code.strip().splitlines()),
                len(new_code.strip().splitlines())
            )
            self._state.edit_minimal = lines_changed <= 5

        # Scope warning
        scope_warning = None
        irrelevant = self._task.get("irrelevant_files", [])
        if fname in irrelevant:
            scope_warning = f"Warning: '{fname}' is outside the expected scope of this fix."

        msg = f"Edit applied to '{fname}'."
        if step_penalty < 0:
            msg += f" Warning: you edited a file you hadn't read first ({step_penalty} reward penalty)."

        return self._make_obs(
            success=True,
            message=msg,
            diff=diff,
            syntax_valid=True,
            scope_warning=scope_warning,
        )

    def _handle_add_code(self, action: CodeNavAction) -> CodeNavObservation:
        fname = action.filename
        new_code = action.new_code
        location = action.location or "bottom"

        if not fname or fname not in self._files:
            return self._make_obs(success=False, message=f"File '{fname}' not found.")
        if not new_code:
            return self._make_obs(success=False, message="new_code is required for add_code.")

        content = self._files[fname]

        if location == "top":
            new_content = new_code + "\n" + content
        elif location.startswith("after:"):
            func_name = location.split("after:", 1)[1]
            new_content = self._insert_after_function(content, func_name, new_code)
            if new_content is None:
                return self._make_obs(
                    success=False,
                    message=f"Function '{func_name}' not found for after: insertion.",
                )
        else:
            new_content = content + "\n" + new_code

        if not self._check_syntax(new_content):
            return self._make_obs(
                success=False,
                message="Add rejected — result is not valid Python.",
                syntax_valid=False,
            )

        diff = "\n".join(difflib.unified_diff(
            content.splitlines(), new_content.splitlines(),
            fromfile=f"a/{fname}", tofile=f"b/{fname}", lineterm="",
        ))
        self._files[fname] = new_content
        self._state.edits_made += 1

        return self._make_obs(
            success=True,
            message=f"Code added to '{fname}' at location '{location}'.",
            diff=diff,
            syntax_valid=True,
        )

    def _handle_delete_code(self, action: CodeNavAction) -> CodeNavObservation:
        fname = action.filename
        target_code = action.target_code

        if not fname or fname not in self._files:
            return self._make_obs(success=False, message=f"File '{fname}' not found.")
        if not target_code:
            return self._make_obs(success=False, message="target_code is required for delete_code.")

        content = self._files[fname]
        if target_code not in content:
            return self._make_obs(
                success=False,
                message="target_code not found in file. Check exact whitespace.",
            )

        new_content = content.replace(target_code, "", 1)
        if not self._check_syntax(new_content):
            return self._make_obs(
                success=False,
                message="Delete rejected — result is not valid Python.",
                syntax_valid=False,
            )

        diff = "\n".join(difflib.unified_diff(
            content.splitlines(), new_content.splitlines(),
            fromfile=f"a/{fname}", tofile=f"b/{fname}", lineterm="",
        ))
        self._files[fname] = new_content
        self._state.edits_made += 1

        return self._make_obs(
            success=True,
            message=f"Code deleted from '{fname}'.",
            diff=diff,
            syntax_valid=True,
        )

    def _handle_run_tests(self, action: CodeNavAction) -> CodeNavObservation:
        self._state.tests_run = True
        self._state.verified_before_submit = True

        # Run all test files in the current file set
        test_files = {k: v for k, v in self._files.items() if k.startswith("test_")}
        all_results = {}
        overall_passed = True

        for test_fname, test_content in test_files.items():
            results = self._run_test_file(test_fname, test_content)
            all_results[test_fname] = results
            if not results.get("all_passed", False):
                overall_passed = False

        self._state.tests_passed = overall_passed

        total_pass = sum(r.get("passed", 0) for r in all_results.values())
        total_fail = sum(r.get("failed", 0) for r in all_results.values())

        test_result_payload = {
            "files": all_results,
            "overall_passed": overall_passed,
            "total_passed": total_pass,
            "total_failed": total_fail,
        }

        # Check if bug_1 just got fixed and bug_2 exists
        bug_2_def = self._task.get("bug_2")
        if (
            overall_passed
            and self._state.current_bug == 1
            and self._state.bug_2_exists
            and bug_2_def is not None
            and not self._state.bug_1_fixed
        ):
            self._state.bug_1_fixed = True
            self._state.current_bug = 2
            self._state.diagnosis_submitted = False
            self._wrong_diagnosis_count = 0

            # Inject additional test cases for bug_2 into the files
            for fname, extra_tests in bug_2_def.get("test_additions", {}).items():
                if fname in self._files:
                    self._files[fname] += extra_tests

            # Switch active task config to bug_2
            self._active_task = {
                **self._task,
                "relevant_files": bug_2_def["relevant_files"],
                "correct_diagnosis_keywords": bug_2_def["correct_diagnosis_keywords"],
                "correct_fix": bug_2_def["correct_fix"],
            }

            # Reset per-bug tracking fields for bug_2
            self._state.diagnosis_correct = None
            self._state.diagnosis_partial = None
            self._state.diagnosis_before_edit = False
            self._state.edit_correct = None
            self._state.edit_minimal = None
            self._state.tests_run = False
            self._state.tests_passed = None
            self._state.verified_before_submit = False
            self._state.edits_made = 0

            bug_2_msg = (
                f"\n\n{'='*50}\n"
                f"Bug 1 fixed! New issue reported:\n"
                f"{bug_2_def['description']}\n"
                f"Continue investigating."
            )

            return self._make_obs(
                success=True,
                message=f"Tests complete: {total_pass} passed, {total_fail} failed.{bug_2_msg}",
                test_results=test_result_payload,
            )

        # Check if bug_2 just got fixed
        if (
            overall_passed
            and self._state.current_bug == 2
            and not self._state.bug_2_fixed
        ):
            self._state.bug_2_fixed = True

        return self._make_obs(
            success=True,
            message=f"Tests complete: {total_pass} passed, {total_fail} failed.",
            test_results=test_result_payload,
        )

    def _handle_trace_execution(self, action: CodeNavAction) -> CodeNavObservation:
        func_name = action.function_name
        input_args = action.input_args or {}

        if not func_name:
            return self._make_obs(
                success=False,
                message="function_name is required for trace_execution.",
            )

        self._state.verified_before_submit = True

        # Find the file containing this function
        target_file = None
        for fname, content in self._files.items():
            if f"def {func_name}" in content:
                target_file = (fname, content)
                break

        if target_file is None:
            return self._make_obs(
                success=False,
                message=f"Function '{func_name}' not found in any file.",
            )

        fname, content = target_file
        result = self._safe_execute_function(content, func_name, input_args)

        return self._make_obs(
            success=True,
            message=f"Traced execution of '{func_name}' from '{fname}'.",
            execution_trace=result,
        )

    def _handle_submit(self, action: CodeNavAction) -> CodeNavObservation:
        if self._state is None:
            raise RuntimeError("Call reset() first.")

        # If bug_2 exists and not yet fixed, warn agent before ending
        if (
            self._state.bug_2_exists
            and self._state.bug_1_fixed
            and not self._state.bug_2_fixed
            and self._state.current_bug == 2
        ):
            # Allow submit but penalize — bug_2 was not fixed
            pass

        self._state.done = True

        # Final reward computation
        breakdown = self._reward_computer.compute_final_reward(self._state, self._task)

        # For two-bug episodes, add bug_2 contribution
        if self._state.bug_2_exists:
            bug_2_breakdown = self._reward_computer.compute_bug_2_reward(self._state)
            breakdown["bug_2_total"] = bug_2_breakdown["total"]
            breakdown["total"] = round(
                breakdown["total"] + bug_2_breakdown["total"], 3
            )
            breakdown.update({f"bug2_{k}": v for k, v in bug_2_breakdown.items()
                               if k != "total"})

        total = breakdown["total"]
        self._state.cumulative_reward += total

        return self._make_obs(
            success=True,
            message=f"Episode complete. Final score: {total:.3f}",
            final_score=total,
            score_breakdown=breakdown,
            done=True,
            reward=total,
        )

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _make_obs(self, success: bool, message: str, **kwargs) -> CodeNavObservation:
        """Build a CodeNavObservation with the current persistent state."""
        if self._state is None:
            step = 0
            max_s = 0
            files = []
            read = []
            edits = 0
            diag = False
            tests = False
        else:
            step = self._state.step_count
            max_s = self._state.max_steps
            files = list(self._files.keys())
            read = list(self._state.files_read)
            edits = self._state.edits_made
            diag = self._state.diagnosis_submitted
            tests = self._state.tests_run

        return CodeNavObservation(
            success=success,
            message=message,
            current_step=step,
            max_steps=max_s,
            available_files=files,
            files_read=read,
            edits_made=edits,
            diagnosis_submitted=diag,
            tests_run=tests,
            done=kwargs.pop("done", False),
            reward=kwargs.pop("reward", 0.0),
            **kwargs,
        )

    def _extract_function(self, source: str, func_name: str) -> Optional[str]:
        """Extract a single function's source from a Python file."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return None

        lines = source.splitlines()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                start = node.lineno - 1
                end = node.end_lineno
                return "\n".join(lines[start:end])
        return None

    def _insert_after_function(self, source: str, func_name: str, new_code: str) -> Optional[str]:
        """Insert code after a named function."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return None

        lines = source.splitlines()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                insert_at = node.end_lineno
                lines.insert(insert_at, new_code)
                return "\n".join(lines)
        return None

    def _check_syntax(self, source: str) -> bool:
        """Return True if source is valid Python."""
        try:
            ast.parse(source)
            return True
        except SyntaxError:
            return False

    def _build_module_registry(self) -> "Dict[str, Any]":
        """
        Build an in-memory module registry from all current non-test files.

        Two-pass exec strategy:
        - Pass 1: register all modules as empty shells in sys.modules,
          then exec all files. Files with no cross-imports succeed.
        - Pass 2: re-exec files that failed in pass 1. By now their
          dependencies are populated, so cross-imports resolve.

        This handles one level of dependency depth which covers all
        three of our tasks.
        """
        import sys
        import types

        module_objects: "Dict[str, Any]" = {}
        original_modules: "Dict[str, Any]" = {}

        # Register all modules as empty shells first
        for f in self._files:
            if not f.startswith("test_"):
                name = f.replace(".py", "")
                mod = types.ModuleType(name)
                mod.__file__ = f
                module_objects[name] = mod
                original_modules[name] = sys.modules.get(name)
                sys.modules[name] = mod

        # Multi-pass exec — retry failed files until nothing new resolves.
        # Each pass may unblock files that depend on what just succeeded.
        # Max passes = number of files (handles arbitrary depth chains).
        pending: "Dict[str, str]" = {
            f: src for f, src in self._files.items()
            if not f.startswith("test_")
        }

        max_passes = len(pending) + 1
        for _ in range(max_passes):
            if not pending:
                break
            still_failing: "Dict[str, str]" = {}
            for f, src in pending.items():
                name = f.replace(".py", "")
                try:
                    exec(compile(src, f, "exec"), module_objects[name].__dict__)
                except Exception:
                    still_failing[f] = src
            # If nothing improved this pass, stop — circular dep or real error
            if len(still_failing) == len(pending):
                break
            pending = still_failing

        return module_objects, original_modules

    def _restore_modules(self, original_modules: "Dict[str, Any]") -> None:
        """Restore sys.modules to state before _build_module_registry."""
        import sys
        for name, original in original_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original

    def _run_test_file(self, fname: str, content: str) -> Dict[str, Any]:
        """
        Run all test_ functions in a test file string.
        Returns a dict with passed/failed counts and per-test results.
        """
        module_objects, original_modules = self._build_module_registry()

        namespace: Dict[str, Any] = {}
        try:
            exec(compile(content, fname, "exec"), namespace)
        except Exception as e:
            self._restore_modules(original_modules)
            return {"error": str(e), "passed": 0, "failed": 0, "all_passed": False}
        finally:
            self._restore_modules(original_modules)

        test_funcs = {k: v for k, v in namespace.items()
                      if k.startswith("test_") and callable(v)}

        results = {}
        passed = 0
        failed = 0

        for name, func in test_funcs.items():
            try:
                func()
                results[name] = {"status": "PASS"}
                passed += 1
            except AssertionError as e:
                results[name] = {"status": "FAIL", "error": str(e)}
                failed += 1
            except Exception as e:
                results[name] = {"status": "ERROR", "error": str(e)}
                failed += 1

        return {
            "tests": results,
            "passed": passed,
            "failed": failed,
            "all_passed": failed == 0,
        }

    def _silent_test_run(self) -> bool:
        """
        Silently run all tests against current file state.
        Returns True if all tests pass, False otherwise.
        Does NOT update agent-visible state.
        """
        test_files = {k: v for k, v in self._files.items()
                      if k.startswith("test_")}
        if not test_files:
            return False

        for fname, content in test_files.items():
            result = self._run_test_file(fname, content)
            if not result.get("all_passed", False):
                return False
        return True

    def _safe_execute_function(
        self, source: str, func_name: str, input_args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Safely execute a function with given arguments and return the trace."""
        namespace: Dict[str, Any] = {}

        # Load all current files into namespace
        for f, src in self._files.items():
            try:
                exec(compile(src, f, "exec"), namespace)
            except Exception:
                pass

        func = namespace.get(func_name)
        if func is None or not callable(func):
            return {"error": f"'{func_name}' is not callable"}

        try:
            result = func(**input_args)
            return {"return_value": result, "exception": None}
        except Exception as e:
            return {"return_value": None, "exception": f"{type(e).__name__}: {e}"}