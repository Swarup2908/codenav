# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
CodeNav Environment — strictly following OpenEnv API spec.

Per OpenEnv docs (deepwiki.com/meta-pytorch/OpenEnv):
- Environment subclasses abstract base from openenv.core.env_server.interfaces
- Observation base class has reward field (types.py:86-88)
- State requires episode_id and step_count (types.py:190-197)
- Rubric.forward(action, observation) -> float, strictly between 0 and 1
- super().__init__(rubric=...) called first in __init__
- SUPPORTS_CONCURRENT_SESSIONS = True for multi-session WebSocket support
"""

import ast
import difflib
from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

# --- Rubric (RFC 004) ---
try:
    from openenv.core.rubrics.base import Rubric
except ImportError:
    class Rubric:
        def forward(self, action, observation) -> float:
            return 0.5
        def __call__(self, action, observation) -> float:
            return self.forward(action, observation)


class CodeNavRubric(Rubric):
    """
    Per RFC 004: Rubric.forward(action, observation) -> float in (0, 1).
    Attached to Environment via super().__init__(rubric=self.rubric).
    Updated on every submit action with the computed final score.
    """

    def __init__(self):
        self._score = 0.5

    def forward(self, action, observation) -> float:
        return round(max(0.12, min(0.88, float(self._score))), 3)

    def set_score(self, score: float) -> None:
        self._score = max(0.12, min(0.88, float(score)))


# --- Models ---
try:
    from ..models import CodeNavAction, CodeNavObservation, CodeNavState
except ImportError:
    from models import CodeNavAction, CodeNavObservation, CodeNavState

try:
    from ..scenarios import get_scenario, get_pool_size
except ImportError:
    from scenarios import get_scenario, get_pool_size


def _clamp(v: float) -> float:
    """Clamp to strictly (0, 1) with safe buffer."""
    return round(max(0.12, min(0.88, float(v))), 3)


# --- Reward Computer ---

class RewardComputer:
    BASE = 0.15

    RELEVANT_FILE_READ      =  0.03
    IRRELEVANT_FILE_READ    = -0.02
    BRUTE_FORCE_PENALTY     = -0.05
    CORRECT_DIAG_EARLY      =  0.15
    CORRECT_DIAG_LATE       =  0.08
    PARTIAL_DIAG            =  0.05
    NO_DIAG                 =  0.0
    WRONG_DIAG              = -0.05
    EDIT_WITHOUT_READ       = -0.05
    CORRECT_MINIMAL_FIX     =  0.25
    CORRECT_NONMINIMAL_FIX  =  0.18
    PARTIAL_FIX             =  0.08
    FIX_BROKE_TESTS         = -0.08
    RAN_TESTS               =  0.04
    TESTS_PASSED            =  0.08
    SUBMIT_WITHOUT_VERIFY   = -0.03
    EFFICIENCY_HIGH         =  0.08
    EFFICIENCY_MED          =  0.04
    HIT_MAX_STEPS           = -0.05

    def compute(self, state, task):
        b = {}

        # Reading
        r = self.BASE
        for _ in state.relevant_files_read:   r += self.RELEVANT_FILE_READ
        for _ in state.irrelevant_files_read: r += self.IRRELEVANT_FILE_READ
        if state.read_more_than_threshold:    r += self.BRUTE_FORCE_PENALTY
        b["reading"] = round(r, 3)

        # Diagnosis
        if not state.diagnosis_submitted:
            d = self.BASE + self.NO_DIAG
        elif state.diagnosis_correct:
            d = self.BASE + (self.CORRECT_DIAG_EARLY if state.diagnosis_before_edit else self.CORRECT_DIAG_LATE)
        elif state.diagnosis_partial:
            d = self.BASE + self.PARTIAL_DIAG
        else:
            d = self.BASE + self.WRONG_DIAG
        b["diagnosis"] = round(d, 3)

        # Edit
        e = self.BASE
        if state.edit_correct is not None:
            if state.edit_correct and state.edit_minimal:
                e = self.BASE + self.CORRECT_MINIMAL_FIX
            elif state.edit_correct:
                e = self.BASE + self.CORRECT_NONMINIMAL_FIX
            else:
                e = self.BASE + self.PARTIAL_FIX
            if state.tests_passed is False and state.tests_run:
                e += self.FIX_BROKE_TESTS
        b["edit"] = round(e, 3)

        # Verification
        v = self.BASE
        if state.tests_run:
            v += self.RAN_TESTS
            if state.tests_passed:
                v += self.TESTS_PASSED
        if not state.verified_before_submit:
            v += self.SUBMIT_WITHOUT_VERIFY
        b["verification"] = round(v, 3)

        # Efficiency
        eff = self.BASE
        ratio = state.step_count / state.max_steps if state.max_steps else 1.0
        if state.done and state.edit_correct:
            if ratio <= 0.60:   eff = self.BASE + self.EFFICIENCY_HIGH
            elif ratio <= 0.80: eff = self.BASE + self.EFFICIENCY_MED
        if state.step_count >= state.max_steps:
            eff += self.HIT_MAX_STEPS
        b["efficiency"] = round(eff, 3)

        raw = round(sum(b.values()), 3)
        if state.bug_2_exists:
            raw = round(raw * 0.5, 3)
        b["total"] = raw
        return b

    def compute_bug2(self, state):
        b = {}

        if not state.diagnosis_submitted:
            b["diagnosis"] = self.BASE + self.NO_DIAG
        elif state.diagnosis_correct:
            b["diagnosis"] = self.BASE + (self.CORRECT_DIAG_EARLY if state.diagnosis_before_edit else self.CORRECT_DIAG_LATE)
        elif state.diagnosis_partial:
            b["diagnosis"] = self.BASE + self.PARTIAL_DIAG
        else:
            b["diagnosis"] = self.BASE + self.WRONG_DIAG

        if state.edit_correct:
            b["edit"] = self.BASE + (self.CORRECT_MINIMAL_FIX if state.edit_minimal else self.CORRECT_NONMINIMAL_FIX)
        else:
            b["edit"] = self.BASE + self.PARTIAL_FIX

        v = self.BASE
        if state.tests_run:
            v += self.RAN_TESTS
            if state.tests_passed: v += self.TESTS_PASSED
        if not state.verified_before_submit:
            v += self.SUBMIT_WITHOUT_VERIFY
        b["verification"] = round(v, 3)

        raw = round(sum(b.values()), 3)
        b["total"] = round(raw * 0.5, 3)
        return b


# --- Environment ---

class CodeNavEnvironment(Environment):
    """
    CodeNav RL environment following OpenEnv API spec.
    Per docs: SUPPORTS_CONCURRENT_SESSIONS=True for WebSocket multi-session.
    Per RFC 004: rubric passed to super().__init__(rubric=...).
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, task_id: str = "easy", scenario_index: int = None):
        if task_id not in ("easy", "medium", "hard"):
            raise ValueError(f"task_id must be 'easy', 'medium', or 'hard'")

        # Per OpenEnv docs: call super().__init__() first with rubric
        self._rubric = CodeNavRubric()
        super().__init__(rubric=self._rubric)

        self._task_id = task_id
        self._scenario_index = scenario_index
        self._task = get_scenario(task_id, scenario_index)
        self._active_task = self._task
        self._rc = RewardComputer()
        self._state: Optional[CodeNavState] = None
        self._files: Dict[str, str] = {}
        self._wrong_diag_count: int = 0

    def reset(self, seed=None, episode_id=None, **kwargs) -> CodeNavObservation:
        """
        Per OpenEnv API: reset() initializes episode, resets step_count to 0,
        sets episode_id, returns initial Observation with reward field.
        """
        task_id = kwargs.get("task_id", self._task_id)
        if task_id in ("easy", "medium", "hard"):
            self._task_id = task_id

        self._task = get_scenario(self._task_id, self._scenario_index)
        self._active_task = self._task
        self._files = dict(self._task["files"])
        self._wrong_diag_count = 0

        # Per docs: State requires episode_id and step_count
        self._state = CodeNavState(
            episode_id=str(uuid4()),
            task_id=self._task_id,
            step_count=0,
            max_steps=self._task["max_steps"],
            done=False,
            bug_2_exists=self._task.get("bug_2") is not None,
        )

        # Per docs: Observation.reward field — use safe non-zero value
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
            reward=0.15,
        )

    def step(self, action: CodeNavAction) -> CodeNavObservation:
        """
        Per OpenEnv API: step() increments step_count, returns Observation
        with reward and done fields.
        """
        if self._state is None:
            raise RuntimeError("Call reset() before step()")
        if self._state.done:
            return self._obs(success=False, message="Episode is already done.")

        self._state.step_count += 1

        handlers = {
            "read_file":        self._read_file,
            "read_function":    self._read_function,
            "search_codebase":  self._search_codebase,
            "submit_diagnosis": self._submit_diagnosis,
            "identify_location":self._identify_location,
            "edit_code":        self._edit_code,
            "add_code":         self._add_code,
            "delete_code":      self._delete_code,
            "run_tests":        self._run_tests,
            "trace_execution":  self._trace_execution,
            "submit":           self._submit,
        }

        handler = handlers.get(action.action_type)
        if handler is None:
            return self._obs(success=False, message=f"Unknown action_type: {action.action_type}")

        obs = handler(action)

        if self._state.step_count >= self._state.max_steps and not self._state.done:
            self._state.done = True
            obs.message += "\n[MAX STEPS REACHED]"
            obs.done = True

        return obs

    @property
    def state(self) -> CodeNavState:
        """Per OpenEnv API: state property returns State with episode_id and step_count."""
        if self._state is None:
            self.reset()
        return self._state

    # --- Action handlers ---

    def _read_file(self, action):
        fname = action.filename
        if not fname or fname not in self._files:
            return self._obs(success=False, message=f"File '{fname}' not found. Available: {list(self._files.keys())}")
        content = self._files[fname]
        line_count = len(content.splitlines())
        if fname not in self._state.files_read:
            self._state.files_read.append(fname)
            if fname in self._task["relevant_files"]:
                self._state.relevant_files_read.append(fname)
            else:
                self._state.irrelevant_files_read.append(fname)
        if len(self._state.files_read) / len(self._files) > 0.70:
            self._state.read_more_than_threshold = True
        return self._obs(success=True, message=f"Read '{fname}' ({line_count} lines)",
                         file_content=content, file_line_count=line_count)

    def _read_function(self, action):
        fname, func = action.filename, action.function_name
        if not fname or fname not in self._files:
            return self._obs(success=False, message=f"File '{fname}' not found.")
        if not func:
            return self._obs(success=False, message="function_name is required.")
        extracted = self._extract_function(self._files[fname], func)
        if extracted is None:
            return self._obs(success=False, message=f"Function '{func}' not found in '{fname}'.")
        if fname not in self._state.files_read:
            self._state.files_read.append(fname)
            if fname in self._task["relevant_files"]:
                self._state.relevant_files_read.append(fname)
            else:
                self._state.irrelevant_files_read.append(fname)
        return self._obs(success=True, message=f"Read function '{func}' from '{fname}'",
                         file_content=extracted, file_line_count=len(extracted.splitlines()))

    def _search_codebase(self, action):
        query = action.query
        if not query:
            return self._obs(success=False, message="query is required.")
        results = []
        for fname, content in self._files.items():
            for i, line in enumerate(content.splitlines(), 1):
                if query.lower() in line.lower():
                    results.append({"file": fname, "line": i, "content": line.strip()})
        return self._obs(success=True, message=f"Search '{query}' returned {len(results)} results.",
                         search_results=results)

    def _submit_diagnosis(self, action):
        diagnosis = action.diagnosis
        if not diagnosis:
            return self._obs(success=False, message="diagnosis is required.")
        keywords = self._active_task["correct_diagnosis_keywords"]
        d_lower = diagnosis.lower()
        is_correct = any(kw.lower() in d_lower for kw in keywords)
        is_partial = not is_correct and any(kw.split()[0].lower() in d_lower for kw in keywords)
        if is_correct:
            self._state.diagnosis_correct = True
            self._state.diagnosis_partial = False
            if self._state.edits_made == 0:
                self._state.diagnosis_before_edit = True
            feedback = "Diagnosis matches root cause. Proceed with fix."
        elif is_partial:
            self._state.diagnosis_partial = True
            feedback = "On the right track but incomplete."
        else:
            self._state.diagnosis_correct = False
            self._state.diagnosis_partial = False
            feedback = "Diagnosis does not match root cause."
        self._state.diagnosis_submitted = True
        return self._obs(success=True, message="Diagnosis submitted.", diagnosis_feedback=feedback)

    def _identify_location(self, action):
        fname = action.filename
        bug_loc = self._task["bug_location"]
        if fname == bug_loc["file"]:
            ls, le = action.line_start, action.line_end
            if ls is not None and le is not None:
                overlap = not (le < bug_loc["line_start"] or ls > bug_loc["line_end"])
                feedback = "Location correct." if overlap else "Wrong line range."
            else:
                feedback = "Provide line_start and line_end."
        else:
            feedback = f"Wrong file. Bug is not in '{fname}'."
        return self._obs(success=True, message="Location submitted.", diagnosis_feedback=feedback)

    def _edit_code(self, action):
        fname, old_code, new_code = action.filename, action.old_code, action.new_code
        if not fname or fname not in self._files:
            return self._obs(success=False, message=f"File '{fname}' not found.")
        if old_code is None or new_code is None:
            return self._obs(success=False, message="old_code and new_code are required.")
        if fname not in self._state.files_read:
            self._state.cumulative_reward += RewardComputer.EDIT_WITHOUT_READ
        content = self._files[fname]
        if old_code not in content:
            return self._obs(success=False, message="old_code not found in file.")
        new_content = content.replace(old_code, new_code, 1)
        diff = "\n".join(difflib.unified_diff(content.splitlines(), new_content.splitlines(),
                                               fromfile=f"a/{fname}", tofile=f"b/{fname}", lineterm=""))
        if not self._check_syntax(new_content):
            return self._obs(success=False, message="Edit rejected — invalid Python.", diff=diff, syntax_valid=False)
        self._files[fname] = new_content
        self._state.edits_made += 1
        self._state.edit_correct = self._silent_test_run()
        if self._state.edit_correct:
            lines_changed = max(len(old_code.strip().splitlines()), len(new_code.strip().splitlines()))
            self._state.edit_minimal = lines_changed <= 5
        scope_warning = None
        if fname in self._task.get("irrelevant_files", []):
            scope_warning = f"Warning: '{fname}' is outside expected scope."
        return self._obs(success=True, message=f"Edit applied to '{fname}'.",
                         diff=diff, syntax_valid=True, scope_warning=scope_warning)

    def _add_code(self, action):
        fname, new_code = action.filename, action.new_code
        location = action.location or "bottom"
        if not fname or fname not in self._files:
            return self._obs(success=False, message=f"File '{fname}' not found.")
        if not new_code:
            return self._obs(success=False, message="new_code is required.")
        content = self._files[fname]
        if location == "top":
            new_content = new_code + "\n" + content
        elif location.startswith("after:"):
            new_content = self._insert_after_function(content, location.split("after:", 1)[1], new_code)
            if new_content is None:
                return self._obs(success=False, message="Function not found for after: insertion.")
        else:
            new_content = content + "\n" + new_code
        if not self._check_syntax(new_content):
            return self._obs(success=False, message="Add rejected — invalid Python.", syntax_valid=False)
        diff = "\n".join(difflib.unified_diff(content.splitlines(), new_content.splitlines(),
                                               fromfile=f"a/{fname}", tofile=f"b/{fname}", lineterm=""))
        self._files[fname] = new_content
        self._state.edits_made += 1
        return self._obs(success=True, message=f"Code added to '{fname}'.", diff=diff, syntax_valid=True)

    def _delete_code(self, action):
        fname, target = action.filename, action.target_code
        if not fname or fname not in self._files:
            return self._obs(success=False, message=f"File '{fname}' not found.")
        if not target:
            return self._obs(success=False, message="target_code is required.")
        content = self._files[fname]
        if target not in content:
            return self._obs(success=False, message="target_code not found in file.")
        new_content = content.replace(target, "", 1)
        if not self._check_syntax(new_content):
            return self._obs(success=False, message="Delete rejected — invalid Python.", syntax_valid=False)
        diff = "\n".join(difflib.unified_diff(content.splitlines(), new_content.splitlines(),
                                               fromfile=f"a/{fname}", tofile=f"b/{fname}", lineterm=""))
        self._files[fname] = new_content
        self._state.edits_made += 1
        return self._obs(success=True, message=f"Code deleted from '{fname}'.", diff=diff, syntax_valid=True)

    def _run_tests(self, action):
        self._state.tests_run = True
        self._state.verified_before_submit = True
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
        payload = {"files": all_results, "overall_passed": overall_passed,
                   "total_passed": total_pass, "total_failed": total_fail}
        bug_2_def = self._task.get("bug_2")
        if (overall_passed and self._state.current_bug == 1 and
                self._state.bug_2_exists and bug_2_def is not None and
                not self._state.bug_1_fixed):
            self._state.bug_1_fixed = True
            self._state.current_bug = 2
            self._state.diagnosis_submitted = False
            self._wrong_diag_count = 0
            for fn, extra in bug_2_def.get("test_additions", {}).items():
                if fn in self._files:
                    self._files[fn] += extra
            self._active_task = {
                **self._task,
                "relevant_files": bug_2_def["relevant_files"],
                "correct_diagnosis_keywords": bug_2_def["correct_diagnosis_keywords"],
                "correct_fix": bug_2_def["correct_fix"],
            }
            self._state.diagnosis_correct = None
            self._state.diagnosis_partial = None
            self._state.diagnosis_before_edit = False
            self._state.edit_correct = None
            self._state.edit_minimal = None
            self._state.tests_run = False
            self._state.tests_passed = None
            self._state.verified_before_submit = False
            self._state.edits_made = 0
            bug2_msg = (f"\n\n{'='*50}\nBug 1 fixed! New issue:\n"
                        f"{bug_2_def['description']}\nContinue investigating.")
            return self._obs(success=True,
                             message=f"Tests: {total_pass} passed, {total_fail} failed.{bug2_msg}",
                             test_results=payload)
        if overall_passed and self._state.current_bug == 2 and not self._state.bug_2_fixed:
            self._state.bug_2_fixed = True
        return self._obs(success=True, message=f"Tests: {total_pass} passed, {total_fail} failed.",
                         test_results=payload)

    def _trace_execution(self, action):
        func_name = action.function_name
        if not func_name:
            return self._obs(success=False, message="function_name is required.")
        self._state.verified_before_submit = True
        target_file = None
        for fname, content in self._files.items():
            if f"def {func_name}" in content:
                target_file = (fname, content)
                break
        if target_file is None:
            return self._obs(success=False, message=f"Function '{func_name}' not found.")
        fname, content = target_file
        result = self._safe_execute(content, func_name, action.input_args or {})
        return self._obs(success=True, message=f"Traced '{func_name}' from '{fname}'.",
                         execution_trace=result)

    def _submit(self, action):
        if self._state is None:
            raise RuntimeError("Call reset() first.")
        self._state.done = True

        # Compute rewards per RFC 004 rubric pattern
        breakdown = self._rc.compute(self._state, self._task)

        if self._state.bug_2_exists:
            b2 = self._rc.compute_bug2(self._state)
            breakdown["bug_2_total"] = b2["total"]
            breakdown["total"] = round(breakdown["total"] + b2["total"], 3)
            breakdown.update({f"bug2_{k}": v for k, v in b2.items() if k != "total"})

        # Clamp all breakdown values strictly between 0 and 1
        for k in list(breakdown.keys()):
            if breakdown[k] is not None:
                breakdown[k] = _clamp(float(breakdown[k]))

        total = breakdown["total"]
        self._state.cumulative_reward += total

        # Per RFC 004: update rubric score so forward() returns valid value
        self._rubric.set_score(total)

        return self._obs(
            success=True,
            message=f"Episode complete. Score: {total:.3f}",
            final_score=total,
            score_breakdown=breakdown,
            done=True,
            reward=total,
        )

    # --- Observation factory ---

    def _obs(self, success: bool, message: str, **kwargs) -> CodeNavObservation:
        if self._state is None:
            step, max_s, files, read, edits, diag, tests = 0, 0, [], [], 0, False, False
        else:
            step  = self._state.step_count
            max_s = self._state.max_steps
            files = list(self._files.keys())
            read  = list(self._state.files_read)
            edits = self._state.edits_made
            diag  = self._state.diagnosis_submitted
            tests = self._state.tests_run

        # Clamp final_score and reward if present
        if "final_score" in kwargs and kwargs["final_score"] is not None:
            kwargs["final_score"] = _clamp(kwargs["final_score"])
        if "reward" in kwargs and kwargs["reward"] is not None:
            kwargs["reward"] = _clamp(kwargs["reward"])

        # Per OpenEnv docs: Observation.reward is the reward signal
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
            reward=kwargs.pop("reward", 0.15),
            **kwargs,
        )

    # --- Helpers ---

    def _extract_function(self, source: str, func_name: str) -> Optional[str]:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return None
        lines = source.splitlines()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                return "\n".join(lines[node.lineno - 1:node.end_lineno])
        return None

    def _insert_after_function(self, source: str, func_name: str, new_code: str) -> Optional[str]:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return None
        lines = source.splitlines()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                lines.insert(node.end_lineno, new_code)
                return "\n".join(lines)
        return None

    def _check_syntax(self, source: str) -> bool:
        try:
            ast.parse(source)
            return True
        except SyntaxError:
            return False

    def _build_module_registry(self):
        import sys, types
        module_objects = {}
        original_modules = {}
        for f in self._files:
            if not f.startswith("test_"):
                name = f.replace(".py", "")
                mod = types.ModuleType(name)
                mod.__file__ = f
                module_objects[name] = mod
                original_modules[name] = sys.modules.get(name)
                sys.modules[name] = mod
        pending = {f: src for f, src in self._files.items() if not f.startswith("test_")}
        for _ in range(len(pending) + 1):
            if not pending:
                break
            still_failing = {}
            for f, src in pending.items():
                name = f.replace(".py", "")
                try:
                    exec(compile(src, f, "exec"), module_objects[name].__dict__)
                except Exception:
                    still_failing[f] = src
            if len(still_failing) == len(pending):
                break
            pending = still_failing
        return module_objects, original_modules

    def _restore_modules(self, original_modules) -> None:
        import sys
        for name, original in original_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original

    def _run_test_file(self, fname: str, content: str) -> Dict[str, Any]:
        module_objects, original_modules = self._build_module_registry()
        namespace: Dict[str, Any] = {}
        try:
            exec(compile(content, fname, "exec"), namespace)
        except Exception as e:
            self._restore_modules(original_modules)
            return {"error": str(e), "passed": 0, "failed": 0, "all_passed": False}
        finally:
            self._restore_modules(original_modules)
        test_funcs = {k: v for k, v in namespace.items() if k.startswith("test_") and callable(v)}
        results = {}
        passed = failed = 0
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
        return {"tests": results, "passed": passed, "failed": failed, "all_passed": failed == 0}

    def _silent_test_run(self) -> bool:
        test_files = {k: v for k, v in self._files.items() if k.startswith("test_")}
        if not test_files:
            return False
        for fname, content in test_files.items():
            if not self._run_test_file(fname, content).get("all_passed", False):
                return False
        return True

    def _safe_execute(self, source: str, func_name: str, input_args: Dict[str, Any]) -> Dict[str, Any]:
        namespace: Dict[str, Any] = {}
        for f, src in self._files.items():
            try:
                exec(compile(src, f, "exec"), namespace)
            except Exception:
                pass
        func = namespace.get(func_name)
        if func is None or not callable(func):
            return {"error": f"'{func_name}' is not callable"}
        try:
            return {"return_value": func(**input_args), "exception": None}
        except Exception as e:
            return {"return_value": None, "exception": f"{type(e).__name__}: {e}"}