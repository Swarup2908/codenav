# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the CodeNav Environment.

CodeNav is an RL environment where an agent acts as a software developer
dropped into an unfamiliar Python codebase. The agent must read, diagnose,
act, verify, and submit — rewarded for the quality of the entire process,
not just the final output.
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import Field
from openenv.core.env_server.types import Action, Observation, State


# ---------------------------------------------------------------------------
# Action Types
# ---------------------------------------------------------------------------

class CodeNavAction(Action):
    """
    A single action the agent takes inside a CodeNav episode.

    The action_type field determines which operation is performed.
    Each action_type uses a specific subset of the other fields.

    Action types and their required fields:

    READ ACTIONS (exploring the codebase):
        read_file         — filename
        read_function     — filename, function_name
        search_codebase   — query

    DIAGNOSE ACTIONS (forming understanding):
        submit_diagnosis  — diagnosis
        identify_location — filename, line_start, line_end

    ACT ACTIONS (making changes):
        edit_code         — filename, old_code, new_code
        add_code          — filename, location, new_code
        delete_code       — filename, target_code

    VERIFY ACTIONS (checking work):
        run_tests         — (no extra fields needed)
        trace_execution   — function_name, input_args

    TERMINAL ACTION:
        submit            — (no extra fields needed)
    """

    action_type: Literal[
        # Read
        "read_file",
        "read_function",
        "search_codebase",
        # Diagnose
        "submit_diagnosis",
        "identify_location",
        # Act
        "edit_code",
        "add_code",
        "delete_code",
        # Verify
        "run_tests",
        "trace_execution",
        # Terminal
        "submit",
    ] = Field(..., description="The type of action to perform")

    # --- Read fields ---
    filename: Optional[str] = Field(
        default=None,
        description="Target filename for read/edit/diagnose actions (e.g. 'utils.py')"
    )
    function_name: Optional[str] = Field(
        default=None,
        description="Function name for read_function or trace_execution actions"
    )
    query: Optional[str] = Field(
        default=None,
        description="Search query string for search_codebase action"
    )

    # --- Diagnose fields ---
    diagnosis: Optional[str] = Field(
        default=None,
        description="Agent's natural language description of what is wrong and why"
    )
    line_start: Optional[int] = Field(
        default=None,
        description="Start line number for identify_location action"
    )
    line_end: Optional[int] = Field(
        default=None,
        description="End line number for identify_location action"
    )

    # --- Act fields ---
    old_code: Optional[str] = Field(
        default=None,
        description="Exact code block to be replaced in edit_code action"
    )
    new_code: Optional[str] = Field(
        default=None,
        description="New code to insert or replace with in edit_code or add_code actions"
    )
    location: Optional[str] = Field(
        default=None,
        description="Where to insert new code: 'top', 'bottom', or 'after:<function_name>'"
    )
    target_code: Optional[str] = Field(
        default=None,
        description="Exact code block to delete in delete_code action"
    )

    # --- Verify fields ---
    input_args: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Input arguments for trace_execution action as a dict"
    )


# ---------------------------------------------------------------------------
# Observation Types
# ---------------------------------------------------------------------------

class CodeNavObservation(Observation):
    """
    What the agent receives back after every action.

    Every observation includes the persistent state panel so the agent
    always knows where it is in the workflow. The content fields are
    populated based on what action was just taken.
    """

    # --- Always present ---
    success: bool = Field(
        ...,
        description="Whether the action was executed successfully"
    )
    message: str = Field(
        ...,
        description="Human-readable description of what happened"
    )

    # --- Persistent state panel (always visible) ---
    current_step: int = Field(
        ...,
        description="Current step number in the episode"
    )
    max_steps: int = Field(
        ...,
        description="Maximum steps allowed in this episode"
    )
    available_files: List[str] = Field(
        ...,
        description="All files available in this episode (agent always sees this list)"
    )
    files_read: List[str] = Field(
        ...,
        description="Files the agent has read so far this episode"
    )
    edits_made: int = Field(
        ...,
        description="Number of edits made so far this episode"
    )
    diagnosis_submitted: bool = Field(
        ...,
        description="Whether the agent has submitted a diagnosis yet"
    )
    tests_run: bool = Field(
        ...,
        description="Whether the agent has run tests yet this episode"
    )

    # --- Content fields (populated based on action type) ---

    # Read actions
    file_content: Optional[str] = Field(
        default=None,
        description="Content of a file returned by read_file or read_function"
    )
    file_line_count: Optional[int] = Field(
        default=None,
        description="Total line count of the file that was read"
    )
    search_results: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="List of search matches from search_codebase action"
    )

    # Diagnose actions
    diagnosis_feedback: Optional[str] = Field(
        default=None,
        description="Weak signal feedback on diagnosis — not the answer, just a hint"
    )

    # Act actions
    diff: Optional[str] = Field(
        default=None,
        description="Diff showing exactly what changed after an edit action"
    )
    syntax_valid: Optional[bool] = Field(
        default=None,
        description="Whether the edited file is still valid Python after the edit"
    )
    scope_warning: Optional[str] = Field(
        default=None,
        description="Warning if the edit touched code outside the intended scope"
    )

    # Verify actions
    test_results: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Test results: pass/fail per test, output, errors"
    )
    execution_trace: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Return value and any exceptions from trace_execution"
    )

    # Terminal action
    final_score: Optional[float] = Field(
        default=None,
        description="Final score from the grader after submit (strictly between 0 to 1)"
    )
    score_breakdown: Optional[Dict[str, float]] = Field(
        default=None,
        description="Per-dimension score breakdown after submit"
    )


# ---------------------------------------------------------------------------
# State Type
# ---------------------------------------------------------------------------

class CodeNavState(State):
    """
    Full internal state of a CodeNav episode.

    This is what state() returns — episode metadata and tracking.
    Not visible to the agent during the episode, used for logging
    and evaluation.
    """

    episode_id: str = Field(
        ...,
        description="Unique identifier for this episode"
    )
    task_id: str = Field(
        ...,
        description="Which task is being run: 'easy', 'medium', or 'hard'"
    )
    step_count: int = Field(
        ...,
        description="Current step count"
    )
    max_steps: int = Field(
        ...,
        description="Maximum steps for this episode"
    )
    done: bool = Field(
        ...,
        description="Whether the episode has ended"
    )

    # Tracking for reward computation
    files_read: List[str] = Field(
        default_factory=list,
        description="All files read this episode"
    )
    relevant_files_read: List[str] = Field(
        default_factory=list,
        description="Subset of files_read that are relevant to the bug"
    )
    irrelevant_files_read: List[str] = Field(
        default_factory=list,
        description="Subset of files_read that are irrelevant to the bug"
    )
    diagnosis_submitted: bool = Field(
        default=False,
        description="Whether a diagnosis was submitted"
    )
    diagnosis_correct: Optional[bool] = Field(
        default=None,
        description="Whether the submitted diagnosis was correct"
    )
    diagnosis_partial: Optional[bool] = Field(
        default=None,
        description="Whether the submitted diagnosis was partially correct"
    )
    diagnosis_before_edit: bool = Field(
        default=False,
        description="Whether the diagnosis was submitted before any edits"
    )
    edits_made: int = Field(
        default=0,
        description="Number of edit actions taken"
    )
    edit_correct: Optional[bool] = Field(
        default=None,
        description="Whether the final edit correctly fixes the bug"
    )
    edit_minimal: Optional[bool] = Field(
        default=None,
        description="Whether the edit was minimal (touched only necessary code)"
    )
    tests_run: bool = Field(
        default=False,
        description="Whether tests were run"
    )
    tests_passed: Optional[bool] = Field(
        default=None,
        description="Whether tests passed after the fix"
    )
    verified_before_submit: bool = Field(
        default=False,
        description="Whether the agent ran verification before submitting"
    )
    read_more_than_threshold: bool = Field(
        default=False,
        description="Whether agent read more than 70% of available files (brute force penalty)"
    )

    # Multi-bug tracking
    current_bug: int = Field(
        default=1,
        description="Which bug the agent is currently working on (1 or 2)"
    )
    bug_1_fixed: bool = Field(
        default=False,
        description="Whether bug 1 has been verified fixed via passing tests"
    )
    bug_2_exists: bool = Field(
        default=False,
        description="Whether this task has a second bug to fix"
    )
    bug_2_fixed: bool = Field(
        default=False,
        description="Whether bug 2 has been verified fixed via passing tests"
    )

    # Cumulative reward tracking
    cumulative_reward: float = Field(
        default=0.0,
        description="Running total of reward accumulated this episode"
    )