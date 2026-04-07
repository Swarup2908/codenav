# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CodeNav Environment Client."""

from typing import Any, Dict, List, Optional

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import CodeNavAction, CodeNavObservation, CodeNavState


class CodeNavEnv(
    EnvClient[CodeNavAction, CodeNavObservation, CodeNavState]
):
    """
    Client for the CodeNav Environment.

    Maintains a persistent WebSocket connection to the environment server.
    Each client instance has its own dedicated environment session.

    Example — basic usage:
        >>> with CodeNavEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.message)
        ...
        ...     # Read a file
        ...     result = client.step(CodeNavAction(
        ...         action_type="read_file",
        ...         filename="processor.py"
        ...     ))
        ...     print(result.observation.file_content)
        ...
        ...     # Submit a diagnosis
        ...     result = client.step(CodeNavAction(
        ...         action_type="submit_diagnosis",
        ...         diagnosis="The bug is an off-by-one error in the averages calculation"
        ...     ))
        ...
        ...     # Fix the bug
        ...     result = client.step(CodeNavAction(
        ...         action_type="edit_code",
        ...         filename="processor.py",
        ...         old_code="        averages[cid] = totals[cid] / counts[cid] + 1",
        ...         new_code="        averages[cid] = totals[cid] / counts[cid]"
        ...     ))
        ...
        ...     # Run tests
        ...     result = client.step(CodeNavAction(action_type="run_tests"))
        ...
        ...     # Submit
        ...     result = client.step(CodeNavAction(action_type="submit"))
        ...     print(result.observation.final_score)

    Example with Docker:
        >>> client = CodeNavEnv.from_docker_image("codenav-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(CodeNavAction(action_type="run_tests"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: CodeNavAction) -> Dict:
        """
        Convert CodeNavAction to JSON payload for the step message.

        Serializes only the fields that are set (non-None) to keep
        payloads clean and minimal.

        Args:
            action: CodeNavAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        payload = {"action_type": action.action_type}

        # Include only fields that are actually set
        optional_fields = [
            "filename",
            "function_name",
            "query",
            "diagnosis",
            "line_start",
            "line_end",
            "old_code",
            "new_code",
            "location",
            "target_code",
            "input_args",
        ]

        for field in optional_fields:
            value = getattr(action, field, None)
            if value is not None:
                payload[field] = value

        return payload

    def _parse_result(self, payload: Dict) -> StepResult[CodeNavObservation]:
        """
        Parse server response into StepResult[CodeNavObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with typed CodeNavObservation
        """
        obs_data = payload.get("observation", {})

        observation = CodeNavObservation(
            # Always present
            success=obs_data.get("success", False),
            message=obs_data.get("message", ""),

            # Persistent state panel
            current_step=obs_data.get("current_step", 0),
            max_steps=obs_data.get("max_steps", 0),
            available_files=obs_data.get("available_files", []),
            files_read=obs_data.get("files_read", []),
            edits_made=obs_data.get("edits_made", 0),
            diagnosis_submitted=obs_data.get("diagnosis_submitted", False),
            tests_run=obs_data.get("tests_run", False),

            # Episode control
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),

            # Read actions
            file_content=obs_data.get("file_content"),
            file_line_count=obs_data.get("file_line_count"),
            search_results=obs_data.get("search_results"),

            # Diagnose actions
            diagnosis_feedback=obs_data.get("diagnosis_feedback"),

            # Act actions
            diff=obs_data.get("diff"),
            syntax_valid=obs_data.get("syntax_valid"),
            scope_warning=obs_data.get("scope_warning"),

            # Verify actions
            test_results=obs_data.get("test_results"),
            execution_trace=obs_data.get("execution_trace"),

            # Terminal action
            final_score=obs_data.get("final_score"),
            score_breakdown=obs_data.get("score_breakdown"),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> CodeNavState:
        """
        Parse server response into CodeNavState.

        Args:
            payload: JSON response from state request

        Returns:
            CodeNavState with full episode tracking
        """
        return CodeNavState(
            episode_id=payload.get("episode_id", ""),
            task_id=payload.get("task_id", "easy"),
            step_count=payload.get("step_count", 0),
            max_steps=payload.get("max_steps", 0),
            done=payload.get("done", False),

            # Reading tracking
            files_read=payload.get("files_read", []),
            relevant_files_read=payload.get("relevant_files_read", []),
            irrelevant_files_read=payload.get("irrelevant_files_read", []),

            # Diagnosis tracking
            diagnosis_submitted=payload.get("diagnosis_submitted", False),
            diagnosis_correct=payload.get("diagnosis_correct"),
            diagnosis_partial=payload.get("diagnosis_partial"),
            diagnosis_before_edit=payload.get("diagnosis_before_edit", False),

            # Edit tracking
            edits_made=payload.get("edits_made", 0),
            edit_correct=payload.get("edit_correct"),
            edit_minimal=payload.get("edit_minimal"),

            # Verification tracking
            tests_run=payload.get("tests_run", False),
            tests_passed=payload.get("tests_passed"),
            verified_before_submit=payload.get("verified_before_submit", False),

            # Efficiency tracking
            read_more_than_threshold=payload.get("read_more_than_threshold", False),

            # Reward tracking
            cumulative_reward=payload.get("cumulative_reward", 0.0),
        )