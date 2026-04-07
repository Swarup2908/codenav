# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the CodeNav Environment.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    uvicorn server.app:app --host 0.0.0.0 --port 8000
    python -m server.app
"""

import os

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from models import CodeNavAction, CodeNavObservation
    from server.codenav_environment import CodeNavEnvironment
except ModuleNotFoundError:
    from ..models import CodeNavAction, CodeNavObservation
    from .codenav_environment import CodeNavEnvironment


# Task ID is configurable via environment variable.
# Defaults to "easy". Set CODENAV_TASK=medium or CODENAV_TASK=hard
# in Docker or HF Spaces environment variables.
TASK_ID = os.environ.get("CODENAV_TASK", "easy")

if TASK_ID not in ("easy", "medium", "hard"):
    raise ValueError(
        f"CODENAV_TASK must be 'easy', 'medium', or 'hard'. Got: '{TASK_ID}'"
    )


def env_factory():
    """Factory function — creates a fresh CodeNavEnvironment per session."""
    return CodeNavEnvironment(task_id=TASK_ID)


# Create the FastAPI app
app = create_app(
    env_factory,
    CodeNavAction,
    CodeNavObservation,
    env_name="codenav",
    max_concurrent_envs=10,
)


def main():
    """Entry point for direct execution — callable with no arguments."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()