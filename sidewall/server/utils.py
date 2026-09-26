"""Shared helpers for the SIDEWALL FastAPI server."""

from __future__ import annotations

import os
import subprocess
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def short_git_hash(repo_root: str | Path) -> str:
    """Return the current repository revision as a short git hash.

    Args:
        repo_root: Path inside the git worktree.

    Returns:
        The short git hash for HEAD, or ``"unknown"`` when git metadata is unavailable.

    Raises:
        No exceptions are raised; failures return ``"unknown"``.
    """
    env_hash = os.getenv("SIDEWALL_MODEL_VERSION")
    if env_hash:
        return env_hash[:12]

    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--short=12", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except Exception:  # noqa: BLE001 - model serving should survive without git metadata
        return "unknown"
    value = result.stdout.strip()
    return value or "unknown"
