"""Runtime environment helpers for the SIDEWALL API."""

from __future__ import annotations

import os
from pathlib import Path


def load_runtime_env(root: Path) -> Path | None:
    """Load simple KEY=VALUE pairs from the selected env file.

    Args:
        root: Repository root that may contain ``.env.local`` or ``.env.prod``.

    Returns:
        The path that was loaded, or ``None`` when no matching env file exists.

    Raises:
        No exceptions are raised for missing files or malformed comment/blank lines.
    """
    env_name = os.getenv("SIDEWALL_ENV", "local").lower()
    file_name = ".env.prod" if env_name in {"prod", "production"} else ".env.local"
    path = root / file_name
    if not path.exists():
        return None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
    return path


def runtime_host(default: str = "0.0.0.0") -> str:
    """Return the FastAPI bind host from environment settings."""
    return os.getenv("SIDEWALL_HOST", default)


def runtime_port(default: int = 8000) -> int:
    """Return the FastAPI bind port from environment settings."""
    try:
        return int(os.getenv("PORT", str(default)))
    except ValueError:
        return default
