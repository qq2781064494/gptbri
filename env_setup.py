"""Small .env loader for local example scripts."""

from __future__ import annotations

import os
from pathlib import Path


def load_project_env(env_file: str = ".env", override: bool = False) -> None:
    """Load KEY=VALUE pairs from the project root .env file."""
    path = Path(__file__).resolve().parent / env_file
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if override or key not in os.environ:
            os.environ[key] = value
