"""Load a local `.env` into the process environment.

A convenience for local development only. Two properties matter:

* **The real environment always wins.** Variables already set in the shell, in CI, or by a
  container runtime are never overwritten by the file, so a deployed process behaves the
  same whether or not a stray `.env` is present.
* **A missing file is normal.** No `.env` means no change, and the application runs.

This is imported from `app/__init__.py` rather than from `app.config`, because `config`
reads `os.environ` at module scope: the file has to be loaded before any `app` submodule is
imported, and the package `__init__` is the only hook guaranteed to run first. That ordering
is also why the project root is recomputed here instead of importing it from `config`.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"


def load_local_env(path: Path | None = None) -> bool:
    """Load `.env` if it exists. Returns True when something was loaded."""
    target = path or ENV_FILE
    if not target.exists():
        return False
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - python-dotenv is a declared dependency
        logger.debug("python-dotenv is not installed; skipping %s", target)
        return False
    # override=False keeps the shell authoritative over the file.
    return bool(load_dotenv(target, override=False))
