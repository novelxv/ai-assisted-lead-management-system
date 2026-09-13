"""Lead management service.

A local `.env` is loaded here, before any submodule reads `os.environ` at import time.
Shell variables take precedence; a missing file is a no-op. See app/env.py.
"""

from app.env import load_local_env

load_local_env()
