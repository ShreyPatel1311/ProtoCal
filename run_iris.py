#!/usr/bin/env python
"""Start IRIS with the Claude Code CLI as its LLM backend.

Loads .env, patches litellm so every completion is answered by a headless
Claude Code call, then runs the stock IRIS Flask/SocketIO app unchanged.

    python run_iris.py            # serve on 127.0.0.1:5000
    IRIS_PORT=5050 python run_iris.py
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def load_dotenv(path: Path) -> None:
    """Minimal .env loader (no extra dependency)."""
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if value and key not in os.environ:
            os.environ[key] = value


def main() -> int:
    load_dotenv(ROOT / ".env")

    missing = [k for k in ("CLAUDE_CODE_OAUTH_TOKEN",) if not os.environ.get(k)]
    if missing:
        print(f"ERROR: missing in .env: {', '.join(missing)}", file=sys.stderr)
        print("Create the token with:  claude setup-token", file=sys.stderr)
        return 2

    if not os.environ.get("SEMANTIC_SCHOLAR_API_KEY"):
        print("WARNING: SEMANTIC_SCHOLAR_API_KEY not set - retrieval will fail.")

    # Patch litellm before any agent issues a completion.
    from src.models.claude_code_llm import install
    install()

    import app as iris_app

    host = os.environ.get("IRIS_HOST", "127.0.0.1")
    port = int(os.environ.get("IRIS_PORT", "5000"))
    print(f"[run_iris] IRIS listening on http://{host}:{port}")
    iris_app.socketio.run(
        iris_app.app, host=host, port=port, debug=False, allow_unsafe_werkzeug=True
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
