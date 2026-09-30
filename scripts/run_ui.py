"""Start the Streamlit chat page (implementation plan, phase 8).

``python scripts/run_ui.py`` runs ``src/ui/app.py``. The page talks to the API
over HTTP, so start that first:

    uvicorn src.api.main:app

Point the page somewhere else with ``CHAT_API_BASE_URL``, or with
``--api-base-url``, which sets the same variable for the Streamlit process.
Any other argument is passed through to ``streamlit run``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ui.client import API_BASE_URL_ENV, api_base_url  # noqa: E402

APP_PATH = PROJECT_ROOT / "src" / "ui" / "app.py"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--api-base-url",
        default=None,
        help=f"Base URL of the chat API (default: {API_BASE_URL_ENV} or the local uvicorn address)",
    )
    args, passthrough = parser.parse_known_args()

    environment = dict(os.environ)
    if args.api_base_url:
        environment[API_BASE_URL_ENV] = args.api_base_url

    print(f"Chat API: {args.api_base_url or api_base_url()}")
    command = [sys.executable, "-m", "streamlit", "run", str(APP_PATH), *passthrough]
    raise SystemExit(subprocess.call(command, env=environment))


if __name__ == "__main__":
    main()
