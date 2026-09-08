"""Configuration and shared paths for MultiDeviceLLM.

Everything is overridable via environment variables so the same CLI works on a
laptop that hosts the cluster and on a worker node.
"""

from __future__ import annotations

import os
from pathlib import Path

# Base URL of the exo API + dashboard. exo serves both from this address.
EXO_API_URL: str = os.environ.get("MDLLM_EXO_URL", "http://localhost:52415")

# Where we clone/build the exo engine. Kept out of git via .gitignore.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
EXO_DIR: Path = Path(
    os.environ.get("MDLLM_EXO_DIR", str(PROJECT_ROOT / "vendor" / "exo"))
).expanduser()

# Repo to clone the engine from.
EXO_REPO: str = os.environ.get("MDLLM_EXO_REPO", "https://github.com/exo-explore/exo")

# Default request timeout (seconds) for control-plane calls. Chat streaming
# uses its own, effectively unbounded, timeout.
HTTP_TIMEOUT: float = float(os.environ.get("MDLLM_HTTP_TIMEOUT", "30"))
