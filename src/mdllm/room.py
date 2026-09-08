"""Rooms — shareable cluster codes.

Inspired by SwarmLLM's "create a room, share the code, everyone joins" UX
(https://github.com/Nehanth/swarmllm). Under the hood a room is just an exo
libp2p namespace (``EXO_LIBP2P_NAMESPACE``): every device that starts with the
same room code forms one isolated swarm, and ignores everyone else on the
network.

The active room is persisted to ``.mdllm/room`` so `mdllm up` picks it up
automatically. Precedence: ``--room`` flag > ``MDLLM_ROOM`` env > saved file.
"""

from __future__ import annotations

import os
import re
import secrets
from pathlib import Path

from . import config

ROOM_FILE: Path = config.PROJECT_ROOT / ".mdllm" / "room"

# Small, friendly wordlists so codes are easy to read aloud / share.
_ADJECTIVES = [
    "amber", "brave", "clever", "cosmic", "crimson", "electric", "gentle",
    "golden", "hidden", "jolly", "lunar", "mellow", "nimble", "purple",
    "quiet", "rapid", "scarlet", "silent", "solar", "swift", "teal", "vivid",
]
_NOUNS = [
    "otter", "falcon", "maple", "comet", "harbor", "lynx", "canyon", "raven",
    "willow", "ember", "pixel", "cobra", "delta", "quartz", "meadow", "orbit",
    "badger", "cedar", "puffin", "walrus", "zephyr", "beacon",
]

_VALID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{1,62}$")


def generate_code() -> str:
    """Return a memorable, network-safe room code like ``swift-otter-4821``."""
    adj = secrets.choice(_ADJECTIVES)
    noun = secrets.choice(_NOUNS)
    num = secrets.randbelow(9000) + 1000
    return f"{adj}-{noun}-{num}"


def is_valid(code: str) -> bool:
    return bool(_VALID.match(code))


def save_room(code: str) -> None:
    if not is_valid(code):
        raise ValueError(
            f"Invalid room code {code!r}. Use letters, digits, '.', '-', '_' "
            "(2-63 chars)."
        )
    ROOM_FILE.parent.mkdir(parents=True, exist_ok=True)
    ROOM_FILE.write_text(code.strip() + "\n", encoding="utf-8")


def load_saved_room() -> str | None:
    if ROOM_FILE.exists():
        code = ROOM_FILE.read_text(encoding="utf-8").strip()
        return code or None
    return None


def clear_room() -> None:
    if ROOM_FILE.exists():
        ROOM_FILE.unlink()


def resolve_room(explicit: str | None = None) -> str | None:
    """Active room by precedence: explicit flag > MDLLM_ROOM env > saved file."""
    if explicit:
        return explicit
    env = os.environ.get("MDLLM_ROOM")
    if env:
        return env
    return load_saved_room()
