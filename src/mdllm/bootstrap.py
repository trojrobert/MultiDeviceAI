"""Environment checks and one-command setup for the exo engine.

`doctor` verifies prerequisites; `bootstrap` clones + builds exo into
`config.EXO_DIR`; `up` launches the cluster node.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
from dataclasses import dataclass

from . import config


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    hint: str = ""


def _which(cmd: str) -> str | None:
    return shutil.which(cmd)


def _xcode_ok() -> bool:
    if platform.system() != "Darwin":
        return True
    try:
        subprocess.run(
            ["xcode-select", "-p"],
            check=True,
            capture_output=True,
            text=True,
        )
        return True
    except Exception:
        return False


def doctor() -> list[Check]:
    """Return prerequisite checks for the current OS."""
    is_mac = platform.system() == "Darwin"
    checks: list[Check] = []

    # git — always needed to fetch exo
    git = _which("git")
    checks.append(
        Check("git", bool(git), git or "not found", "Install Xcode CLT or git")
    )

    # uv — python dependency manager exo uses
    uv = _which("uv")
    checks.append(
        Check(
            "uv",
            bool(uv),
            uv or "not found",
            "curl -LsSf https://astral.sh/uv/install.sh | sh",
        )
    )

    # node — to build exo's dashboard
    node = _which("node")
    checks.append(
        Check("node", bool(node), node or "not found", "brew install node")
    )

    # rust/cargo — exo builds rust bindings (nightly)
    cargo = _which("cargo")
    checks.append(
        Check(
            "rust (cargo)",
            bool(cargo),
            cargo or "not found",
            "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh "
            "&& rustup toolchain install nightly",
        )
    )

    if is_mac:
        brew = _which("brew")
        checks.append(
            Check(
                "brew",
                bool(brew),
                brew or "not found",
                '/bin/bash -c "$(curl -fsSL '
                'https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"',
            )
        )
        checks.append(
            Check(
                "Xcode Metal toolchain",
                _xcode_ok(),
                "xcode-select -p ok" if _xcode_ok() else "not found",
                "Install Xcode (required by MLX for Metal compilation)",
            )
        )
        macmon = _which("macmon")
        checks.append(
            Check(
                "macmon (optional)",
                bool(macmon),
                macmon or "not found",
                "cargo install --git https://github.com/vladkens/macmon macmon --force",
            )
        )

    return checks


def clone_or_update_exo() -> list[str]:
    """Return the shell commands used to fetch/update exo (for transparency)."""
    if config.EXO_DIR.exists():
        return [f"cd {config.EXO_DIR} && git pull --ff-only"]
    return [
        f"mkdir -p {config.EXO_DIR.parent}",
        f"git clone {config.EXO_REPO} {config.EXO_DIR}",
    ]


def build_commands() -> list[str]:
    """The build/sync steps run inside the exo checkout."""
    mlx_extra = "mlx" if platform.system() == "Darwin" else "mlx-cpu"
    return [
        f"cd {config.EXO_DIR}/dashboard && npm install && npm run build",
        f"cd {config.EXO_DIR} && uv sync --extra {mlx_extra}",
    ]


def run_command(worker: bool = True) -> str:
    flag = "" if worker else " --no-worker"
    return f"cd {config.EXO_DIR} && uv run exo{flag}"
