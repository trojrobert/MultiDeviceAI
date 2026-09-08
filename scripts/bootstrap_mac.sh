#!/usr/bin/env bash
#
# One-shot setup for a macOS (Apple Silicon) node.
# Installs prerequisites, then clones + builds the exo engine.
#
# Usage: ./scripts/bootstrap_mac.sh
set -euo pipefail

EXO_DIR="${MDLLM_EXO_DIR:-$(cd "$(dirname "$0")/.." && pwd)/vendor/exo}"
EXO_REPO="${MDLLM_EXO_REPO:-https://github.com/exo-explore/exo}"

echo "==> Checking Homebrew"
if ! command -v brew >/dev/null 2>&1; then
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

echo "==> Installing uv + node"
brew install uv node || true

echo "==> Checking Rust (nightly)"
if ! command -v cargo >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  # shellcheck disable=SC1090
  source "$HOME/.cargo/env"
fi
rustup toolchain install nightly || true

echo "==> Installing macmon (hardware monitoring)"
cargo install --git https://github.com/vladkens/macmon macmon --force || true

echo "==> Fetching exo into ${EXO_DIR}"
if [ -d "${EXO_DIR}/.git" ]; then
  git -C "${EXO_DIR}" pull --ff-only
else
  mkdir -p "$(dirname "${EXO_DIR}")"
  git clone "${EXO_REPO}" "${EXO_DIR}"
fi

echo "==> Building exo dashboard"
( cd "${EXO_DIR}/dashboard" && npm install && npm run build )

echo "==> Installing exo Python deps (MLX backend)"
( cd "${EXO_DIR}" && uv sync --extra mlx )

echo ""
echo "Done. Start this node with:  mdllm up   (or: cd ${EXO_DIR} && uv run exo)"
