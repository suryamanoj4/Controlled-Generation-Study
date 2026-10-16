#!/usr/bin/env bash
# SAS-E1 setup with uv (https://docs.astral.sh/uv/):
#   - creates .venv with the pinned Python version, installs deps
#   - downloads models and/or prepares datasets (idempotent)
# Usage:
#   bash scripts/setup.sh [configs/lean.yaml]                  # models + data
#   bash scripts/setup.sh [configs/lean.yaml] --data-only      # datasets only
#   bash scripts/setup.sh [configs/lean.yaml] --models-only    # model snapshots only
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
  echo "[setup] uv not found — installing it..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

CONFIG="${1:-configs/lean.yaml}"
shift || true
EXTRA="${*:-}"

echo "[setup] creating environment + installing dependencies"
uv sync

echo "[setup] downloading models + preparing datasets (${CONFIG})"
uv run python scripts/download_assets.py --config "$CONFIG" ${EXTRA:+"$EXTRA"}

echo
echo "Setup complete. Next steps:"
echo "  uv run python scripts/run_experiment.py --config $CONFIG --stage all"
echo "  uv run python scripts/smoke_test.py        # quick numerical sanity check"
echo
echo "Selective setup: pass --data-only or --models-only to this script (or to"
echo "scripts/download_assets.py) to run just one half."