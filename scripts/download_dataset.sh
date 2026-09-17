#!/usr/bin/env bash
# Run again after an interruption to reuse completed downloads.
set -euo pipefail

workers="${1:-32}"
if [[ $# -gt 1 || ! "$workers" =~ ^[1-9][0-9]*$ ]]; then
    echo "Usage: bash scripts/download_dataset.sh [positive worker count]" >&2
    exit 2
fi
command -v uvx >/dev/null || { echo "Install uv first (uvx is required)." >&2; exit 1; }
command -v flock >/dev/null || { echo "Install util-linux (flock is required)." >&2; exit 1; }

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
mkdir -p data/tsrd
# Prevent two instances of this script competing for the same files.
exec 9>data/tsrd/.download.lock
flock -n 9 || { echo "Another dataset download script is already running." >&2; exit 1; }

export HF_XET_HIGH_PERFORMANCE=1
echo "Downloading current scan/stare splits with $workers workers (archive excluded)."
echo "Destination: $project_dir/data/tsrd"
echo "If access is denied, request dataset access and run: uvx hf auth login"
uvx hf download alan-turing-institute/turing-synthetic-radar-dataset \
    --repo-type dataset \
    --local-dir data/tsrd \
    --exclude 'archive/**' \
    --max-workers "$workers"
echo "Download complete."
