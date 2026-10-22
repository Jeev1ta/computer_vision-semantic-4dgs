#!/usr/bin/env bash
# scripts/run_pipeline.sh
# Runs the full 2D->3D semantic association pipeline, ablation, and evaluation end-to-end.
set -e
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python3 "$ROOT_DIR/scripts/execute_all.py"

