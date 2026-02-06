#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

nvcc -o "$ROOT_DIR/test_cuda" "$ROOT_DIR/test_cuda.cu"
"$ROOT_DIR/test_cuda"