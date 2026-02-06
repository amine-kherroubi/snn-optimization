#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

TEST_BIN="$ROOT_DIR/test_cuda"

# trap on error only
trap 'if [ $? -ne 0 ]; then rm -f "$TEST_BIN"; fi' ERR

nvcc -o "$TEST_BIN" "$ROOT_DIR/test_cuda.cu"

"$TEST_BIN"