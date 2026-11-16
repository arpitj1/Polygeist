#!/usr/bin/env bash
# Native CUDA build by default; POLYGEIST_GPU_TARGET=jetson selects cross-builds.
set -euo pipefail
export POLYGEIST_GPU_TARGET=${POLYGEIST_GPU_TARGET:-x86_64}
exec bash "$(dirname "$0")/build_jetson.sh" "$@"
