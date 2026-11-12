#!/usr/bin/env bash
set -euo pipefail
# ccache tracks the CUDA source, headers, compiler and flags for each object.
compiler="${POLYGEIST_REAL_NVCC:-/usr/local/cuda-12.6/bin/nvcc}"
if command -v ccache >/dev/null 2>&1; then
  exec ccache "$compiler" "$@"
fi
exec "$compiler" "$@"
