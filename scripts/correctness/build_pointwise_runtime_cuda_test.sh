#!/usr/bin/env bash
# Build the CUDA runtime memory/capture regression. Run the result on a CUDA GPU.
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
: "${CC:=cc}"
: "${CUDA_TARGET_ROOT:=/usr/local/cuda}"
: "${CUDNN_INCLUDE:=/usr/include}"
: "${CUDNN_LIB:=/usr/lib}"
: "${CUTENSOR_ROOT:?Set CUTENSOR_ROOT to the cuTENSOR installation root}"
: "${NVCC:=/usr/local/cuda/bin/nvcc}"
output=${1:?Usage: build_pointwise_runtime_cuda_test.sh OUTPUT_BINARY}
mkdir -p "$(dirname -- "$output")"
build_dir=$(mktemp -d)
trap 'rm -rf "$build_dir"' EXIT
includes=(-I"$repo_root/runtime" -I"$CUDA_TARGET_ROOT/include" -I"$CUDNN_INCLUDE" -I"$CUTENSOR_ROOT/include")
"$CC" -O2 -ffunction-sections -fdata-sections -DPOLYGEIST_DISABLE_CUSPARSE \
  -DPOLYGEIST_DISABLE_CUSOLVER -DPOLYGEIST_ENABLE_CUTENSOR "${includes[@]}" \
  -c "$repo_root/runtime/polygeist_cublas_rt_cuda.c" -o "$build_dir/runtime.o"
"$CC" -O2 "${includes[@]}" \
  -c "$repo_root/scripts/correctness/tests/test_pointwise_runtime_cuda.c" -o "$build_dir/test.o"
nvcc_flags=(-O2 -std=c++17 -arch="${POLYGEIST_GPU_ARCH:-sm_87}")
if [ -n "${CUDA_NVCC_CCBIN:-}" ]; then
  nvcc_flags+=(-ccbin="$CUDA_NVCC_CCBIN")
fi
if [ -n "${CUDA_NVCC_TARGET_DIR:-}" ]; then
  nvcc_flags+=(-target-dir "$CUDA_NVCC_TARGET_DIR")
fi
"$NVCC" "${nvcc_flags[@]}" -Xcompiler=-ffunction-sections,-fdata-sections \
  -I"$repo_root/runtime" -I"$repo_root/third_party/cuda_headers/cuda_cccl/include/cccl" \
  -c "$repo_root/runtime/polygeist_thrust_rt.cu" -o "$build_dir/thrust.o"
"$CC" "$build_dir/test.o" "$build_dir/runtime.o" "$build_dir/thrust.o" \
  -L"$CUDA_TARGET_ROOT/lib" -L"$CUDA_TARGET_ROOT/lib64" -L"$CUDA_TARGET_ROOT/lib/stubs" -L"$CUDNN_LIB" -L"$CUTENSOR_ROOT/lib" \
  -Wl,--gc-sections -Wl,--allow-shlib-undefined \
  -lcudnn -lcublas -lcublasLt -lcutensor -lcudart -lstdc++ -lm -ldl -o "$output"
printf 'Built %s. Run it with CUDA/cuDNN/cuTENSOR on LD_LIBRARY_PATH.\n' "$output"
