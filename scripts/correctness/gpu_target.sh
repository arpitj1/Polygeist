#!/usr/bin/env bash
# Shared CUDA toolchain selection. Sourced by the build drivers.
configure_gpu_target() {
  POLYGEIST_GPU_TARGET=${1:-${POLYGEIST_GPU_TARGET:-x86_64}}
  GPU_CLANG_ARGS=()
  GPU_NVCC_ARGS=()
  case "$POLYGEIST_GPU_TARGET" in
    x86_64)
      GPU_CC=${POLYGEIST_HOST_CC:-gcc}
      GPU_CXX=${POLYGEIST_HOST_CXX:-g++}
      GPU_READELF=readelf
      GPU_TRIPLE=x86_64-linux-gnu
      GPU_ARCH=${POLYGEIST_GPU_ARCH:-sm_80}
      CUDA=${CUDA:-${POLYGEIST_CUDA_ROOT:-/usr/local/cuda}/targets/x86_64-linux}
      GPU_SYSTEM_LIB=/usr/lib/x86_64-linux-gnu
      GPU_CLANG_ARGS=(--target=x86_64-linux-gnu --gcc-toolchain=/usr)
      GPU_NVCC_ARGS=(-ccbin "$GPU_CXX")
      ;;
    jetson)
      GPU_CC=${AARCH64_CC:-aarch64-linux-gnu-gcc}
      GPU_CXX=${POLYGEIST_CROSS_CXX:-${AARCH64_CXX:-aarch64-linux-gnu-g++}}
      GPU_READELF=${AARCH64_READELF:-aarch64-linux-gnu-readelf}
      GPU_TRIPLE=aarch64-linux-gnu
      GPU_ARCH=${POLYGEIST_GPU_ARCH:-sm_87}
      CUDA=${CUDA:-${POLYGEIST_CUDA_CROSS_ROOT:-/usr/local/cuda-12.6/targets/sbsa-linux}}
      GPU_SYSTEM_LIB=/usr/lib/aarch64-linux-gnu
      GPU_CLANG_ARGS=(--target=aarch64-linux-gnu --gcc-toolchain=/usr)
      GPU_NVCC_ARGS=(-ccbin "$GPU_CXX" -target-dir sbsa-linux)
      ;;
    *) echo "ERROR: GPU target must be x86_64 or jetson" >&2; return 1 ;;
  esac
  CUDA_LIBDIR=${POLYGEIST_CUDA_LIBDIR:-$CUDA/lib}
  if [ ! -d "$CUDA_LIBDIR" ] && [ -d "$CUDA/lib64" ]; then CUDA_LIBDIR=$CUDA/lib64; fi
  GPU_CUDA_TOOLKIT_ROOT=$CUDA
  if [ ! -d "$CUDA/nvvm" ]; then GPU_CUDA_TOOLKIT_ROOT=$(dirname "$(dirname "$CUDA")"); fi
  CUDNN_INC=${CUDNN_INC:-${POLYGEIST_CUDNN_INCLUDE:-/usr/include/${GPU_TRIPLE}}}
  CUDNN_LIB=${CUDNN_LIB:-${POLYGEIST_CUDNN_LIBDIR:-$GPU_SYSTEM_LIB}}
  GPU_RPATH="/usr/local/cuda/lib64:$GPU_SYSTEM_LIB:$CUDNN_LIB"
  export POLYGEIST_GPU_TARGET CUDA GPU_ARCH
}
