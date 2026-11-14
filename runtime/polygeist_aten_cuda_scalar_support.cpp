// The public CUDA definition is parsed on the host for reference import.
// C++ math.h exposes the same float overload of ::sin selected by CUDA.
#include <math.h>
#define AT_DISABLE_JITERATOR
#include <ATen/native/Math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/Dispatch.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/native/cuda/Math.cuh>
extern "C" float calc_cuda_trigammaf(float x) {
  return at::native::calc_trigamma<float>(x);
}
