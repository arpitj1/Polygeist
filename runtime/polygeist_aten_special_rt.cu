// Fixed ABI instances of public Thrust transform with pinned ATen scalar
// functions. There is no application matching or formula selection here.
#define AT_DISABLE_JITERATOR
#include <ATen/native/Math.h>
#include <cuda.h>
#include <ATen/Dispatch.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/native/cuda/Math.cuh>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include "polygeist_cuda_graph_rt.h"

static void check_transform() {
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    std::fprintf(stderr, "polygeist ATen scalar transform: %s\n",
                 cudaGetErrorString(status));
    std::abort();
  }
}
#define POLYGEIST_ATEN_SCALAR1(NAME, IMPL, ENTRY)                            \
  struct Transform_##NAME {                                                 \
    __host__ __device__ float operator()(float x) const {                    \
      return IMPL<float>(x);                                                \
    }                                                                      \
  };                                                                       \
  extern "C" void ENTRY(int32_t n, const float *input, float *output) {     \
    if (n <= 0 || !input || !output) std::abort();                            \
    auto stream = static_cast<cudaStream_t>(polygeist_cuda_graph_stream());  \
    thrust::transform(thrust::cuda::par_nosync.on(stream), input, input + n,  \
                      output, Transform_##NAME{});                         \
    check_transform();                                                     \
  }
#define POLYGEIST_ATEN_SCALAR2(NAME, IMPL, ENTRY)                            \
  struct Transform_##NAME {                                                 \
    __host__ __device__ float operator()(float x, float y) const {           \
      return IMPL<float>(x, y);                                             \
    }                                                                      \
  };                                                                       \
  extern "C" void ENTRY(int32_t n, const float *left, const float *right,   \
                        float *output) {                                   \
    if (n <= 0 || !left || !right || !output) std::abort();                   \
    auto stream = static_cast<cudaStream_t>(polygeist_cuda_graph_stream());  \
    thrust::transform(thrust::cuda::par_nosync.on(stream), left, left + n,    \
                      right, output, Transform_##NAME{});                  \
    check_transform();                                                     \
  }
#define POLYGEIST_ATEN_SCALARINT(NAME, IMPL, ENTRY)                          \
  struct Transform_##NAME {                                               \
    int order;                                                             \
    __host__ __device__ float operator()(float x) const {                    \
      return IMPL<float>(x, order);                                         \
    }                                                                      \
  };                                                                       \
  extern "C" void ENTRY(int32_t n, const float *input, int32_t order,         \
                        float *output) {                                   \
    if (n <= 0 || !input || !output) std::abort();                            \
    auto stream = static_cast<cudaStream_t>(polygeist_cuda_graph_stream());  \
    thrust::transform(thrust::cuda::par_nosync.on(stream), input, input + n,  \
                      output, Transform_##NAME{order});                    \
    check_transform();                                                     \
  }
#include "polygeist_aten_scalar_bindings.inc"

struct CudaTrigamma {
  __host__ __device__ float operator()(float x) const {
    return at::native::calc_trigamma<float>(x);
  }
};
extern "C" void polygeist_thrust_calc_trigamma_f32(
    int32_t n, const float *input, float *output) {
  if (n <= 0 || !input || !output) std::abort();
  auto stream = static_cast<cudaStream_t>(polygeist_cuda_graph_stream());
  thrust::transform(thrust::cuda::par_nosync.on(stream), input, input + n,
                    output, CudaTrigamma{});
  check_transform();
}
