// Fixed C ABI for the actual public Thrust transform/divides implementation.
// Inputs/output are contiguous device-visible FP32 arrays; caller marshals views.
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/for_each.h>
#include <thrust/fill.h>
#include <cuda/std/functional>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include "polygeist_cuda_graph_rt.h"

// FP32 specialization of ATen's public log1p CUDA scalar functor. Thrust
// provides the traversal, and CUDA libdevice supplies log1pf itself.
struct Log1pF32 {
  __host__ __device__ float operator()(float value) const {
    return ::log1pf(value);
  }
};
// The public ATen CUDA scalar lambdas use these libdevice operations. Keep
// full-range argument reduction rather than approximate pointwise engines.
struct SinF32 {
  __host__ __device__ float operator()(float x) const { return ::sinf(x); }
};
struct CosF32 {
  __host__ __device__ float operator()(float x) const { return ::cosf(x); }
};
// FP32 specialization of ATen CUDA ComplexKernel.cu's polar lambda.
struct PolarComponentsF32 {
  __host__ __device__ thrust::tuple<float, float> operator()(float magnitude, float angle) const {
    return thrust::make_tuple(magnitude * ::cosf(angle), magnitude * ::sinf(angle));
  }
};
extern "C" void polygeist_thrust_polar_components_f32(
    int32_t n, const float *magnitude, const float *angle, float *real, float *imag) {
  if (n <= 0 || !magnitude || !angle || !real || !imag) std::abort();
  auto stream = static_cast<cudaStream_t>(polygeist_cuda_graph_stream());
  auto output = thrust::make_zip_iterator(thrust::make_tuple(real, imag));
  thrust::transform(thrust::cuda::par_nosync.on(stream), magnitude, magnitude + n,
                    angle, output, PolarComponentsF32{});
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    std::fprintf(stderr, "polygeist Thrust polar: %s\n", cudaGetErrorString(status));
    std::abort();
  }
}
template<class Function>
static void trigonometric_transform(int32_t n, const float *input,
                                    float *output, Function function) {
  if (n <= 0 || !input || !output) std::abort();
  auto stream = static_cast<cudaStream_t>(polygeist_cuda_graph_stream());
  thrust::transform(thrust::cuda::par_nosync.on(stream), input, input + n,
                    output, function);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    std::fprintf(stderr, "polygeist Thrust trigonometry: %s\n", cudaGetErrorString(status));
    std::abort();
  }
}
extern "C" void polygeist_thrust_sin_f32(int32_t n, const float *input, float *output) {
  trigonometric_transform(n, input, output, SinF32{});
}
extern "C" void polygeist_thrust_cos_f32(int32_t n, const float *input, float *output) {
  trigonometric_transform(n, input, output, CosF32{});
}
extern "C" void polygeist_thrust_log1p_f32(int32_t n, const float *input,
                                         float *output) {
  if (n <= 0 || !input || !output) std::abort();
  auto stream = static_cast<cudaStream_t>(polygeist_cuda_graph_stream());
  thrust::transform(thrust::cuda::par_nosync.on(stream), input, input + n,
                    output, Log1pF32{});
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    std::fprintf(stderr, "polygeist Thrust log1p: %s\n", cudaGetErrorString(status));
    std::abort();
  }
}

extern "C" void polygeist_thrust_fill_f32(int32_t n, float value, float *output) {
  if (n <= 0 || !output) std::abort();
  auto stream = static_cast<cudaStream_t>(polygeist_cuda_graph_stream());
  thrust::fill_n(thrust::cuda::par_nosync.on(stream), output, n, value);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    std::fprintf(stderr, "polygeist Thrust fill: %s\n", cudaGetErrorString(status));
    std::abort();
  }
}

// Descriptor-directed byte movement, not arithmetic recognition. Keeping the
// payload byte-typed preserves NaN payloads, signaling NaNs, and signed zeros.
// The caller has checked footprints, device residency, and disjoint buffers.
struct MemrefCopy {
  static constexpr int max_rank = 16;
  int64_t rank, element, sizes[max_rank], source_strides[max_rank],
      destination_strides[max_rank];
  const unsigned char *source;
  unsigned char *destination;
  __device__ void operator()(int64_t linear) const {
    int64_t src = 0, dst = 0;
    for (int64_t dim = rank; dim-- > 0;) {
      int64_t index = linear % sizes[dim];
      linear /= sizes[dim];
      src += index * source_strides[dim];
      dst += index * destination_strides[dim];
    }
    for (int64_t byte = 0; byte < element; ++byte)
      destination[dst * element + byte] = source[src * element + byte];
  }
};

extern "C" int polygeist_thrust_memref_copy(
    int64_t rank, int64_t element, const int64_t *sizes,
    const int64_t *source_strides, const int64_t *destination_strides,
    const void *source, void *destination) {
  if (rank < 1 || rank > MemrefCopy::max_rank) return 0;
  MemrefCopy copy{};
  copy.rank = rank;
  copy.element = element;
  copy.source = static_cast<const unsigned char *>(source);
  copy.destination = static_cast<unsigned char *>(destination);
  int64_t count = 1;
  int order[MemrefCopy::max_rank];
  for (int i = 0; i < rank; ++i) {
    if (sizes[i] <= 0 || destination_strides[i] < 0) return 0;
    copy.sizes[i] = sizes[i];
    copy.source_strides[i] = source_strides[i];
    copy.destination_strides[i] = destination_strides[i];
    count *= sizes[i]; // The caller already checked this product.
    order[i] = i;
  }
  // Prove distinct output elements before launching parallel stores. This
  // sufficient test accepts arbitrary dimension order, padding and unit dims.
  for (int i = 1; i < rank; ++i) {
    int dim = order[i], j = i;
    while (j && destination_strides[order[j - 1]] > destination_strides[dim]) {
      order[j] = order[j - 1];
      --j;
    }
    order[j] = dim;
  }
  __int128 span = 1;
  for (int i = 0; i < rank; ++i) {
    int dim = order[i];
    if (sizes[dim] == 1) continue;
    if (destination_strides[dim] < span) return 0;
    span += (__int128)(sizes[dim] - 1) * destination_strides[dim];
  }
  auto stream = static_cast<cudaStream_t>(polygeist_cuda_graph_stream());
  thrust::for_each_n(thrust::cuda::par_nosync.on(stream),
                    thrust::make_counting_iterator<int64_t>(0), count, copy);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    std::fprintf(stderr, "polygeist descriptor copy: %s\n", cudaGetErrorString(status));
    std::abort();
  }
  return 1;
}

template<class Left, class Right>
static void thrust_divide(int32_t n, Left left, Right right, float *output) {
  if (n <= 0 || !output) std::abort();
  const char *trace = std::getenv("POLYGEIST_V2_TRACE");
  if (trace && trace[0] != '0')
    std::fprintf(stderr, "POLYGEIST_VENDOR_CALL library=Thrust entry=transform functor=divides_f32 n=%d\n", n);
  auto stream = static_cast<cudaStream_t>(polygeist_cuda_graph_stream());
  thrust::transform(thrust::cuda::par_nosync.on(stream), left, left+n, right,
                    output, cuda::std::divides<float>{});
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    std::fprintf(stderr, "polygeist Thrust transform: %s\n", cudaGetErrorString(status));
    std::abort();
  }
}
extern "C" void polygeist_thrust_divide_f32(int32_t n,const float*x,const float*y,float*out) {
  thrust_divide(n,x,y,out);
}
extern "C" void polygeist_thrust_divide_scalar_left_f32(int32_t n,float x,const float*y,float*out) {
  thrust_divide(n,thrust::make_constant_iterator(x),y,out);
}
extern "C" void polygeist_thrust_divide_scalar_right_f32(int32_t n,const float*x,float y,float*out) {
  thrust_divide(n,x,thrust::make_constant_iterator(y),out);
}
extern "C" void polygeist_thrust_divide_scalars_f32(int32_t n,float x,float y,float*out) {
  thrust_divide(n,thrust::make_constant_iterator(x),thrust::make_constant_iterator(y),out);
}

// Actual public binary functors, with pointer/constant iterator variants.
template<class Left, class Right, class Element, class Functor>
static void thrust_arithmetic(int32_t n, Left left, Right right, Element *output,
                              Functor operation, const char *name) {
  if (n <= 0 || !output) std::abort();
  const char *trace = std::getenv("POLYGEIST_V2_TRACE");
  if (trace && trace[0] != '0')
    std::fprintf(stderr, "POLYGEIST_VENDOR_CALL library=Thrust entry=transform functor=%s n=%d\n", name, n);
  auto stream = static_cast<cudaStream_t>(polygeist_cuda_graph_stream());
  thrust::transform(thrust::cuda::par_nosync.on(stream), left, left+n, right,
                    output, operation);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    std::fprintf(stderr, "polygeist Thrust transform: %s\n", cudaGetErrorString(status));
    std::abort();
  }
}
extern "C" void polygeist_thrust_plus_f32(int32_t n,const float*x,const float*y,float*out) {
  thrust_arithmetic(n,x,y,out,cuda::std::plus<float>{},"plus_f32");
}
extern "C" void polygeist_thrust_plus_scalar_left_f32(int32_t n,float x,const float*y,float*out) {
  thrust_arithmetic(n,thrust::make_constant_iterator(x),y,out,cuda::std::plus<float>{},"plus_f32");
}
extern "C" void polygeist_thrust_plus_scalar_right_f32(int32_t n,const float*x,float y,float*out) {
  thrust_arithmetic(n,x,thrust::make_constant_iterator(y),out,cuda::std::plus<float>{},"plus_f32");
}
extern "C" void polygeist_thrust_plus_scalars_f32(int32_t n,float x,float y,float*out) {
  thrust_arithmetic(n,thrust::make_constant_iterator(x),thrust::make_constant_iterator(y),out,cuda::std::plus<float>{},"plus_f32");
}
extern "C" void polygeist_thrust_minus_f32(int32_t n,const float*x,const float*y,float*out) {
  thrust_arithmetic(n,x,y,out,cuda::std::minus<float>{},"minus_f32");
}
extern "C" void polygeist_thrust_minus_scalar_left_f32(int32_t n,float x,const float*y,float*out) {
  thrust_arithmetic(n,thrust::make_constant_iterator(x),y,out,cuda::std::minus<float>{},"minus_f32");
}
extern "C" void polygeist_thrust_minus_scalar_right_f32(int32_t n,const float*x,float y,float*out) {
  thrust_arithmetic(n,x,thrust::make_constant_iterator(y),out,cuda::std::minus<float>{},"minus_f32");
}
extern "C" void polygeist_thrust_minus_scalar_right_i32(int32_t n,const int*x,int y,int*out) {
  thrust_arithmetic(n,x,thrust::make_constant_iterator(y),out,cuda::std::minus<int>{},"minus_i32");
}
extern "C" void polygeist_thrust_minus_scalars_f32(int32_t n,float x,float y,float*out) {
  thrust_arithmetic(n,thrust::make_constant_iterator(x),thrust::make_constant_iterator(y),out,cuda::std::minus<float>{},"minus_f32");
}
extern "C" void polygeist_thrust_multiplies_f32(int32_t n,const float*x,const float*y,float*out) {
  thrust_arithmetic(n,x,y,out,cuda::std::multiplies<float>{},"multiplies_f32");
}
extern "C" void polygeist_thrust_multiplies_scalar_left_f32(int32_t n,float x,const float*y,float*out) {
  thrust_arithmetic(n,thrust::make_constant_iterator(x),y,out,cuda::std::multiplies<float>{},"multiplies_f32");
}
extern "C" void polygeist_thrust_multiplies_scalar_right_f32(int32_t n,const float*x,float y,float*out) {
  thrust_arithmetic(n,x,thrust::make_constant_iterator(y),out,cuda::std::multiplies<float>{},"multiplies_f32");
}
extern "C" void polygeist_thrust_multiplies_scalars_f32(int32_t n,float x,float y,float*out) {
  thrust_arithmetic(n,thrust::make_constant_iterator(x),thrust::make_constant_iterator(y),out,cuda::std::multiplies<float>{},"multiplies_f32");
}
