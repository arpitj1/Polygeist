#include "polygeist_thrust_gather.h"
#include <thrust/gather.h>
#include <thrust/iterator/transform_iterator.h>
// Fixed C ABI for the actual public Thrust transform/divides implementation.
// Inputs/output are contiguous device-visible FP32 arrays; caller marshals views.
#include <thrust/transform.h>
#include <thrust/copy.h>
#include <thrust/reverse.h>
#include <cuda/std/iterator>
#include <thrust/replace.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/for_each.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <cuda/std/functional>
#include <cuda/functional>
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
static void scalar_transform(int32_t n, const float *input,
                                    float *output, Function function) {
  if (n <= 0 || !input || !output) std::abort();
  auto stream = static_cast<cudaStream_t>(polygeist_cuda_graph_stream());
  thrust::transform(thrust::cuda::par_nosync.on(stream), input, input + n,
                    output, function);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    std::fprintf(stderr, "polygeist Thrust transform: %s\n", cudaGetErrorString(status));
    std::abort();
  }
}
extern "C" void polygeist_thrust_sin_f32(int32_t n, const float *input, float *output) {
  scalar_transform(n, input, output, SinF32{});
}
extern "C" void polygeist_thrust_cos_f32(int32_t n, const float *input, float *output) {
  scalar_transform(n, input, output, CosF32{});
}
// Fixed-ABI instances of the imported public scalar functors. These wrappers
// only dispatch to Thrust; specification matching determines their arguments.
#define POLYGEIST_UNARY_MATH(NAME)                                            \
  struct Math_##NAME {                                                       \
    __host__ __device__ float operator()(float x) const {                     \
      return ::NAME##f(x);                                                   \
    }                                                                       \
  };                                                                        \
  extern "C" void polygeist_thrust_##NAME##_f32(                             \
      int32_t n, const float *input, float *output) {                         \
    scalar_transform(n, input, output, Math_##NAME{});                        \
  }
POLYGEIST_UNARY_MATH(exp2)
POLYGEIST_UNARY_MATH(expm1)
POLYGEIST_UNARY_MATH(log2)
POLYGEIST_UNARY_MATH(log10)
POLYGEIST_UNARY_MATH(erfc)
POLYGEIST_UNARY_MATH(lgamma)
POLYGEIST_UNARY_MATH(round)
#undef POLYGEIST_UNARY_MATH

template<class Function>
static void scalar_transform(int32_t n, const float *left, const float *right,
                             float *output, Function function) {
  if (n <= 0 || !left || !right || !output) std::abort();
  auto stream = static_cast<cudaStream_t>(polygeist_cuda_graph_stream());
  thrust::transform(thrust::cuda::par_nosync.on(stream), left, left + n, right,
                    output, function);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    std::fprintf(stderr, "polygeist Thrust transform: %s\n", cudaGetErrorString(status));
    std::abort();
  }
}
#define POLYGEIST_BINARY_MATH(NAME)                                           \
  struct Math_##NAME {                                                       \
    __host__ __device__ float operator()(float x, float y) const {             \
      return ::NAME##f(x, y);                                                \
    }                                                                       \
  };                                                                        \
  extern "C" void polygeist_thrust_##NAME##_f32(                             \
      int32_t n, const float *left, const float *right, float *output) {       \
    scalar_transform(n, left, right, output, Math_##NAME{});                   \
  }
POLYGEIST_BINARY_MATH(hypot)
POLYGEIST_BINARY_MATH(fmod)
POLYGEIST_BINARY_MATH(fmax)
POLYGEIST_BINARY_MATH(fmin)
POLYGEIST_BINARY_MATH(nextafter)
POLYGEIST_BINARY_MATH(copysign)
POLYGEIST_BINARY_MATH(remainder)
#undef POLYGEIST_BINARY_MATH
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

// Descriptor-directed bit movement, not arithmetic recognition. Integer vector
// loads preserve every payload bit (including signaling NaNs and signed zeros).
// The caller checks footprints, device residency, and disjoint source/output.
struct MemrefCopy {
  static constexpr int max_rank = 16;
  int64_t rank, element, sizes[max_rank], source_strides[max_rank],
      destination_strides[max_rank];
  uint32_t reciprocal[max_rank];
  const unsigned char *source;
  unsigned char *destination;
};

// Exchange the two contiguous axes through shared memory. Reading and writing
// in their respective physical orders avoids a scattered transaction per value
// for transposed views. This copies bits only; the descriptors determine every
// address, and the caller has already established disjoint storage and unique
// destinations. Other axes (including source broadcasts) are independent tiles.
__device__ static uint32_t copy_quotient(uint32_t n, uint32_t d,
                                        uint32_t reciprocal) {
  uint32_t q = d == 1 ? n : __umulhi(n, reciprocal);
  return q - (q * d > n);
}

static uint32_t copy_reciprocal(uint32_t divisor) {
  return divisor == 1 ? 0 :
      static_cast<uint32_t>(((uint64_t(1) << 32) + divisor - 1) / divisor);
}

template<class Word, bool Narrow>
__global__ void memref_copy_transpose(MemrefCopy copy, int source_axis,
                                      int destination_axis,
                                      int64_t source_tiles,
                                      int64_t destination_tiles,
                                      uint32_t source_reciprocal,
                                      uint32_t destination_reciprocal,
                                      int64_t source_size, int64_t destination_size,
                                      int64_t source_cross_stride,
                                      int64_t destination_cross_stride) {
  __shared__ Word tile[32][33];
  int64_t batch = blockIdx.x;
  int64_t sx, dy;
  if (Narrow) {
    uint32_t q = copy_quotient(batch, source_tiles, source_reciprocal);
    sx = (batch - q * source_tiles) * 32;
    batch = q;
    q = copy_quotient(batch, destination_tiles, destination_reciprocal);
    dy = (batch - q * destination_tiles) * 32;
    batch = q;
  } else {
    sx = (batch % source_tiles) * 32;
    batch /= source_tiles;
    dy = (batch % destination_tiles) * 32;
    batch /= destination_tiles;
  }
  int64_t source_base = 0, destination_base = 0;
  #pragma unroll
  for (int dim = MemrefCopy::max_rank - 1; dim >= 0; --dim) {
    if (dim >= copy.rank || dim == source_axis || dim == destination_axis)
      continue;
    int64_t coordinate;
    if (Narrow) {
      uint32_t q = copy_quotient(batch, copy.sizes[dim], copy.reciprocal[dim]);
      coordinate = batch - q * copy.sizes[dim];
      batch = q;
    } else {
      coordinate = batch % copy.sizes[dim];
      batch /= copy.sizes[dim];
    }
    source_base += coordinate * copy.source_strides[dim];
    destination_base += coordinate * copy.destination_strides[dim];
  }
  const Word *source = reinterpret_cast<const Word *>(copy.source);
  Word *destination = reinterpret_cast<Word *>(copy.destination);
  for (int step = 0; step < 32; step += 8) {
    int64_t x = sx + threadIdx.x, y = dy + threadIdx.y + step;
    if (x < source_size && y < destination_size)
      tile[threadIdx.y + step][threadIdx.x] = source[source_base +
          x + y * source_cross_stride];
  }
  __syncthreads();
  for (int step = 0; step < 32; step += 8) {
    int64_t x = sx + threadIdx.y + step, y = dy + threadIdx.x;
    if (x < source_size && y < destination_size)
      destination[destination_base + x * destination_cross_stride + y] =
          tile[threadIdx.x][threadIdx.y + step];
  }
}

template<bool Narrow>
static bool launch_tiled_memref_copy(const MemrefCopy &copy, int64_t count,
                                     cudaStream_t stream) {
  if (count < 4096) return false;
  int source_axis = -1, destination_axis = -1;
  for (int dim = 0; dim < copy.rank; ++dim) {
    if (copy.source_strides[dim] == 1) source_axis = dim;
    if (copy.destination_strides[dim] == 1) destination_axis = dim;
  }
  if (source_axis < 0 || destination_axis < 0 || source_axis == destination_axis ||
      copy.sizes[source_axis] < 8 || copy.sizes[destination_axis] < 8)
    return false;
  // Small axis pairs leave most of each 32x32 tile empty. The flat traversal
  // combines those short independent batches into full blocks instead.
  if (std::min(copy.sizes[source_axis], int64_t(32)) *
      std::min(copy.sizes[destination_axis], int64_t(32)) < 512)
    return false;
  int64_t source_tiles = (copy.sizes[source_axis] + 31) / 32;
  int64_t destination_tiles = (copy.sizes[destination_axis] + 31) / 32;
  int64_t tiles = (count / copy.sizes[source_axis] /
                   copy.sizes[destination_axis]) * source_tiles * destination_tiles;
  if (tiles > INT32_MAX) return false;
  uintptr_t alignment = reinterpret_cast<uintptr_t>(copy.source) |
                        reinterpret_cast<uintptr_t>(copy.destination);
  if (alignment % copy.element) return false;
  dim3 threads(32, 8);
  #define COPY_TILED_WORD(bytes, type) \
    case bytes: memref_copy_transpose<type, Narrow><<<tiles, threads, 0, stream>>>( \
        copy, source_axis, destination_axis, source_tiles, destination_tiles, \
        Narrow ? copy_reciprocal(source_tiles) : 0, \
        Narrow ? copy_reciprocal(destination_tiles) : 0, \
        copy.sizes[source_axis], copy.sizes[destination_axis], \
        copy.source_strides[destination_axis], \
        copy.destination_strides[source_axis]); break
  switch (copy.element) {
    COPY_TILED_WORD(1, uint8_t);
    COPY_TILED_WORD(2, uint16_t);
    COPY_TILED_WORD(4, uint32_t);
    COPY_TILED_WORD(8, uint2);
    COPY_TILED_WORD(16, uint4);
    default: return false;
  }
  #undef COPY_TILED_WORD
  return true;
}

template<class Word, bool Narrow> struct MemrefCopyOperation : MemrefCopy {
  __device__ void operator()(int64_t linear) const {
    int64_t src = 0, dst = 0;
    // Constant descriptor indices avoid a per-thread local-memory copy of all
    // stride/divisor arrays. Unused dimensions have no arithmetic or accesses.
    #pragma unroll
    for (int dim = max_rank - 1; dim >= 0; --dim) {
      if (dim >= rank) continue;
      int64_t index;
      if (Narrow) {
        // For n < 2^31 and 1 < d <= 2^31, ceil(2^32/d) gives either
        // floor(n/d) or one more. Correct that one-sided error exactly.
        // q*d fits uint32_t because it is at most n+d < 2^32.
        uint32_t n = static_cast<uint32_t>(linear);
        uint32_t d = static_cast<uint32_t>(sizes[dim]);
        uint32_t q = copy_quotient(n, d, reciprocal[dim]);
        index = n - q * d;
        linear = q;
      } else {
        index = linear % sizes[dim];
        linear /= sizes[dim];
      }
      src += index * source_strides[dim];
      dst += index * destination_strides[dim];
    }
    const Word *input = reinterpret_cast<const Word *>(source + src * element);
    Word *output = reinterpret_cast<Word *>(destination + dst * element);
    for (int64_t word = 0; word < element / sizeof(Word); ++word)
      output[word] = input[word];
  }
};

template<class Word, bool Narrow>
static void launch_memref_copy(const MemrefCopy &copy, int64_t count,
                               cudaStream_t stream) {
  MemrefCopyOperation<Word, Narrow> operation;
  static_cast<MemrefCopy &>(operation) = copy;
  thrust::for_each_n(thrust::cuda::par_nosync.on(stream),
                    thrust::make_counting_iterator<int64_t>(0), count, operation);
}

template<bool Narrow>
static void launch_aligned_memref_copy(const MemrefCopy &copy, int64_t count,
                                       cudaStream_t stream) {
  uintptr_t alignment = reinterpret_cast<uintptr_t>(copy.source) |
                        reinterpret_cast<uintptr_t>(copy.destination) |
                        static_cast<uintptr_t>(copy.element);
  if (!(alignment & 15)) launch_memref_copy<uint4, Narrow>(copy, count, stream);
  else if (!(alignment & 7)) launch_memref_copy<uint2, Narrow>(copy, count, stream);
  else if (!(alignment & 3)) launch_memref_copy<uint32_t, Narrow>(copy, count, stream);
  else if (!(alignment & 1)) launch_memref_copy<uint16_t, Narrow>(copy, count, stream);
  else launch_memref_copy<unsigned char, Narrow>(copy, count, stream);
}

extern "C" int polygeist_thrust_memref_copy(
    int64_t rank, int64_t element, const int64_t *sizes,
    const int64_t *source_strides, const int64_t *destination_strides,
    const void *source, void *destination) {
  if (rank < 1 || rank > MemrefCopy::max_rank) return 0;
  MemrefCopy copy{};
  copy.element = element;
  copy.source = static_cast<const unsigned char *>(source);
  copy.destination = static_cast<unsigned char *>(destination);
  int64_t count = 1;
  int order[MemrefCopy::max_rank];
  for (int i = 0; i < rank; ++i) {
    if (sizes[i] <= 0 || destination_strides[i] < 0) return 0;
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
  // Traverse contiguous output dimensions first and collapse dimensions whose
  // source AND output strides express the same linear traversal. Source-only
  // broadcasts remain broadcasts; arbitrary permutations/reversals remain valid.
  for (int i = rank; i-- > 0;) {
    int dim = order[i];
    if (sizes[dim] == 1) continue;
    int64_t last = copy.rank - 1;
    if (last >= 0 &&
        (__int128)source_strides[dim] * sizes[dim] == copy.source_strides[last] &&
        (__int128)destination_strides[dim] * sizes[dim] == copy.destination_strides[last]) {
      copy.sizes[last] *= sizes[dim];
      copy.source_strides[last] = source_strides[dim];
      copy.destination_strides[last] = destination_strides[dim];
    } else {
      copy.sizes[copy.rank] = sizes[dim];
      copy.source_strides[copy.rank] = source_strides[dim];
      copy.destination_strides[copy.rank++] = destination_strides[dim];
    }
  }
  bool narrow = count <= INT32_MAX;
  if (narrow) for (int64_t i = 0; i < copy.rank; ++i) {
    copy.reciprocal[i] = copy_reciprocal(copy.sizes[i]);
  }
  auto stream = static_cast<cudaStream_t>(polygeist_cuda_graph_stream());
  bool tiled = narrow ? launch_tiled_memref_copy<true>(copy, count, stream)
                      : launch_tiled_memref_copy<false>(copy, count, stream);
  if (!tiled) {
    if (narrow) launch_aligned_memref_copy<true>(copy, count, stream);
    else launch_aligned_memref_copy<false>(copy, count, stream);
  }
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

// Integer scalar functors from the public ATen CPU operators, instantiated
// through Thrust. C's defined-domain requirements for signed shifts apply.
#define POLYGEIST_INTEGER_TRANSFORM(NAME, EXPRESSION)                        \
  struct Integer_##NAME {                                                  \
    __host__ __device__ int32_t operator()(int32_t x, int32_t y) const {     \
      return EXPRESSION;                                                   \
    }                                                                      \
  };                                                                       \
  extern "C" void polygeist_thrust_##NAME##_i32(                           \
      int32_t n, const int32_t *left, const int32_t *right, int32_t *output) {\
    if (n <= 0 || !left || !right || !output) std::abort();                  \
    auto stream = static_cast<cudaStream_t>(polygeist_cuda_graph_stream()); \
    thrust::transform(thrust::cuda::par_nosync.on(stream), left, left + n,   \
                      right, output, Integer_##NAME{});                   \
    if (cudaGetLastError() != cudaSuccess) std::abort();                    \
  }
POLYGEIST_INTEGER_TRANSFORM(bit_and, x & y)
POLYGEIST_INTEGER_TRANSFORM(bit_or, x | y)
POLYGEIST_INTEGER_TRANSFORM(bit_xor, x ^ y)
POLYGEIST_INTEGER_TRANSFORM(left_shift, x << y)
POLYGEIST_INTEGER_TRANSFORM(right_shift, x >> y)
#undef POLYGEIST_INTEGER_TRANSFORM
struct IntegerBitNot {
  __host__ __device__ int32_t operator()(int32_t x) const { return ~x; }
};
extern "C" void polygeist_thrust_bit_not_i32(int32_t n, const int32_t *input,
                                             int32_t *output) {
  if (n <= 0 || !input || !output) std::abort();
  auto stream = static_cast<cudaStream_t>(polygeist_cuda_graph_stream());
  thrust::transform(thrust::cuda::par_nosync.on(stream), input, input + n,
                    output, IntegerBitNot{});
  if (cudaGetLastError() != cudaSuccess) std::abort();
}
struct IntegerWhere {
  template<class Tuple> __host__ __device__ void operator()(Tuple values) const {
    thrust::get<3>(values) = thrust::get<0>(values) ? thrust::get<1>(values)
                                                  : thrust::get<2>(values);
  }
};
extern "C" void polygeist_thrust_where_i32_f32(
    int32_t n, const int32_t *condition, const float *left, const float *right,
    float *output) {
  if (n <= 0 || !condition || !left || !right || !output) std::abort();
  auto stream = static_cast<cudaStream_t>(polygeist_cuda_graph_stream());
  auto begin = thrust::make_zip_iterator(thrust::make_tuple(condition, left, right, output));
  thrust::for_each(thrust::cuda::par_nosync.on(stream), begin, begin + n, IntegerWhere{});
  if (cudaGetLastError() != cudaSuccess) std::abort();
}

// Fixed ABI instantiations of the public generic copy and replace_if APIs.
extern "C" void polygeist_thrust_copy_u64(int32_t n, const uint64_t *input,
                                         uint64_t *output) {
  if (n <= 0 || !input || !output) std::abort();
  auto stream = static_cast<cudaStream_t>(polygeist_cuda_graph_stream());
  thrust::copy(thrust::cuda::par_nosync.on(stream), input, input + n, output);
  if (cudaGetLastError() != cudaSuccess) std::abort();
}
extern "C" void polygeist_thrust_replace_if_i32_f32(
    int32_t n, float *output, const int32_t *mask, float value) {
  if (n <= 0 || !output || !mask) std::abort();
  auto stream = static_cast<cudaStream_t>(polygeist_cuda_graph_stream());
  thrust::replace_if(thrust::cuda::par_nosync.on(stream), output, output + n,
                     mask, ::cuda::std::identity{}, value);
  if (cudaGetLastError() != cudaSuccess) std::abort();
}

extern "C" void polygeist_thrust_multiplies_f64(int32_t n,const double*x,const double*y,double*out) {
  thrust_arithmetic(n,x,y,out,cuda::std::multiplies<double>{},"multiplies_f64");
}
extern "C" void polygeist_thrust_plus_scalar_right_f64(int32_t n,const double*x,double y,double*out) {
  thrust_arithmetic(n,x,thrust::make_constant_iterator(y),out,cuda::std::plus<double>{},"plus_f64");
}
extern "C" void polygeist_thrust_plus_scalar_right_i32(int32_t n,const int32_t*x,int32_t y,int32_t*out) {
  thrust_arithmetic(n,x,thrust::make_constant_iterator(y),out,cuda::std::plus<uint32_t>{},"plus_i32");
}
extern "C" void polygeist_thrust_modulus_scalar_right_u32_to_i32(int32_t n,const uint32_t*x,uint32_t y,int32_t*out) {
  if (!y) std::abort();
  thrust_arithmetic(n,x,thrust::make_constant_iterator(y),out,cuda::std::modulus<uint32_t>{},"modulus_u32");
}
struct LdexpF32I32 {
  __host__ __device__ float operator()(float x,int32_t exponent)const {return ::ldexpf(x,exponent);}
};
extern "C" void polygeist_thrust_ldexp_f32_i32(int32_t n,const float*input,const int32_t*exponent,float*out) {
  thrust_arithmetic(n,input,exponent,out,LdexpF32I32{},"ldexp_f32_i32");
}

extern "C" void polygeist_thrust_sequence_f32(int32_t n,float init,float step,float*output) {
  if(n<=0 || !output) std::abort();
  auto stream=static_cast<cudaStream_t>(polygeist_cuda_graph_stream());
  thrust::sequence(thrust::cuda::par_nosync.on(stream),output,output+n,init,step);
  auto status=cudaGetLastError();
  if(status!=cudaSuccess){
    std::fprintf(stderr,"polygeist Thrust sequence: %s\n",cudaGetErrorString(status));
    std::abort();
  }
}

struct IntegerClampToI8 {
  int32_t lower,upper;
  __host__ __device__ int8_t operator()(int32_t x)const {
    return static_cast<int8_t>(cuda::minimum<int32_t>{}(upper,cuda::maximum<int32_t>{}(x,lower)));
  }
};
extern "C" void polygeist_thrust_clamp_i32_to_i8(int32_t n,const int32_t*input,int32_t lower,int32_t upper,int8_t*output) {
  if(n<=0 || !input || !output)std::abort();
  auto stream=static_cast<cudaStream_t>(polygeist_cuda_graph_stream());
  thrust::transform(thrust::cuda::par_nosync.on(stream),input,input+n,output,IntegerClampToI8{lower,upper});
  auto status=cudaGetLastError();
  if(status!=cudaSuccess){
    std::fprintf(stderr,"polygeist Thrust integer clamp: %s\n",cudaGetErrorString(status));
    std::abort();
  }
}

extern "C" void polygeist_thrust_index_i32(int32_t n, int32_t *output) {
  if (n <= 0 || !output) std::abort();
  auto stream = static_cast<cudaStream_t>(polygeist_cuda_graph_stream());
  auto input = thrust::make_counting_iterator<int32_t>(0);
  thrust::copy(thrust::cuda::par_nosync.on(stream), input, input + n, output);
  if (cudaGetLastError() != cudaSuccess) std::abort();
}

extern "C" void polygeist_thrust_index_f32(int32_t n, float *output) {
  if (n <= 0 || !output) std::abort();
  auto stream = static_cast<cudaStream_t>(polygeist_cuda_graph_stream());
  auto input = thrust::make_counting_iterator<int32_t>(0);
  thrust::copy(thrust::cuda::par_nosync.on(stream), input, input + n, output);
  if (cudaGetLastError() != cudaSuccess) std::abort();
}

extern "C" void polygeist_thrust_copy_i32(int32_t n, const int32_t *input, int32_t *output) {
  if (n <= 0 || !output || !input) std::abort();
  auto stream = static_cast<cudaStream_t>(polygeist_cuda_graph_stream());
  thrust::copy(thrust::cuda::par_nosync.on(stream), input, input + n, output);
  if (cudaGetLastError() != cudaSuccess) std::abort();
}

extern "C" void polygeist_thrust_copy_u32(int32_t n, const uint32_t *input, uint32_t *output) {
  if (n <= 0 || !output || !input) std::abort();
  auto stream = static_cast<cudaStream_t>(polygeist_cuda_graph_stream());
  thrust::copy(thrust::cuda::par_nosync.on(stream), input, input + n, output);
  if (cudaGetLastError() != cudaSuccess) std::abort();
}

extern "C" void polygeist_thrust_copy_i32_f32(int32_t n, const int32_t *input, float *output) {
  if (n <= 0 || !output || !input) std::abort();
  auto stream = static_cast<cudaStream_t>(polygeist_cuda_graph_stream());
  thrust::copy(thrust::cuda::par_nosync.on(stream), input, input + n, output);
  if (cudaGetLastError() != cudaSuccess) std::abort();
}


extern "C" void polygeist_thrust_minus_u32_to_i32(
    int32_t n,const uint32_t*x,const uint32_t*y,int32_t*output) {
  thrust_arithmetic(n,x,y,output,cuda::std::minus<uint32_t>{},"minus_u32");
}
extern "C" void polygeist_thrust_copy_f32(
    int32_t n,const float*input,float*output) {
  if(n<=0 || !input || !output) std::abort();
  auto stream=static_cast<cudaStream_t>(polygeist_cuda_graph_stream());
  thrust::copy(thrust::cuda::par_nosync.on(stream),input,input+n,output);
  if(cudaGetLastError()!=cudaSuccess)std::abort();
}

extern "C" void polygeist_thrust_copy_window_f32(
    int32_t n,int32_t offset,const float*input,float*output) {
  if(n<=0 || offset<0 || !input || !output)std::abort();
  auto stream=static_cast<cudaStream_t>(polygeist_cuda_graph_stream());
  thrust::copy(thrust::cuda::par_nosync.on(stream),input,input+n,output+offset);
  if(cudaGetLastError()!=cudaSuccess)std::abort();
}
extern "C" void polygeist_thrust_reverse_window_f32(
    int32_t n,int32_t prefix,int32_t gap,float*buffer) {
  if(n<=0 || prefix<0 || gap<0 || !buffer)std::abort();
  auto stream=static_cast<cudaStream_t>(polygeist_cuda_graph_stream());
  auto first=buffer+prefix;auto last=first+n;
  thrust::reverse_copy(thrust::cuda::par_nosync.on(stream),first,last,last+gap);
  if(cudaGetLastError()!=cudaSuccess)std::abort();
}
extern "C" void polygeist_thrust_reverse_negate_window_f32(
    int32_t n,int32_t prefix,int32_t gap,float*buffer) {
  if(n<=0 || prefix<0 || gap<0 || !buffer)std::abort();
  auto stream=static_cast<cudaStream_t>(polygeist_cuda_graph_stream());
  auto first=buffer+prefix;auto last=first+n;
  thrust::transform(thrust::cuda::par_nosync.on(stream),
                    cuda::std::make_reverse_iterator(last),
                    cuda::std::make_reverse_iterator(first),last+gap,
                    cuda::std::negate<float>{});
  if(cudaGetLastError()!=cudaSuccess)std::abort();
}

// Counted scalar coordinates use public unsigned arithmetic at 64-bit width.
template <class Operation>
static void thrust_counting_quotient(int32_t n,uint32_t stride,int32_t* output,Operation op) {
 if(n<=0 || stride==0 || !output)std::abort();
 auto stream=static_cast<cudaStream_t>(polygeist_cuda_graph_stream());
 auto first=thrust::make_counting_iterator<uint64_t>(0);
 thrust::transform(thrust::cuda::par_nosync.on(stream),first,first+n,
   thrust::make_constant_iterator<uint64_t>(stride),output,op);
 if(cudaGetLastError()!=cudaSuccess)std::abort();
}
extern "C" void polygeist_thrust_counting_divide_i32(int32_t n,uint32_t stride,int32_t* output) {
 thrust_counting_quotient(n,stride,output,cuda::std::divides<uint64_t>{});
}
extern "C" void polygeist_thrust_counting_remainder_i32(int32_t n,uint32_t stride,int32_t* output) {
 thrust_counting_quotient(n,stride,output,cuda::std::modulus<uint64_t>{});
}
extern "C" void polygeist_thrust_greater_equal_scalar_i32(int32_t n,const int32_t* x,int32_t y,int32_t* output) {
 if(n<=0 || !x || !output)std::abort();
 auto stream=static_cast<cudaStream_t>(polygeist_cuda_graph_stream());
 thrust::transform(thrust::cuda::par_nosync.on(stream),x,x+n,
   thrust::make_constant_iterator<int32_t>(y),output,cuda::std::greater_equal<int32_t>{});
 if(cudaGetLastError()!=cudaSuccess)std::abort();
}
extern "C" void polygeist_thrust_equal_i32_to_f32(int32_t n,const int32_t* x,const int32_t* y,float* output) {
 if(n<=0 || !x || !y || !output)std::abort();
 auto stream=static_cast<cudaStream_t>(polygeist_cuda_graph_stream());
 thrust::transform(thrust::cuda::par_nosync.on(stream),x,x+n,y,output,cuda::std::equal_to<int32_t>{});
 if(cudaGetLastError()!=cudaSuccess)std::abort();
}

extern "C" void polygeist_thrust_negate_window_f32(
    int32_t n,int32_t offset,const float*input,float*output) {
  if(n<=0 || offset<0 || !input || !output)std::abort();
  auto stream=static_cast<cudaStream_t>(polygeist_cuda_graph_stream());
  thrust::transform(thrust::cuda::par_nosync.on(stream),input,input+n,
                    output+offset,cuda::std::negate<float>{});
  if(cudaGetLastError()!=cudaSuccess)std::abort();
}

extern "C" void polygeist_thrust_equal_i32(int32_t n,const int32_t*x,const int32_t*y,int32_t*output) {
  thrust_arithmetic(n,x,y,output,cuda::std::equal_to<int32_t>{},"thrust_equal_i32");
}

extern "C" void polygeist_thrust_less_i32(int32_t n,const int32_t*x,const int32_t*y,int32_t*output) {
  thrust_arithmetic(n,x,y,output,cuda::std::less<int32_t>{},"thrust_less_i32");
}

extern "C" void polygeist_thrust_less_equal_scalar_i32(int32_t n,const int32_t*x,int32_t y,int32_t*output) {
  thrust_arithmetic(n,x,thrust::make_constant_iterator(y),output,
                    cuda::std::less_equal<int32_t>{},"less_equal_i32");
}

// Public gather over typed flat or multidimensional index-iterator views.
template<int Mode, class Element>
static void thrust_gather_indexed(int64_t n,int64_t columns,int64_t sourceColumns,
                                 const int32_t*indices,const Element*input,Element*output,
                                 const int32_t*secondary=nullptr) {
 if(n<=0 || columns<=0 || sourceColumns<=0 || !indices || !input || !output)std::abort();
 auto stream=static_cast<cudaStream_t>(polygeist_cuda_graph_stream());
 auto first=thrust::make_transform_iterator(thrust::make_counting_iterator<int64_t>(0),
     PolygeistGatherIndex<Mode>{indices,columns,sourceColumns,secondary});
 thrust::gather(thrust::cuda::par_nosync.on(stream),first,first+n,input,output);
 if(cudaGetLastError()!=cudaSuccess)std::abort();
}
extern "C" void polygeist_thrust_gather_f32(int32_t n,int32_t m,const int32_t*index,const float*input,float*output) {
 thrust_gather_indexed<0>(n,1,m,index,input,output);
}
extern "C" void polygeist_thrust_gather_rows_f32(int32_t rows,int32_t cols,int32_t source_rows,const int32_t*index,const float*input,float*output) {
 if(rows<=0 || cols<=0 || source_rows<=0)std::abort();
 thrust_gather_indexed<1>(int64_t(rows)*cols,cols,cols,index,input,output);
}
extern "C" void polygeist_thrust_gather_columns_f32(int32_t rows,int32_t cols,int32_t source_cols,const int32_t*index,const float*input,float*output) {
 if(rows<=0 || cols<=0 || source_cols<=0)std::abort();
 thrust_gather_indexed<2>(int64_t(rows)*cols,cols,source_cols,index,input,output);
}
extern "C" void polygeist_thrust_gather_elements_f32(int32_t rows,int32_t cols,int32_t source_cols,const int32_t*index,const float*input,float*output) {
 if(rows<=0 || cols<=0 || source_cols<=0)std::abort();
 thrust_gather_indexed<3>(int64_t(rows)*cols,cols,source_cols,index,input,output);
}
extern "C" void polygeist_thrust_gather_pairs_f32(int32_t rows,int32_t cols,const int32_t*index,const float*input,float*output) {
 thrust_gather_indexed<4>(rows,cols,cols,index,input,output);
}

extern "C" void polygeist_thrust_plus_u32_to_i32(int32_t n,const uint32_t*x,const uint32_t*y,int32_t*output) {
 thrust_arithmetic(n,x,y,output,cuda::std::plus<uint32_t>{},"plus_u32");
}
extern "C" void polygeist_thrust_multiplies_scalar_right_u32_to_i32(int32_t n,const uint32_t*x,uint32_t y,int32_t*output) {
 thrust_arithmetic(n,x,thrust::make_constant_iterator(y),output,cuda::std::multiplies<uint32_t>{},"multiplies_u32");
}
extern "C" void polygeist_thrust_minimum_scalar_i32(int32_t n,const int32_t*input,int32_t bound,int32_t*output) {
 thrust_arithmetic(n,input,thrust::make_constant_iterator(bound),output,cuda::minimum<int32_t>{},"minimum_i32");
}
extern "C" void polygeist_thrust_maximum_scalar_i32(int32_t n,const int32_t*input,int32_t bound,int32_t*output) {
 thrust_arithmetic(n,input,thrust::make_constant_iterator(bound),output,cuda::maximum<int32_t>{},"maximum_i32");
}
extern "C" void polygeist_thrust_negate_u32_to_i32(int32_t n,const uint32_t*input,int32_t*output) {
 if(n<=0 || !input || !output)std::abort();
 auto stream=static_cast<cudaStream_t>(polygeist_cuda_graph_stream());
 thrust::transform(thrust::cuda::par_nosync.on(stream),input,input+n,output,cuda::std::negate<uint32_t>{});
 if(cudaGetLastError()!=cudaSuccess)std::abort();
}

extern "C" void polygeist_thrust_gather_rows_i32(int32_t rows,int32_t cols,int32_t source_rows,const int32_t*index,const int32_t*input,int32_t*output) {
 if(rows<=0 || cols<=0 || source_rows<=0)std::abort();
 thrust_gather_indexed<1>(int64_t(rows)*cols,cols,cols,index,input,output);
}
extern "C" void polygeist_thrust_gather_coordinates_f32(int32_t n,int32_t rows,int32_t cols,const int32_t*row,const int32_t*column,const float*input,float*output) {
 if(rows<=0 || cols<=0 || !column)std::abort();
 thrust_gather_indexed<5>(n,cols,cols,row,input,output,column);
}
