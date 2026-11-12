// Fixed C ABI for CUB's seeded inclusive scan. Inputs are device pointers;
// the generated typed adapter owns any host/device marshalling.
#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>
#include "polygeist_cuda_graph_rt.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

static void require_cuda(cudaError_t status, const char *operation) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "polygeist CUB %s: %s\n", operation, cudaGetErrorString(status));
    std::abort();
  }
}

template<class Op>
static void cub_scan_inclusive_seed(
    int32_t n, float seed, const float *input, float *output, Op op) {
  cudaStream_t stream = (cudaStream_t)polygeist_cuda_graph_stream();
  int device;
  require_cuda(cudaGetDevice(&device), "device");
  struct Scratch { int device; void *pointer; size_t capacity; Scratch *next; };
  static thread_local Scratch *head = nullptr;
  Scratch *scratch = head;
  while (scratch && scratch->device != device) scratch = scratch->next;
  if (!scratch) { scratch = new Scratch{device, nullptr, 0, head}; head = scratch; }
  size_t bytes = 0;
  require_cuda(cub::DeviceScan::InclusiveScanInit(nullptr, bytes, input, output,
      op, seed, n, stream), "scratch query");
  if (bytes > scratch->capacity) {
    // This is reached only when reserving a larger shape, before replay/capture.
    require_cuda(cudaStreamSynchronize(stream), "scratch synchronization");
    require_cuda(cudaFree(scratch->pointer), "scratch free");
    scratch->pointer = nullptr; scratch->capacity = 0;
    require_cuda(cudaMalloc(&scratch->pointer, bytes), "scratch allocation");
    scratch->capacity = bytes;
  }
  require_cuda(cub::DeviceScan::InclusiveScanInit(scratch->pointer, bytes, input,
      output, op, seed, n, stream), "inclusive scan");
  const char *trace = std::getenv("POLYGEIST_CUB_TRACE");
  if (trace && *trace && std::strcmp(trace, "0"))
    std::fprintf(stderr, "POLYGEIST_CUB_CALL entry=DeviceScanInclusiveScanInit n=%d\n", n);
}

extern "C" void polygeist_cub_scan_inclusive_seed_f32(
    int32_t n, float seed, const float *input, float *output) {
  cub_scan_inclusive_seed(n, seed, input, output, cuda::std::plus<float>{});
}
extern "C" void polygeist_cub_scan_product_seed_f32(
    int32_t n, float seed, const float *input, float *output) {
  cub_scan_inclusive_seed(n, seed, input, output, cuda::std::multiplies<float>{});
}

// Fixed typed ABI for CUB DeviceReduce. The generated adapter marshals buffers.
#include <cub/device/device_reduce.cuh>
#include <cuda_runtime.h>
#include "polygeist_cuda_graph_rt.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

static void reduce_require_cuda(cudaError_t status, const char *operation) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "polygeist CUB %s: %s\n", operation, cudaGetErrorString(status));
    std::abort();
  }
}

template<class T, class Op>
static void cub_reduce(int32_t n, T seed, const T *input, T *output, Op op) {
  cudaStream_t stream = (cudaStream_t)polygeist_cuda_graph_stream();
  int device;
  reduce_require_cuda(cudaGetDevice(&device), "device");
  struct Scratch { int device; void *pointer; size_t capacity; Scratch *next; };
  static thread_local Scratch *head = nullptr;
  Scratch *scratch = head;
  while (scratch && scratch->device != device) scratch = scratch->next;
  if (!scratch) { scratch = new Scratch{device, nullptr, 0, head}; head = scratch; }
  size_t bytes = 0;
  reduce_require_cuda(cub::DeviceReduce::Reduce(nullptr, bytes, input, output,
      n, op, seed, stream), "reduce scratch query");
  if (bytes > scratch->capacity) {
    reduce_require_cuda(cudaStreamSynchronize(stream), "reduce scratch synchronization");
    reduce_require_cuda(cudaFree(scratch->pointer), "reduce scratch free");
    scratch->pointer = nullptr; scratch->capacity = 0;
    reduce_require_cuda(cudaMalloc(&scratch->pointer, bytes), "reduce scratch allocation");
    scratch->capacity = bytes;
  }
  reduce_require_cuda(cub::DeviceReduce::Reduce(scratch->pointer, bytes, input,
      output, n, op, seed, stream), "reduce");
  const char *trace = std::getenv("POLYGEIST_CUB_TRACE");
  if (trace && *trace && std::strcmp(trace, "0"))
    std::fprintf(stderr, "POLYGEIST_CUB_CALL entry=DeviceReduce n=%d bytes=%zu\n", n, sizeof(T));
}

extern "C" void polygeist_cub_reduce_sum_f32(int32_t n, float seed, const float *x, float *y) {
  cub_reduce(n, seed, x, y, cuda::std::plus<float>{});
}
extern "C" void polygeist_cub_reduce_sum_f64(int32_t n, double seed, const double *x, double *y) {
  cub_reduce(n, seed, x, y, cuda::std::plus<double>{});
}
extern "C" void polygeist_cub_reduce_product_f32(int32_t n, float seed, const float *x, float *y) {
  cub_reduce(n, seed, x, y, cuda::std::multiplies<float>{});
}

// CUB exposes device output storage; this C ABI returns the completed scalar.
// No scalar computation is reimplemented here.
template<class T, class Op>
static T cub_reduce_value(int32_t n, T seed, const T *input, Op op) {
  cudaStream_t stream = (cudaStream_t)polygeist_cuda_graph_stream();
  cudaStreamCaptureStatus capture;
  reduce_require_cuda(cudaStreamIsCapturing(stream, &capture), "scalar capture query");
  if (capture != cudaStreamCaptureStatusNone) {
    std::fprintf(stderr, "polygeist CUB scalar return requires a completed device result\n");
    std::abort();
  }
  int device;
  reduce_require_cuda(cudaGetDevice(&device), "scalar result device");
  struct Result { int device; T *pointer; Result *next; };
  static thread_local Result *head = nullptr;
  Result *result = head;
  while (result && result->device != device) result = result->next;
  if (!result) {
    result = new Result{device, nullptr, head};
    reduce_require_cuda(cudaMalloc(&result->pointer, sizeof(T)), "scalar result allocation");
    head = result;
  }
  cub_reduce(n, seed, input, result->pointer, op);
  T value;
  reduce_require_cuda(cudaMemcpyAsync(&value, result->pointer, sizeof(T),
                                     cudaMemcpyDeviceToHost, stream), "scalar result copy");
  reduce_require_cuda(cudaStreamSynchronize(stream), "scalar result completion");
  return value;
}
extern "C" float polygeist_cub_reduce_value_sum_f32(int32_t n, float seed, const float *input) {
  return cub_reduce_value(n, seed, input, cuda::std::plus<float>{});
}
extern "C" double polygeist_cub_reduce_value_sum_f64(int32_t n, double seed, const double *input) {
  return cub_reduce_value(n, seed, input, cuda::std::plus<double>{});
}


// CUB's accumulator type follows the functor's return type. Keep the fixed
// int32 C ABI while invoking the actual public CUDA logical operators.
struct LogicalAndI32 {
  __host__ __device__ int operator()(int a, int b) const {
    return cuda::std::logical_and<int>{}(a, b);
  }
};
struct LogicalOrI32 {
  __host__ __device__ int operator()(int a, int b) const {
    return cuda::std::logical_or<int>{}(a, b);
  }
};
extern "C" int polygeist_cub_reduce_value_and_i32(int n, int seed, const int *input) {
  return cub_reduce_value(n, seed, input, LogicalAndI32{});
}
extern "C" int polygeist_cub_reduce_value_or_i32(int n, int seed, const int *input) {
  return cub_reduce_value(n, seed, input, LogicalOrI32{});
}
extern "C" void polygeist_cub_reduce_and_i32(int n, int seed, const int *input, int *output) {
  cub_reduce(n, seed, input, output, LogicalAndI32{});
}
extern "C" void polygeist_cub_reduce_or_i32(int n, int seed, const int *input, int *output) {
  cub_reduce(n, seed, input, output, LogicalOrI32{});
}

// Equal-length segments use library iterators, with no materialized offsets.
#include <cub/device/device_segmented_reduce.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
struct RowOffset {
  int64_t columns;
  __host__ __device__ int64_t operator()(int64_t row) const {
    return row * columns;
  }
};
template<class T, class Input, class Op>
static void cub_segmented_reduce(int rows, int cols, T seed,
                                 Input input, T *output, Op op) {
  cudaStream_t stream = (cudaStream_t)polygeist_cuda_graph_stream();
  auto offsets = thrust::make_transform_iterator(
      thrust::counting_iterator<int64_t>(0), RowOffset{cols});
  int device;
  reduce_require_cuda(cudaGetDevice(&device), "segmented device");
  struct Scratch { int device; void *pointer; size_t capacity; Scratch *next; };
  static thread_local Scratch *head = nullptr;
  Scratch *scratch = head;
  while (scratch && scratch->device != device) scratch = scratch->next;
  if (!scratch) { scratch = new Scratch{device, nullptr, 0, head}; head = scratch; }
  size_t bytes = 0;
  reduce_require_cuda(cub::DeviceSegmentedReduce::Reduce(nullptr, bytes, input,
      output, rows, offsets, offsets + 1, op, seed, stream), "segmented scratch query");
  if (bytes > scratch->capacity) {
    reduce_require_cuda(cudaStreamSynchronize(stream), "segmented scratch synchronization");
    reduce_require_cuda(cudaFree(scratch->pointer), "segmented scratch free");
    scratch->pointer = nullptr; scratch->capacity = 0;
    reduce_require_cuda(cudaMalloc(&scratch->pointer, bytes), "segmented scratch allocation");
    scratch->capacity = bytes;
  }
  reduce_require_cuda(cub::DeviceSegmentedReduce::Reduce(scratch->pointer,
      bytes, input, output, rows, offsets, offsets + 1, op, seed, stream), "segmented reduce");
  const char *trace = std::getenv("POLYGEIST_CUB_TRACE");
  if (trace && *trace && std::strcmp(trace, "0"))
    std::fprintf(stderr, "POLYGEIST_CUB_CALL entry=DeviceSegmentedReduce rows=%d cols=%d\n", rows, cols);
}
extern "C" void polygeist_cub_segmented_reduce_and_i32(
    int rows, int cols, int seed, const int *input, int *output) {
  cub_segmented_reduce(rows, cols, seed, input, output, LogicalAndI32{});
}
extern "C" void polygeist_cub_segmented_reduce_or_i32(
    int rows, int cols, int seed, const int *input, int *output) {
  cub_segmented_reduce(rows, cols, seed, input, output, LogicalOrI32{});
}
extern "C" void polygeist_cub_segmented_logical_mode_i32(
    int rows, int cols, int all, const int *input, int *output) {
  if (all) polygeist_cub_segmented_reduce_and_i32(rows, cols, all, input, output);
  else polygeist_cub_segmented_reduce_or_i32(rows, cols, all, input, output);
}

struct NonzeroF32 {
  __host__ __device__ int operator()(float value) const {
    return cuda::std::not_equal_to<float>{}(value, 0.0f);
  }
};
extern "C" void polygeist_cub_segmented_count_nonzero_f32(
    int rows, int cols, const float *input, int *output) {
  auto values = thrust::make_transform_iterator(input, NonzeroF32{});
  cub_segmented_reduce(rows, cols, 0, values, output, cuda::std::plus<int>{});
}
extern "C" void polygeist_cub_segmented_sum_f32(int rows,int cols,float seed,const float *input,float *output) {
 cub_segmented_reduce(rows,cols,seed,input,output,cuda::std::plus<float>{});
}
extern "C" void polygeist_cub_segmented_xor_i32(int rows,int cols,int seed,const int *input,int *output) {
 cub_segmented_reduce(rows,cols,seed,input,output,cuda::std::bit_xor<int>{});
}

struct ColumnValueI8I32 {
  const int8_t *input;
  int64_t rows, cols;
  __host__ __device__ int operator()(int64_t index) const {
    const int64_t col = index / rows;
    const int64_t row = index % rows;
    return static_cast<int>(input[row * cols + col]);
  }
};
extern "C" void polygeist_cub_column_sum_i8_i32(
    int rows, int cols, int seed, const int8_t *input, int *output) {
  auto values = thrust::make_transform_iterator(thrust::counting_iterator<int64_t>(0),
                                                ColumnValueI8I32{input, rows, cols});
  cub_segmented_reduce(cols, rows, seed, values, output, cuda::std::plus<int>{});
}

// Public pinned CUB ThreadReduce, one contiguous segment per thread.
// The custom operator disables CUB's SIMD reassociation and forwards actual
// additions to CUDA's public plus operator. Absent padding items are not added.
#include <cub/thread/thread_reduce.cuh>
#include <cuda/std/array>
#include <cuda/std/functional>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
struct OptionalF32 { float value; bool present; };
struct AddPresent {
  __host__ __device__ OptionalF32 operator()(OptionalF32 left,OptionalF32 right) const {
    return right.present ? OptionalF32{cuda::std::plus<float>{}(left.value,right.value),true} : left;
  }
};
struct IdentityF32 { __device__ float operator()(float x) const { return x; } };
struct ScaleF32 { float factor; __device__ float operator()(float x) const { return cuda::std::multiplies<float>{}(factor,x); } };
struct DivideF32 { float factor; __device__ float operator()(float x) const { return cuda::std::divides<float>{}(x,factor); } };
// ATen SharedReduceOps.h NanSumOps<float,float>::reduce: replace only NaNs
// with +0 before the ordered addition. Infinities and signed zeros are kept.
struct NanSumF32 { __device__ float operator()(float x) const { return isnan(x) ? 0.0f : x; } };
template<int Width,class Transform,bool VariableLength=false>
__global__ void ordered_rows(int rows,int cols,float seed,const float *input,float *output,Transform transform,const int *lengths=nullptr) {
  int row=blockIdx.x*blockDim.x+threadIdx.x;
  if(row>=rows)return;
  const int count=VariableLength ? lengths[row] : cols;
  if(VariableLength && count>cols)asm("trap;");
  float result=seed;
  for(int64_t start=0;start<count;start+=Width) {
    cuda::std::array<OptionalF32,Width+1> items;
    items[0]={result,true};
    #pragma unroll
    for(int i=0;i<Width;++i)
      items[i+1]=start+i<count ? OptionalF32{transform(input[(int64_t)row*cols+start+i]),true} : OptionalF32{0.f,false};
    result=cub::ThreadReduce(items,AddPresent{}).value;
  }
  output[row]=result;
}
template<class Transform>
static void dispatch_rows(int rows,int cols,float seed,const float *input,float *output,Transform transform) {
  if(rows<=0||cols<=0||!input||!output)std::abort();
  auto stream=static_cast<cudaStream_t>(polygeist_cuda_graph_stream());
  const int blocks=((int64_t)rows+127)/128;
  if(cols<=4)ordered_rows<4><<<blocks,128,0,stream>>>(rows,cols,seed,input,output,transform);
  else if(cols<=16)ordered_rows<16><<<blocks,128,0,stream>>>(rows,cols,seed,input,output,transform);
  else ordered_rows<64><<<blocks,128,0,stream>>>(rows,cols,seed,input,output,transform);
  cudaError_t e=cudaGetLastError();
  if(e!=cudaSuccess){std::fprintf(stderr,"CUB ordered rows: %s\n",cudaGetErrorString(e));std::abort();}
}
extern "C" void polygeist_ordered_rows_sum_f32(int rows,int cols,float seed,const float *input,float *output) {
 dispatch_rows(rows,cols,seed,input,output,IdentityF32{});
}
extern "C" void polygeist_ordered_rows_scaled_sum_f32(int rows,int cols,float seed,float factor,const float *input,float *output) {
 dispatch_rows(rows,cols,seed,input,output,ScaleF32{factor});
}
extern "C" void polygeist_ordered_rows_divided_sum_f32(int rows,int cols,float seed,float factor,const float *input,float *output) {
 dispatch_rows(rows,cols,seed,input,output,DivideF32{factor});
}
extern "C" void polygeist_ordered_rows_nansum_f32(int rows,int cols,const float *input,float *output) {
 dispatch_rows(rows,cols,0.0f,input,output,NanSumF32{});
}
extern "C" void polygeist_ordered_row_prefix_sum_f32(
    int rows,int cols,const float *input,const int *length,float *output) {
  if(rows<=0||cols<=0||!input||!length||!output)std::abort();
  auto stream=static_cast<cudaStream_t>(polygeist_cuda_graph_stream());
  const int blocks=((int64_t)rows+127)/128;
  ordered_rows<64,IdentityF32,true><<<blocks,128,0,stream>>>(
      rows,cols,0.0f,input,output,IdentityF32{},length);
  require_cuda(cudaGetLastError(),"ordered row prefixes");
}
