/*===- polygeist_v2_rt_cuda.c - imported ABI on CUDA libraries ------------===*
 *
 * CUDA implementation of polygeist_v2_rt.h: thin pass-through wrappers around
 * cuBLAS, cuSOLVER-Dn, and typed cuDNN descriptor APIs named by manifests.
 *
 * Device data handling:
 *   * Device/managed pointers remain resident and use the default stream.
 *     With tracing disabled these calls do not add synchronization; callers
 *     synchronize before reading results or measuring completed execution.
 *   * The host footprints of array
 *     operands are merged into disjoint intervals (so operands that alias in
 *     host memory -- e.g. an in-place trmm with C == B -- alias on the device
 *     too), each interval is copied host->device with cudaMemcpy, the library
 *     routine runs on the legacy default stream, and every interval that
 *     contains a written operand is copied back device->host before return.
 *     Untouched gaps inside an interval (rows beyond the view when ld > m)
 *     round-trip unchanged.
 *   * Device buffers are grow-only per-slot scratch allocations, reused across
 *     calls, so steady-state calls do no cudaMalloc.
 *   * cuBLAS uses host pointer mode: alpha/beta/result live on the host.
 *   * The CUDA context, cuBLAS handle and cuSOLVER handle are created in a
 *     constructor so the one-off initialisation is outside timed regions.
 *
 * Host pointers still incur staged copies on each call. Resident pointers
 * bypass that staging; this does not migrate or schedule application buffers
 * automatically. Handles, staging slots, and descriptor caches retain the
 * runtime's existing single-host-thread execution contract.
 *
 * Library-ABI conventions handled generically by the generated code:
 *   cuSOLVER legacy API: `<call>_bufferSize` workspace query, device
 *   workspace, device status word (`devInfo`) checked after the call.
 *   cuDNN descriptor calls: typed manifest bindings checked against vendor
 *   headers, contiguous FP32 row descriptors, nonoverlapping operands.
 *
 * The wrappers between the GENERATED markers are written by
 * scripts/correctness/lib_v2/gen_v2_runtime.py; edit the generator, not them.
 *===----------------------------------------------------------------------===*/
#include "polygeist_v2_rt.h"
#include "polygeist_cublas_rt.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cudnn.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- error handling ----------------------------------------------------- */

static void polygeist_v2_fatal(const char *what, const char *detail) {
  fprintf(stderr, "polygeist_v2 %s: %s\n", what, detail);
  abort();
}

#define POLYGEIST_V2_CUDA(expr)                                                \
  do {                                                                         \
    cudaError_t polygeist_v2_e = (expr);                                       \
    if (polygeist_v2_e != cudaSuccess)                                         \
      polygeist_v2_fatal(#expr, cudaGetErrorString(polygeist_v2_e));           \
  } while (0)

#define POLYGEIST_V2_CUBLAS(expr, name)                                        \
  do {                                                                         \
    cublasStatus_t polygeist_v2_s = (expr);                                    \
    if (polygeist_v2_s != CUBLAS_STATUS_SUCCESS) {                             \
      char polygeist_v2_buf[64];                                               \
      snprintf(polygeist_v2_buf, sizeof polygeist_v2_buf, "cublas status %d",  \
               (int)polygeist_v2_s);                                           \
      polygeist_v2_fatal(name, polygeist_v2_buf);                              \
    }                                                                          \
  } while (0)

#define POLYGEIST_V2_CUSOLVER(expr, name)                                      \
  do {                                                                         \
    cusolverStatus_t polygeist_v2_s = (expr);                                  \
    if (polygeist_v2_s != CUSOLVER_STATUS_SUCCESS) {                           \
      char polygeist_v2_buf[64];                                               \
      snprintf(polygeist_v2_buf, sizeof polygeist_v2_buf,                      \
               "cusolver status %d", (int)polygeist_v2_s);                     \
      polygeist_v2_fatal(name, polygeist_v2_buf);                              \
    }                                                                          \
  } while (0)

/* ---- handles ------------------------------------------------------------ */

static cublasHandle_t polygeist_v2_cublas_h;
static cusolverDnHandle_t polygeist_v2_cusolver_h;

__attribute__((constructor)) static void polygeist_v2_init(void) {
  POLYGEIST_V2_CUDA(cudaFree(0));
  POLYGEIST_V2_CUBLAS(cublasCreate(&polygeist_v2_cublas_h), "cublasCreate");
  POLYGEIST_V2_CUSOLVER(cusolverDnCreate(&polygeist_v2_cusolver_h),
                        "cusolverDnCreate");
}

static cublasHandle_t polygeist_v2_cublas(void) { return polygeist_v2_cublas_h; }
static cusolverDnHandle_t polygeist_v2_cusolver(void) {
  return polygeist_v2_cusolver_h;
}

#define POLYGEIST_V2_CUDNN(expr)                                              \
  do {                                                                       \
    cudnnStatus_t s = (expr);                                                 \
    if (s != CUDNN_STATUS_SUCCESS)                                            \
      polygeist_v2_fatal(#expr, cudnnGetErrorString(s));                        \
  } while (0)

static cudnnHandle_t polygeist_v2_cudnn(void) {
  static cudnnHandle_t handle;
  if (!handle)
    POLYGEIST_V2_CUDNN(cudnnCreate(&handle));
  return handle;
}

/* Shape-keyed descriptors never mutate while a submitted call can use them. */
static cudnnTensorDescriptor_t polygeist_v2_tensor4d(
    cudnnTensorFormat_t format, cudnnDataType_t type, int n, int c, int h, int w) {
  struct entry {
    cudnnTensorDescriptor_t descriptor;
    int format, type, n, c, h, w;
    struct entry *next;
  };
  static struct entry *cache;
  for (struct entry *p = cache; p; p = p->next)
    if (p->format == (int)format && p->type == (int)type &&
        p->n == n && p->c == c && p->h == h && p->w == w)
      return p->descriptor;
  struct entry *p = malloc(sizeof *p);
  if (!p)
    polygeist_v2_fatal("tensor descriptor", "allocation failed");
  POLYGEIST_V2_CUDNN(cudnnCreateTensorDescriptor(&p->descriptor));
  POLYGEIST_V2_CUDNN(cudnnSetTensor4dDescriptor(p->descriptor, format, type, n, c, h, w));
  p->format = format; p->type = type;
  p->n = n; p->c = c; p->h = h; p->w = w;
  p->next = cache; cache = p;
  return p->descriptor;
}

static int polygeist_v2_overlaps(const void *a, int64_t an,
                                 const void *b, int64_t bn) {
  uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
  return x <= y ? y - x < (uint64_t)an : x - y < (uint64_t)bn;
}

/* ---- call accounting (POLYGEIST_V2_TRACE) ------------------------------- */

#define POLYGEIST_V2_MAX_CALLS 64

typedef struct {
  const char *call;
  const char *symbol;
  uint64_t count;
  double h2d_ms, lib_ms, d2h_ms;
  uint64_t h2d_bytes, d2h_bytes;
} polygeist_v2_stat_t;

static polygeist_v2_stat_t polygeist_v2_stats[POLYGEIST_V2_MAX_CALLS];
static int polygeist_v2_nstats;
static int polygeist_v2_trace = -1;

static void polygeist_v2_report(void) {
  for (int i = 0; i < polygeist_v2_nstats; ++i) {
    polygeist_v2_stat_t *s = &polygeist_v2_stats[i];
    fprintf(stderr,
            "POLYGEIST_V2_STATS backend=cuda call=%s entry=%s count=%llu "
            "lib_ms=%.6f h2d_ms=%.6f d2h_ms=%.6f h2d_bytes=%llu "
            "d2h_bytes=%llu\n",
            s->call, s->symbol, (unsigned long long)s->count, s->lib_ms,
            s->h2d_ms, s->d2h_ms, (unsigned long long)s->h2d_bytes,
            (unsigned long long)s->d2h_bytes);
  }
}

static int polygeist_v2_tracing(void) {
  if (polygeist_v2_trace < 0) {
    const char *e = getenv("POLYGEIST_V2_TRACE");
    polygeist_v2_trace = e && *e && strcmp(e, "0") != 0;
    if (polygeist_v2_trace)
      atexit(polygeist_v2_report);
  }
  return polygeist_v2_trace;
}

static polygeist_v2_stat_t *polygeist_v2_stat(const char *call,
                                              const char *symbol) {
  for (int i = 0; i < polygeist_v2_nstats; ++i)
    if (strcmp(polygeist_v2_stats[i].call, call) == 0)
      return &polygeist_v2_stats[i];
  if (polygeist_v2_nstats == POLYGEIST_V2_MAX_CALLS)
    polygeist_v2_fatal(call, "too many distinct calls");
  polygeist_v2_stat_t *s = &polygeist_v2_stats[polygeist_v2_nstats++];
  s->call = call;
  s->symbol = symbol;
  return s;
}

uint64_t polygeist_v2_call_count(const char *call) {
  for (int i = 0; i < polygeist_v2_nstats; ++i)
    if (strcmp(polygeist_v2_stats[i].call, call) == 0)
      return polygeist_v2_stats[i].count;
  return 0;
}

/* ---- per-call host<->device mapping ------------------------------------- */

#define POLYGEIST_V2_MAX_OPERANDS 8
#define POLYGEIST_V2_SLOTS (POLYGEIST_V2_MAX_OPERANDS + 1)

static char *polygeist_v2_slot_ptr[POLYGEIST_V2_SLOTS];
static size_t polygeist_v2_slot_cap[POLYGEIST_V2_SLOTS];

static void *polygeist_v2_slot(int slot, size_t bytes) {
  if (bytes == 0)
    bytes = 1;
  if (polygeist_v2_slot_cap[slot] < bytes) {
    if (polygeist_v2_slot_ptr[slot])
      POLYGEIST_V2_CUDA(cudaFree(polygeist_v2_slot_ptr[slot]));
    POLYGEIST_V2_CUDA(
        cudaMalloc((void **)&polygeist_v2_slot_ptr[slot], bytes));
    polygeist_v2_slot_cap[slot] = bytes;
  }
  return polygeist_v2_slot_ptr[slot];
}

typedef struct {
  const char *lo, *hi;
  int written;
  int interval;
  void *resident;
} polygeist_v2_operand_t;

typedef struct {
  const char *lo, *hi;
  int written;
  char *dev;
} polygeist_v2_interval_t;

typedef struct {
  const char *call, *symbol;
  polygeist_v2_operand_t op[POLYGEIST_V2_MAX_OPERANDS];
  int nop;
  polygeist_v2_interval_t iv[POLYGEIST_V2_MAX_OPERANDS];
  int niv;
  int tracing;
  cudaEvent_t ev[4];
} polygeist_v2_map_t;

static void polygeist_v2_map_init(polygeist_v2_map_t *m, const char *call,
                                  const char *symbol) {
  memset(m, 0, sizeof *m);
  m->call = call;
  m->symbol = symbol;
  m->tracing = polygeist_v2_tracing();
  if (m->tracing) {
    for (int i = 0; i < 4; ++i)
      POLYGEIST_V2_CUDA(cudaEventCreate(&m->ev[i]));
    POLYGEIST_V2_CUDA(cudaEventRecord(m->ev[0], 0));
  }
}

static int polygeist_v2_map_add(polygeist_v2_map_t *m, const void *host,
                                int64_t bytes, int written) {
  if (m->nop == POLYGEIST_V2_MAX_OPERANDS)
    polygeist_v2_fatal(m->call, "too many array operands");
  if (bytes < 0)
    polygeist_v2_fatal(m->call, "negative byte extent");
  polygeist_v2_operand_t *o = &m->op[m->nop];
  o->lo = (const char *)host;
  o->hi = o->lo + bytes;
  o->written = written;
  o->interval = -1;
  struct cudaPointerAttributes attrs;
  cudaError_t status = cudaPointerGetAttributes(&attrs, host);
  if (status == cudaSuccess &&
      (attrs.type == cudaMemoryTypeDevice || attrs.type == cudaMemoryTypeManaged))
    o->resident = attrs.devicePointer ? attrs.devicePointer : (void *)host;
  else if (status != cudaSuccess)
    (void)cudaGetLastError();
  return m->nop++;
}

/* Merge overlapping footprints, allocate, copy host->device. */
static void polygeist_v2_map_to_device(polygeist_v2_map_t *m) {
  int order[POLYGEIST_V2_MAX_OPERANDS];
  for (int i = 0; i < m->nop; ++i)
    order[i] = i;
  for (int i = 1; i < m->nop; ++i)
    for (int j = i; j > 0 && m->op[order[j]].lo < m->op[order[j - 1]].lo; --j) {
      int t = order[j];
      order[j] = order[j - 1];
      order[j - 1] = t;
    }
  for (int k = 0; k < m->nop; ++k) {
    polygeist_v2_operand_t *o = &m->op[order[k]];
    if (o->resident)
      continue;
    if (o->hi == o->lo)
      continue; /* empty view: no device storage */
    polygeist_v2_interval_t *last = m->niv ? &m->iv[m->niv - 1] : NULL;
    if (last && o->lo < last->hi) {
      if (o->hi > last->hi)
        last->hi = o->hi;
      last->written |= o->written;
    } else {
      last = &m->iv[m->niv++];
      last->lo = o->lo;
      last->hi = o->hi;
      last->written = o->written;
    }
    o->interval = m->niv - 1;
  }
  polygeist_v2_stat_t *s = polygeist_v2_stat(m->call, m->symbol);
  for (int i = 0; i < m->niv; ++i) {
    size_t n = (size_t)(m->iv[i].hi - m->iv[i].lo);
    m->iv[i].dev = (char *)polygeist_v2_slot(i, n);
    POLYGEIST_V2_CUDA(
        cudaMemcpy(m->iv[i].dev, m->iv[i].lo, n, cudaMemcpyHostToDevice));
    s->h2d_bytes += n;
  }
  if (m->tracing)
    POLYGEIST_V2_CUDA(cudaEventRecord(m->ev[1], 0));
}

static void *polygeist_v2_map_dev(polygeist_v2_map_t *m, int k) {
  polygeist_v2_operand_t *o = &m->op[k];
  if (o->resident)
    return o->resident;
  if (o->interval < 0)
    return NULL;
  polygeist_v2_interval_t *iv = &m->iv[o->interval];
  return iv->dev + (o->lo - iv->lo);
}

/* Copy written intervals back, release events, account. */
static void polygeist_v2_map_to_host(polygeist_v2_map_t *m) {
  if (m->tracing)
    POLYGEIST_V2_CUDA(cudaEventRecord(m->ev[2], 0));
  polygeist_v2_stat_t *s = polygeist_v2_stat(m->call, m->symbol);
  for (int i = 0; i < m->niv; ++i) {
    if (!m->iv[i].written)
      continue;
    size_t n = (size_t)(m->iv[i].hi - m->iv[i].lo);
    POLYGEIST_V2_CUDA(cudaMemcpy((void *)m->iv[i].lo, m->iv[i].dev, n,
                                 cudaMemcpyDeviceToHost));
    s->d2h_bytes += n;
  }
  float h2d = 0, lib = 0, d2h = 0;
  if (m->tracing) {
    POLYGEIST_V2_CUDA(cudaEventRecord(m->ev[3], 0));
    POLYGEIST_V2_CUDA(cudaEventSynchronize(m->ev[3]));
    POLYGEIST_V2_CUDA(cudaEventElapsedTime(&h2d, m->ev[0], m->ev[1]));
    POLYGEIST_V2_CUDA(cudaEventElapsedTime(&lib, m->ev[1], m->ev[2]));
    POLYGEIST_V2_CUDA(cudaEventElapsedTime(&d2h, m->ev[2], m->ev[3]));
    for (int i = 0; i < 4; ++i)
      POLYGEIST_V2_CUDA(cudaEventDestroy(m->ev[i]));
  }
  s->count++;
  s->h2d_ms += h2d;
  s->lib_ms += lib;
  s->d2h_ms += d2h;
  if (polygeist_v2_tracing())
    fprintf(stderr,
            "POLYGEIST_V2_CALL backend=cuda call=%s entry=%s lib_ms=%.6f "
            "h2d_ms=%.6f d2h_ms=%.6f\n",
            m->call, m->symbol, lib, h2d, d2h);
}

/* cuSOLVER workspace and status word (library-ABI convention). */
static void *polygeist_v2_workspace(size_t bytes) {
  return polygeist_v2_slot(POLYGEIST_V2_SLOTS - 1, bytes);
}

static int *polygeist_v2_devinfo_ptr;

static int *polygeist_v2_devinfo(void) {
  if (!polygeist_v2_devinfo_ptr)
    POLYGEIST_V2_CUDA(
        cudaMalloc((void **)&polygeist_v2_devinfo_ptr, sizeof(int)));
  POLYGEIST_V2_CUDA(cudaMemset(polygeist_v2_devinfo_ptr, 0, sizeof(int)));
  return polygeist_v2_devinfo_ptr;
}

static void polygeist_v2_check_devinfo(const char *call, int *dev) {
  int info = 0;
  POLYGEIST_V2_CUDA(cudaMemcpy(&info, dev, sizeof info, cudaMemcpyDeviceToHost));
  if (info != 0)
    fprintf(stderr, "polygeist_v2 %s: devInfo=%d\n", call, info);
}

#define POLYGEIST_V2_MATRIX_CUDA 1
#include "polygeist_v2_matrix_layout.inc"
#include "polygeist_v2_memref_copy.inc"
#undef POLYGEIST_V2_MATRIX_CUDA

/* BEGIN GENERATED WRAPPERS */
extern __typeof__(polygeist_cub_reduce_and_i32) polygeist_cub_reduce_and_i32 __attribute__((weak));
void polygeist_v2_DeviceReduce_cub_reduce_and_i32(int32_t n, int32_t seed, const int *input, int64_t input_bytes, int32_t input_stride, int *output, int64_t output_bytes, int32_t output_stride) {
  if (!polygeist_cub_reduce_and_i32)
    polygeist_v2_fatal("DeviceReduce_cub_reduce_and_i32", "optional vendor entry is not linked");
  if (n < 1 || n > 2147483647)
    polygeist_v2_fatal("DeviceReduce_cub_reduce_and_i32", "integer outside wrapper bounds");
  if (seed < -2147483648 || seed > 2147483647)
    polygeist_v2_fatal("DeviceReduce_cub_reduce_and_i32", "integer outside wrapper bounds");
  int64_t pv2_bytes_input = sizeof(int);
  if ((int64_t)n <= 0 || pv2_bytes_input > INT64_MAX / (int64_t)n)
    polygeist_v2_fatal("DeviceReduce_cub_reduce_and_i32", "array extent overflow");
  pv2_bytes_input *= (int64_t)n;
  if (input_stride != 1 || !input || input_bytes < pv2_bytes_input)
    polygeist_v2_fatal("DeviceReduce_cub_reduce_and_i32", "array layout or capacity mismatch");
  int64_t pv2_bytes_output = sizeof(int);
  if ((int64_t)1 <= 0 || pv2_bytes_output > INT64_MAX / (int64_t)1)
    polygeist_v2_fatal("DeviceReduce_cub_reduce_and_i32", "array extent overflow");
  pv2_bytes_output *= (int64_t)1;
  if (output_stride != 1 || !output || output_bytes < pv2_bytes_output)
    polygeist_v2_fatal("DeviceReduce_cub_reduce_and_i32", "array layout or capacity mismatch");
  if (polygeist_v2_overlaps(input, input_bytes, output, output_bytes))
    polygeist_v2_fatal("DeviceReduce_cub_reduce_and_i32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "DeviceReduce_cub_reduce_and_i32", "polygeist_cub_reduce_and_i32");
  int pv2_k_input = polygeist_v2_map_add(&pv2_map, input, input_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cub_reduce_and_i32(n, seed, (const int *)polygeist_v2_map_dev(&pv2_map, pv2_k_input), (int *)polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

extern __typeof__(polygeist_cub_reduce_or_i32) polygeist_cub_reduce_or_i32 __attribute__((weak));
void polygeist_v2_DeviceReduce_cub_reduce_or_i32(int32_t n, int32_t seed, const int *input, int64_t input_bytes, int32_t input_stride, int *output, int64_t output_bytes, int32_t output_stride) {
  if (!polygeist_cub_reduce_or_i32)
    polygeist_v2_fatal("DeviceReduce_cub_reduce_or_i32", "optional vendor entry is not linked");
  if (n < 1 || n > 2147483647)
    polygeist_v2_fatal("DeviceReduce_cub_reduce_or_i32", "integer outside wrapper bounds");
  if (seed < -2147483648 || seed > 2147483647)
    polygeist_v2_fatal("DeviceReduce_cub_reduce_or_i32", "integer outside wrapper bounds");
  int64_t pv2_bytes_input = sizeof(int);
  if ((int64_t)n <= 0 || pv2_bytes_input > INT64_MAX / (int64_t)n)
    polygeist_v2_fatal("DeviceReduce_cub_reduce_or_i32", "array extent overflow");
  pv2_bytes_input *= (int64_t)n;
  if (input_stride != 1 || !input || input_bytes < pv2_bytes_input)
    polygeist_v2_fatal("DeviceReduce_cub_reduce_or_i32", "array layout or capacity mismatch");
  int64_t pv2_bytes_output = sizeof(int);
  if ((int64_t)1 <= 0 || pv2_bytes_output > INT64_MAX / (int64_t)1)
    polygeist_v2_fatal("DeviceReduce_cub_reduce_or_i32", "array extent overflow");
  pv2_bytes_output *= (int64_t)1;
  if (output_stride != 1 || !output || output_bytes < pv2_bytes_output)
    polygeist_v2_fatal("DeviceReduce_cub_reduce_or_i32", "array layout or capacity mismatch");
  if (polygeist_v2_overlaps(input, input_bytes, output, output_bytes))
    polygeist_v2_fatal("DeviceReduce_cub_reduce_or_i32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "DeviceReduce_cub_reduce_or_i32", "polygeist_cub_reduce_or_i32");
  int pv2_k_input = polygeist_v2_map_add(&pv2_map, input, input_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cub_reduce_or_i32(n, seed, (const int *)polygeist_v2_map_dev(&pv2_map, pv2_k_input), (int *)polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

extern __typeof__(polygeist_cub_reduce_product_f32) polygeist_cub_reduce_product_f32 __attribute__((weak));
void polygeist_v2_DeviceReduce_cub_reduce_product_f32(int32_t n, float seed, const float *input, int64_t input_bytes, int32_t input_stride, float *output, int64_t output_bytes, int32_t output_stride) {
  if (!polygeist_cub_reduce_product_f32)
    polygeist_v2_fatal("DeviceReduce_cub_reduce_product_f32", "optional vendor entry is not linked");
  if (n < 1 || n > 2147483647)
    polygeist_v2_fatal("DeviceReduce_cub_reduce_product_f32", "integer outside wrapper bounds");
  int64_t pv2_bytes_input = sizeof(float);
  if ((int64_t)n <= 0 || pv2_bytes_input > INT64_MAX / (int64_t)n)
    polygeist_v2_fatal("DeviceReduce_cub_reduce_product_f32", "array extent overflow");
  pv2_bytes_input *= (int64_t)n;
  if (input_stride != 1 || !input || input_bytes < pv2_bytes_input)
    polygeist_v2_fatal("DeviceReduce_cub_reduce_product_f32", "array layout or capacity mismatch");
  int64_t pv2_bytes_output = sizeof(float);
  if ((int64_t)1 <= 0 || pv2_bytes_output > INT64_MAX / (int64_t)1)
    polygeist_v2_fatal("DeviceReduce_cub_reduce_product_f32", "array extent overflow");
  pv2_bytes_output *= (int64_t)1;
  if (output_stride != 1 || !output || output_bytes < pv2_bytes_output)
    polygeist_v2_fatal("DeviceReduce_cub_reduce_product_f32", "array layout or capacity mismatch");
  if (polygeist_v2_overlaps(input, input_bytes, output, output_bytes))
    polygeist_v2_fatal("DeviceReduce_cub_reduce_product_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "DeviceReduce_cub_reduce_product_f32", "polygeist_cub_reduce_product_f32");
  int pv2_k_input = polygeist_v2_map_add(&pv2_map, input, input_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cub_reduce_product_f32(n, seed, (const float *)polygeist_v2_map_dev(&pv2_map, pv2_k_input), (float *)polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

extern __typeof__(polygeist_cub_reduce_sum_f32) polygeist_cub_reduce_sum_f32 __attribute__((weak));
void polygeist_v2_DeviceReduce_cub_reduce_sum_f32(int32_t n, float seed, const float *input, int64_t input_bytes, int32_t input_stride, float *output, int64_t output_bytes, int32_t output_stride) {
  if (!polygeist_cub_reduce_sum_f32)
    polygeist_v2_fatal("DeviceReduce_cub_reduce_sum_f32", "optional vendor entry is not linked");
  if (n < 1 || n > 2147483647)
    polygeist_v2_fatal("DeviceReduce_cub_reduce_sum_f32", "integer outside wrapper bounds");
  int64_t pv2_bytes_input = sizeof(float);
  if ((int64_t)n <= 0 || pv2_bytes_input > INT64_MAX / (int64_t)n)
    polygeist_v2_fatal("DeviceReduce_cub_reduce_sum_f32", "array extent overflow");
  pv2_bytes_input *= (int64_t)n;
  if (input_stride != 1 || !input || input_bytes < pv2_bytes_input)
    polygeist_v2_fatal("DeviceReduce_cub_reduce_sum_f32", "array layout or capacity mismatch");
  int64_t pv2_bytes_output = sizeof(float);
  if ((int64_t)1 <= 0 || pv2_bytes_output > INT64_MAX / (int64_t)1)
    polygeist_v2_fatal("DeviceReduce_cub_reduce_sum_f32", "array extent overflow");
  pv2_bytes_output *= (int64_t)1;
  if (output_stride != 1 || !output || output_bytes < pv2_bytes_output)
    polygeist_v2_fatal("DeviceReduce_cub_reduce_sum_f32", "array layout or capacity mismatch");
  if (polygeist_v2_overlaps(input, input_bytes, output, output_bytes))
    polygeist_v2_fatal("DeviceReduce_cub_reduce_sum_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "DeviceReduce_cub_reduce_sum_f32", "polygeist_cub_reduce_sum_f32");
  int pv2_k_input = polygeist_v2_map_add(&pv2_map, input, input_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cub_reduce_sum_f32(n, seed, (const float *)polygeist_v2_map_dev(&pv2_map, pv2_k_input), (float *)polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

extern __typeof__(polygeist_cub_reduce_sum_f64) polygeist_cub_reduce_sum_f64 __attribute__((weak));
void polygeist_v2_DeviceReduce_cub_reduce_sum_f64(int32_t n, double seed, const double *input, int64_t input_bytes, int32_t input_stride, double *output, int64_t output_bytes, int32_t output_stride) {
  if (!polygeist_cub_reduce_sum_f64)
    polygeist_v2_fatal("DeviceReduce_cub_reduce_sum_f64", "optional vendor entry is not linked");
  if (n < 1 || n > 2147483647)
    polygeist_v2_fatal("DeviceReduce_cub_reduce_sum_f64", "integer outside wrapper bounds");
  int64_t pv2_bytes_input = sizeof(double);
  if ((int64_t)n <= 0 || pv2_bytes_input > INT64_MAX / (int64_t)n)
    polygeist_v2_fatal("DeviceReduce_cub_reduce_sum_f64", "array extent overflow");
  pv2_bytes_input *= (int64_t)n;
  if (input_stride != 1 || !input || input_bytes < pv2_bytes_input)
    polygeist_v2_fatal("DeviceReduce_cub_reduce_sum_f64", "array layout or capacity mismatch");
  int64_t pv2_bytes_output = sizeof(double);
  if ((int64_t)1 <= 0 || pv2_bytes_output > INT64_MAX / (int64_t)1)
    polygeist_v2_fatal("DeviceReduce_cub_reduce_sum_f64", "array extent overflow");
  pv2_bytes_output *= (int64_t)1;
  if (output_stride != 1 || !output || output_bytes < pv2_bytes_output)
    polygeist_v2_fatal("DeviceReduce_cub_reduce_sum_f64", "array layout or capacity mismatch");
  if (polygeist_v2_overlaps(input, input_bytes, output, output_bytes))
    polygeist_v2_fatal("DeviceReduce_cub_reduce_sum_f64", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "DeviceReduce_cub_reduce_sum_f64", "polygeist_cub_reduce_sum_f64");
  int pv2_k_input = polygeist_v2_map_add(&pv2_map, input, input_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cub_reduce_sum_f64(n, seed, (const double *)polygeist_v2_map_dev(&pv2_map, pv2_k_input), (double *)polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

extern __typeof__(polygeist_cub_reduce_value_and_i32) polygeist_cub_reduce_value_and_i32 __attribute__((weak));
int polygeist_v2_DeviceReduce_cub_reduce_value_and_i32(int32_t n, int32_t seed, const int *input, int64_t input_bytes, int32_t input_stride) {
  if (!polygeist_cub_reduce_value_and_i32)
    polygeist_v2_fatal("DeviceReduce_cub_reduce_value_and_i32", "optional vendor entry is not linked");
  if (n < 1 || n > 2147483647)
    polygeist_v2_fatal("DeviceReduce_cub_reduce_value_and_i32", "integer outside wrapper bounds");
  if (seed < -2147483648 || seed > 2147483647)
    polygeist_v2_fatal("DeviceReduce_cub_reduce_value_and_i32", "integer outside wrapper bounds");
  int64_t pv2_bytes_input = sizeof(int);
  if ((int64_t)n <= 0 || pv2_bytes_input > INT64_MAX / (int64_t)n)
    polygeist_v2_fatal("DeviceReduce_cub_reduce_value_and_i32", "array extent overflow");
  pv2_bytes_input *= (int64_t)n;
  if (input_stride != 1 || !input || input_bytes < pv2_bytes_input)
    polygeist_v2_fatal("DeviceReduce_cub_reduce_value_and_i32", "array layout or capacity mismatch");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "DeviceReduce_cub_reduce_value_and_i32", "polygeist_cub_reduce_value_and_i32");
  int pv2_k_input = polygeist_v2_map_add(&pv2_map, input, input_bytes, 0);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  int pv2_result = polygeist_cub_reduce_value_and_i32(n, seed, (const int *)polygeist_v2_map_dev(&pv2_map, pv2_k_input));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
  return pv2_result;
}

extern __typeof__(polygeist_cub_reduce_value_or_i32) polygeist_cub_reduce_value_or_i32 __attribute__((weak));
int polygeist_v2_DeviceReduce_cub_reduce_value_or_i32(int32_t n, int32_t seed, const int *input, int64_t input_bytes, int32_t input_stride) {
  if (!polygeist_cub_reduce_value_or_i32)
    polygeist_v2_fatal("DeviceReduce_cub_reduce_value_or_i32", "optional vendor entry is not linked");
  if (n < 1 || n > 2147483647)
    polygeist_v2_fatal("DeviceReduce_cub_reduce_value_or_i32", "integer outside wrapper bounds");
  if (seed < -2147483648 || seed > 2147483647)
    polygeist_v2_fatal("DeviceReduce_cub_reduce_value_or_i32", "integer outside wrapper bounds");
  int64_t pv2_bytes_input = sizeof(int);
  if ((int64_t)n <= 0 || pv2_bytes_input > INT64_MAX / (int64_t)n)
    polygeist_v2_fatal("DeviceReduce_cub_reduce_value_or_i32", "array extent overflow");
  pv2_bytes_input *= (int64_t)n;
  if (input_stride != 1 || !input || input_bytes < pv2_bytes_input)
    polygeist_v2_fatal("DeviceReduce_cub_reduce_value_or_i32", "array layout or capacity mismatch");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "DeviceReduce_cub_reduce_value_or_i32", "polygeist_cub_reduce_value_or_i32");
  int pv2_k_input = polygeist_v2_map_add(&pv2_map, input, input_bytes, 0);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  int pv2_result = polygeist_cub_reduce_value_or_i32(n, seed, (const int *)polygeist_v2_map_dev(&pv2_map, pv2_k_input));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
  return pv2_result;
}

extern __typeof__(polygeist_cub_reduce_value_sum_f32) polygeist_cub_reduce_value_sum_f32 __attribute__((weak));
float polygeist_v2_DeviceReduce_cub_reduce_value_sum_f32(int32_t n, float seed, const float *input, int64_t input_bytes, int32_t input_stride) {
  if (!polygeist_cub_reduce_value_sum_f32)
    polygeist_v2_fatal("DeviceReduce_cub_reduce_value_sum_f32", "optional vendor entry is not linked");
  if (n < 1 || n > 2147483647)
    polygeist_v2_fatal("DeviceReduce_cub_reduce_value_sum_f32", "integer outside wrapper bounds");
  int64_t pv2_bytes_input = sizeof(float);
  if ((int64_t)n <= 0 || pv2_bytes_input > INT64_MAX / (int64_t)n)
    polygeist_v2_fatal("DeviceReduce_cub_reduce_value_sum_f32", "array extent overflow");
  pv2_bytes_input *= (int64_t)n;
  if (input_stride != 1 || !input || input_bytes < pv2_bytes_input)
    polygeist_v2_fatal("DeviceReduce_cub_reduce_value_sum_f32", "array layout or capacity mismatch");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "DeviceReduce_cub_reduce_value_sum_f32", "polygeist_cub_reduce_value_sum_f32");
  int pv2_k_input = polygeist_v2_map_add(&pv2_map, input, input_bytes, 0);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  float pv2_result = polygeist_cub_reduce_value_sum_f32(n, seed, (const float *)polygeist_v2_map_dev(&pv2_map, pv2_k_input));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
  return pv2_result;
}

extern __typeof__(polygeist_cub_reduce_value_sum_f64) polygeist_cub_reduce_value_sum_f64 __attribute__((weak));
double polygeist_v2_DeviceReduce_cub_reduce_value_sum_f64(int32_t n, double seed, const double *input, int64_t input_bytes, int32_t input_stride) {
  if (!polygeist_cub_reduce_value_sum_f64)
    polygeist_v2_fatal("DeviceReduce_cub_reduce_value_sum_f64", "optional vendor entry is not linked");
  if (n < 1 || n > 2147483647)
    polygeist_v2_fatal("DeviceReduce_cub_reduce_value_sum_f64", "integer outside wrapper bounds");
  int64_t pv2_bytes_input = sizeof(double);
  if ((int64_t)n <= 0 || pv2_bytes_input > INT64_MAX / (int64_t)n)
    polygeist_v2_fatal("DeviceReduce_cub_reduce_value_sum_f64", "array extent overflow");
  pv2_bytes_input *= (int64_t)n;
  if (input_stride != 1 || !input || input_bytes < pv2_bytes_input)
    polygeist_v2_fatal("DeviceReduce_cub_reduce_value_sum_f64", "array layout or capacity mismatch");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "DeviceReduce_cub_reduce_value_sum_f64", "polygeist_cub_reduce_value_sum_f64");
  int pv2_k_input = polygeist_v2_map_add(&pv2_map, input, input_bytes, 0);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  double pv2_result = polygeist_cub_reduce_value_sum_f64(n, seed, (const double *)polygeist_v2_map_dev(&pv2_map, pv2_k_input));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
  return pv2_result;
}

extern __typeof__(polygeist_cub_scan_product_seed_f32) polygeist_cub_scan_product_seed_f32 __attribute__((weak));
void polygeist_v2_DeviceScanInclusiveScanInit_cub_inclusive_product_f32(int32_t n, float inclusive, const float *input, int64_t input_bytes, int32_t input_stride, float *output, int64_t output_bytes, int32_t output_stride) {
  if (!polygeist_cub_scan_product_seed_f32)
    polygeist_v2_fatal("DeviceScanInclusiveScanInit_cub_inclusive_product_f32", "optional vendor entry is not linked");
  if (n < 1 || n > 2147483647)
    polygeist_v2_fatal("DeviceScanInclusiveScanInit_cub_inclusive_product_f32", "integer outside wrapper bounds");
  int64_t pv2_bytes_input = sizeof(float);
  if ((int64_t)n <= 0 || pv2_bytes_input > INT64_MAX / (int64_t)n)
    polygeist_v2_fatal("DeviceScanInclusiveScanInit_cub_inclusive_product_f32", "array extent overflow");
  pv2_bytes_input *= (int64_t)n;
  if (input_stride != 1 || !input || input_bytes < pv2_bytes_input)
    polygeist_v2_fatal("DeviceScanInclusiveScanInit_cub_inclusive_product_f32", "array layout or capacity mismatch");
  int64_t pv2_bytes_output = sizeof(float);
  if ((int64_t)n <= 0 || pv2_bytes_output > INT64_MAX / (int64_t)n)
    polygeist_v2_fatal("DeviceScanInclusiveScanInit_cub_inclusive_product_f32", "array extent overflow");
  pv2_bytes_output *= (int64_t)n;
  if (output_stride != 1 || !output || output_bytes < pv2_bytes_output)
    polygeist_v2_fatal("DeviceScanInclusiveScanInit_cub_inclusive_product_f32", "array layout or capacity mismatch");
  if (polygeist_v2_overlaps(input, input_bytes, output, output_bytes))
    polygeist_v2_fatal("DeviceScanInclusiveScanInit_cub_inclusive_product_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "DeviceScanInclusiveScanInit_cub_inclusive_product_f32", "polygeist_cub_scan_product_seed_f32");
  int pv2_k_input = polygeist_v2_map_add(&pv2_map, input, input_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cub_scan_product_seed_f32(n, inclusive, (const float *)polygeist_v2_map_dev(&pv2_map, pv2_k_input), (float *)polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

extern __typeof__(polygeist_cub_scan_inclusive_seed_f32) polygeist_cub_scan_inclusive_seed_f32 __attribute__((weak));
void polygeist_v2_DeviceScanInclusiveScanInit_cub_inclusive_scan_f32(int32_t n, float inclusive, const float *input, int64_t input_bytes, int32_t input_stride, float *output, int64_t output_bytes, int32_t output_stride) {
  if (!polygeist_cub_scan_inclusive_seed_f32)
    polygeist_v2_fatal("DeviceScanInclusiveScanInit_cub_inclusive_scan_f32", "optional vendor entry is not linked");
  if (n < 1 || n > 2147483647)
    polygeist_v2_fatal("DeviceScanInclusiveScanInit_cub_inclusive_scan_f32", "integer outside wrapper bounds");
  int64_t pv2_bytes_input = sizeof(float);
  if ((int64_t)n <= 0 || pv2_bytes_input > INT64_MAX / (int64_t)n)
    polygeist_v2_fatal("DeviceScanInclusiveScanInit_cub_inclusive_scan_f32", "array extent overflow");
  pv2_bytes_input *= (int64_t)n;
  if (input_stride != 1 || !input || input_bytes < pv2_bytes_input)
    polygeist_v2_fatal("DeviceScanInclusiveScanInit_cub_inclusive_scan_f32", "array layout or capacity mismatch");
  int64_t pv2_bytes_output = sizeof(float);
  if ((int64_t)n <= 0 || pv2_bytes_output > INT64_MAX / (int64_t)n)
    polygeist_v2_fatal("DeviceScanInclusiveScanInit_cub_inclusive_scan_f32", "array extent overflow");
  pv2_bytes_output *= (int64_t)n;
  if (output_stride != 1 || !output || output_bytes < pv2_bytes_output)
    polygeist_v2_fatal("DeviceScanInclusiveScanInit_cub_inclusive_scan_f32", "array layout or capacity mismatch");
  if (polygeist_v2_overlaps(input, input_bytes, output, output_bytes))
    polygeist_v2_fatal("DeviceScanInclusiveScanInit_cub_inclusive_scan_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "DeviceScanInclusiveScanInit_cub_inclusive_scan_f32", "polygeist_cub_scan_inclusive_seed_f32");
  int pv2_k_input = polygeist_v2_map_add(&pv2_map, input, input_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cub_scan_inclusive_seed_f32(n, inclusive, (const float *)polygeist_v2_map_dev(&pv2_map, pv2_k_input), (float *)polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

extern __typeof__(polygeist_cub_column_sum_i8_i32) polygeist_cub_column_sum_i8_i32 __attribute__((weak));
void polygeist_v2_DeviceSegmentedReduce_cub_column_sum_i8_i32(int32_t rows, int32_t cols, int32_t seed, const signed char *input, int64_t input_bytes, int32_t input_stride, int *output, int64_t output_bytes, int32_t output_stride) {
  if (!polygeist_cub_column_sum_i8_i32)
    polygeist_v2_fatal("DeviceSegmentedReduce_cub_column_sum_i8_i32", "optional vendor entry is not linked");
  if (rows < 1 || rows > 8388607)
    polygeist_v2_fatal("DeviceSegmentedReduce_cub_column_sum_i8_i32", "integer outside wrapper bounds");
  if (cols < 1 || cols > 2147483647)
    polygeist_v2_fatal("DeviceSegmentedReduce_cub_column_sum_i8_i32", "integer outside wrapper bounds");
  if (seed < -1073741824 || seed > 1073741824)
    polygeist_v2_fatal("DeviceSegmentedReduce_cub_column_sum_i8_i32", "integer outside wrapper bounds");
  int64_t pv2_bytes_input = sizeof(signed char);
  if ((int64_t)rows <= 0 || pv2_bytes_input > INT64_MAX / (int64_t)rows)
    polygeist_v2_fatal("DeviceSegmentedReduce_cub_column_sum_i8_i32", "array extent overflow");
  pv2_bytes_input *= (int64_t)rows;
  if ((int64_t)cols <= 0 || pv2_bytes_input > INT64_MAX / (int64_t)cols)
    polygeist_v2_fatal("DeviceSegmentedReduce_cub_column_sum_i8_i32", "array extent overflow");
  pv2_bytes_input *= (int64_t)cols;
  if (input_stride != (int64_t)cols || !input || input_bytes < pv2_bytes_input)
    polygeist_v2_fatal("DeviceSegmentedReduce_cub_column_sum_i8_i32", "array layout or capacity mismatch");
  int64_t pv2_bytes_output = sizeof(int);
  if ((int64_t)cols <= 0 || pv2_bytes_output > INT64_MAX / (int64_t)cols)
    polygeist_v2_fatal("DeviceSegmentedReduce_cub_column_sum_i8_i32", "array extent overflow");
  pv2_bytes_output *= (int64_t)cols;
  if (output_stride != 1 || !output || output_bytes < pv2_bytes_output)
    polygeist_v2_fatal("DeviceSegmentedReduce_cub_column_sum_i8_i32", "array layout or capacity mismatch");
  if (polygeist_v2_overlaps(input, input_bytes, output, output_bytes))
    polygeist_v2_fatal("DeviceSegmentedReduce_cub_column_sum_i8_i32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "DeviceSegmentedReduce_cub_column_sum_i8_i32", "polygeist_cub_column_sum_i8_i32");
  int pv2_k_input = polygeist_v2_map_add(&pv2_map, input, input_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cub_column_sum_i8_i32(rows, cols, seed, (const signed char *)polygeist_v2_map_dev(&pv2_map, pv2_k_input), (int *)polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

extern __typeof__(polygeist_cub_segmented_count_nonzero_f32) polygeist_cub_segmented_count_nonzero_f32 __attribute__((weak));
void polygeist_v2_DeviceSegmentedReduce_cub_segmented_count_nonzero_f32(int32_t rows, int32_t cols, const float *input, int64_t input_bytes, int32_t input_stride, int *output, int64_t output_bytes, int32_t output_stride) {
  if (!polygeist_cub_segmented_count_nonzero_f32)
    polygeist_v2_fatal("DeviceSegmentedReduce_cub_segmented_count_nonzero_f32", "optional vendor entry is not linked");
  if (rows < 1 || rows > 2147483647)
    polygeist_v2_fatal("DeviceSegmentedReduce_cub_segmented_count_nonzero_f32", "integer outside wrapper bounds");
  if (cols < 1 || cols > 2147483647)
    polygeist_v2_fatal("DeviceSegmentedReduce_cub_segmented_count_nonzero_f32", "integer outside wrapper bounds");
  int64_t pv2_bytes_input = sizeof(float);
  if ((int64_t)rows <= 0 || pv2_bytes_input > INT64_MAX / (int64_t)rows)
    polygeist_v2_fatal("DeviceSegmentedReduce_cub_segmented_count_nonzero_f32", "array extent overflow");
  pv2_bytes_input *= (int64_t)rows;
  if ((int64_t)cols <= 0 || pv2_bytes_input > INT64_MAX / (int64_t)cols)
    polygeist_v2_fatal("DeviceSegmentedReduce_cub_segmented_count_nonzero_f32", "array extent overflow");
  pv2_bytes_input *= (int64_t)cols;
  if (input_stride != (int64_t)cols || !input || input_bytes < pv2_bytes_input)
    polygeist_v2_fatal("DeviceSegmentedReduce_cub_segmented_count_nonzero_f32", "array layout or capacity mismatch");
  int64_t pv2_bytes_output = sizeof(int);
  if ((int64_t)rows <= 0 || pv2_bytes_output > INT64_MAX / (int64_t)rows)
    polygeist_v2_fatal("DeviceSegmentedReduce_cub_segmented_count_nonzero_f32", "array extent overflow");
  pv2_bytes_output *= (int64_t)rows;
  if (output_stride != 1 || !output || output_bytes < pv2_bytes_output)
    polygeist_v2_fatal("DeviceSegmentedReduce_cub_segmented_count_nonzero_f32", "array layout or capacity mismatch");
  if (polygeist_v2_overlaps(input, input_bytes, output, output_bytes))
    polygeist_v2_fatal("DeviceSegmentedReduce_cub_segmented_count_nonzero_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "DeviceSegmentedReduce_cub_segmented_count_nonzero_f32", "polygeist_cub_segmented_count_nonzero_f32");
  int pv2_k_input = polygeist_v2_map_add(&pv2_map, input, input_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cub_segmented_count_nonzero_f32(rows, cols, (const float *)polygeist_v2_map_dev(&pv2_map, pv2_k_input), (int *)polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

extern __typeof__(polygeist_cub_segmented_logical_mode_i32) polygeist_cub_segmented_logical_mode_i32 __attribute__((weak));
void polygeist_v2_DeviceSegmentedReduce_cub_segmented_logical_mode_i32(int32_t rows, int32_t cols, int32_t all, const int *input, int64_t input_bytes, int32_t input_stride, int *output, int64_t output_bytes, int32_t output_stride) {
  if (!polygeist_cub_segmented_logical_mode_i32)
    polygeist_v2_fatal("DeviceSegmentedReduce_cub_segmented_logical_mode_i32", "optional vendor entry is not linked");
  if (rows < 1 || rows > 2147483647)
    polygeist_v2_fatal("DeviceSegmentedReduce_cub_segmented_logical_mode_i32", "integer outside wrapper bounds");
  if (cols < 1 || cols > 2147483647)
    polygeist_v2_fatal("DeviceSegmentedReduce_cub_segmented_logical_mode_i32", "integer outside wrapper bounds");
  if (all < -2147483648 || all > 2147483647)
    polygeist_v2_fatal("DeviceSegmentedReduce_cub_segmented_logical_mode_i32", "integer outside wrapper bounds");
  int64_t pv2_bytes_input = sizeof(int);
  if ((int64_t)rows <= 0 || pv2_bytes_input > INT64_MAX / (int64_t)rows)
    polygeist_v2_fatal("DeviceSegmentedReduce_cub_segmented_logical_mode_i32", "array extent overflow");
  pv2_bytes_input *= (int64_t)rows;
  if ((int64_t)cols <= 0 || pv2_bytes_input > INT64_MAX / (int64_t)cols)
    polygeist_v2_fatal("DeviceSegmentedReduce_cub_segmented_logical_mode_i32", "array extent overflow");
  pv2_bytes_input *= (int64_t)cols;
  if (input_stride != (int64_t)cols || !input || input_bytes < pv2_bytes_input)
    polygeist_v2_fatal("DeviceSegmentedReduce_cub_segmented_logical_mode_i32", "array layout or capacity mismatch");
  int64_t pv2_bytes_output = sizeof(int);
  if ((int64_t)rows <= 0 || pv2_bytes_output > INT64_MAX / (int64_t)rows)
    polygeist_v2_fatal("DeviceSegmentedReduce_cub_segmented_logical_mode_i32", "array extent overflow");
  pv2_bytes_output *= (int64_t)rows;
  if (output_stride != 1 || !output || output_bytes < pv2_bytes_output)
    polygeist_v2_fatal("DeviceSegmentedReduce_cub_segmented_logical_mode_i32", "array layout or capacity mismatch");
  if (polygeist_v2_overlaps(input, input_bytes, output, output_bytes))
    polygeist_v2_fatal("DeviceSegmentedReduce_cub_segmented_logical_mode_i32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "DeviceSegmentedReduce_cub_segmented_logical_mode_i32", "polygeist_cub_segmented_logical_mode_i32");
  int pv2_k_input = polygeist_v2_map_add(&pv2_map, input, input_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cub_segmented_logical_mode_i32(rows, cols, all, (const int *)polygeist_v2_map_dev(&pv2_map, pv2_k_input), (int *)polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

extern __typeof__(polygeist_cub_segmented_reduce_and_i32) polygeist_cub_segmented_reduce_and_i32 __attribute__((weak));
void polygeist_v2_DeviceSegmentedReduce_cub_segmented_reduce_and_i32(int32_t rows, int32_t cols, int32_t seed, const int *input, int64_t input_bytes, int32_t input_stride, int *output, int64_t output_bytes, int32_t output_stride) {
  if (!polygeist_cub_segmented_reduce_and_i32)
    polygeist_v2_fatal("DeviceSegmentedReduce_cub_segmented_reduce_and_i32", "optional vendor entry is not linked");
  if (rows < 1 || rows > 2147483647)
    polygeist_v2_fatal("DeviceSegmentedReduce_cub_segmented_reduce_and_i32", "integer outside wrapper bounds");
  if (cols < 1 || cols > 2147483647)
    polygeist_v2_fatal("DeviceSegmentedReduce_cub_segmented_reduce_and_i32", "integer outside wrapper bounds");
  if (seed < -2147483648 || seed > 2147483647)
    polygeist_v2_fatal("DeviceSegmentedReduce_cub_segmented_reduce_and_i32", "integer outside wrapper bounds");
  int64_t pv2_bytes_input = sizeof(int);
  if ((int64_t)rows <= 0 || pv2_bytes_input > INT64_MAX / (int64_t)rows)
    polygeist_v2_fatal("DeviceSegmentedReduce_cub_segmented_reduce_and_i32", "array extent overflow");
  pv2_bytes_input *= (int64_t)rows;
  if ((int64_t)cols <= 0 || pv2_bytes_input > INT64_MAX / (int64_t)cols)
    polygeist_v2_fatal("DeviceSegmentedReduce_cub_segmented_reduce_and_i32", "array extent overflow");
  pv2_bytes_input *= (int64_t)cols;
  if (input_stride != (int64_t)cols || !input || input_bytes < pv2_bytes_input)
    polygeist_v2_fatal("DeviceSegmentedReduce_cub_segmented_reduce_and_i32", "array layout or capacity mismatch");
  int64_t pv2_bytes_output = sizeof(int);
  if ((int64_t)rows <= 0 || pv2_bytes_output > INT64_MAX / (int64_t)rows)
    polygeist_v2_fatal("DeviceSegmentedReduce_cub_segmented_reduce_and_i32", "array extent overflow");
  pv2_bytes_output *= (int64_t)rows;
  if (output_stride != 1 || !output || output_bytes < pv2_bytes_output)
    polygeist_v2_fatal("DeviceSegmentedReduce_cub_segmented_reduce_and_i32", "array layout or capacity mismatch");
  if (polygeist_v2_overlaps(input, input_bytes, output, output_bytes))
    polygeist_v2_fatal("DeviceSegmentedReduce_cub_segmented_reduce_and_i32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "DeviceSegmentedReduce_cub_segmented_reduce_and_i32", "polygeist_cub_segmented_reduce_and_i32");
  int pv2_k_input = polygeist_v2_map_add(&pv2_map, input, input_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cub_segmented_reduce_and_i32(rows, cols, seed, (const int *)polygeist_v2_map_dev(&pv2_map, pv2_k_input), (int *)polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

extern __typeof__(polygeist_cub_segmented_reduce_or_i32) polygeist_cub_segmented_reduce_or_i32 __attribute__((weak));
void polygeist_v2_DeviceSegmentedReduce_cub_segmented_reduce_or_i32(int32_t rows, int32_t cols, int32_t seed, const int *input, int64_t input_bytes, int32_t input_stride, int *output, int64_t output_bytes, int32_t output_stride) {
  if (!polygeist_cub_segmented_reduce_or_i32)
    polygeist_v2_fatal("DeviceSegmentedReduce_cub_segmented_reduce_or_i32", "optional vendor entry is not linked");
  if (rows < 1 || rows > 2147483647)
    polygeist_v2_fatal("DeviceSegmentedReduce_cub_segmented_reduce_or_i32", "integer outside wrapper bounds");
  if (cols < 1 || cols > 2147483647)
    polygeist_v2_fatal("DeviceSegmentedReduce_cub_segmented_reduce_or_i32", "integer outside wrapper bounds");
  if (seed < -2147483648 || seed > 2147483647)
    polygeist_v2_fatal("DeviceSegmentedReduce_cub_segmented_reduce_or_i32", "integer outside wrapper bounds");
  int64_t pv2_bytes_input = sizeof(int);
  if ((int64_t)rows <= 0 || pv2_bytes_input > INT64_MAX / (int64_t)rows)
    polygeist_v2_fatal("DeviceSegmentedReduce_cub_segmented_reduce_or_i32", "array extent overflow");
  pv2_bytes_input *= (int64_t)rows;
  if ((int64_t)cols <= 0 || pv2_bytes_input > INT64_MAX / (int64_t)cols)
    polygeist_v2_fatal("DeviceSegmentedReduce_cub_segmented_reduce_or_i32", "array extent overflow");
  pv2_bytes_input *= (int64_t)cols;
  if (input_stride != (int64_t)cols || !input || input_bytes < pv2_bytes_input)
    polygeist_v2_fatal("DeviceSegmentedReduce_cub_segmented_reduce_or_i32", "array layout or capacity mismatch");
  int64_t pv2_bytes_output = sizeof(int);
  if ((int64_t)rows <= 0 || pv2_bytes_output > INT64_MAX / (int64_t)rows)
    polygeist_v2_fatal("DeviceSegmentedReduce_cub_segmented_reduce_or_i32", "array extent overflow");
  pv2_bytes_output *= (int64_t)rows;
  if (output_stride != 1 || !output || output_bytes < pv2_bytes_output)
    polygeist_v2_fatal("DeviceSegmentedReduce_cub_segmented_reduce_or_i32", "array layout or capacity mismatch");
  if (polygeist_v2_overlaps(input, input_bytes, output, output_bytes))
    polygeist_v2_fatal("DeviceSegmentedReduce_cub_segmented_reduce_or_i32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "DeviceSegmentedReduce_cub_segmented_reduce_or_i32", "polygeist_cub_segmented_reduce_or_i32");
  int pv2_k_input = polygeist_v2_map_add(&pv2_map, input, input_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cub_segmented_reduce_or_i32(rows, cols, seed, (const int *)polygeist_v2_map_dev(&pv2_map, pv2_k_input), (int *)polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

extern __typeof__(polygeist_cub_segmented_sum_f32) polygeist_cub_segmented_sum_f32 __attribute__((weak));
void polygeist_v2_DeviceSegmentedReduce_cub_segmented_sum_f32(int32_t rows, int32_t cols, float seed, const float *input, int64_t input_bytes, int32_t input_stride, float *output, int64_t output_bytes, int32_t output_stride) {
  if (!polygeist_cub_segmented_sum_f32)
    polygeist_v2_fatal("DeviceSegmentedReduce_cub_segmented_sum_f32", "optional vendor entry is not linked");
  if (rows < 1 || rows > 2147483647)
    polygeist_v2_fatal("DeviceSegmentedReduce_cub_segmented_sum_f32", "integer outside wrapper bounds");
  if (cols < 1 || cols > 2147483647)
    polygeist_v2_fatal("DeviceSegmentedReduce_cub_segmented_sum_f32", "integer outside wrapper bounds");
  int64_t pv2_bytes_input = sizeof(float);
  if ((int64_t)rows <= 0 || pv2_bytes_input > INT64_MAX / (int64_t)rows)
    polygeist_v2_fatal("DeviceSegmentedReduce_cub_segmented_sum_f32", "array extent overflow");
  pv2_bytes_input *= (int64_t)rows;
  if ((int64_t)cols <= 0 || pv2_bytes_input > INT64_MAX / (int64_t)cols)
    polygeist_v2_fatal("DeviceSegmentedReduce_cub_segmented_sum_f32", "array extent overflow");
  pv2_bytes_input *= (int64_t)cols;
  if (input_stride != (int64_t)cols || !input || input_bytes < pv2_bytes_input)
    polygeist_v2_fatal("DeviceSegmentedReduce_cub_segmented_sum_f32", "array layout or capacity mismatch");
  int64_t pv2_bytes_output = sizeof(float);
  if ((int64_t)rows <= 0 || pv2_bytes_output > INT64_MAX / (int64_t)rows)
    polygeist_v2_fatal("DeviceSegmentedReduce_cub_segmented_sum_f32", "array extent overflow");
  pv2_bytes_output *= (int64_t)rows;
  if (output_stride != 1 || !output || output_bytes < pv2_bytes_output)
    polygeist_v2_fatal("DeviceSegmentedReduce_cub_segmented_sum_f32", "array layout or capacity mismatch");
  if (polygeist_v2_overlaps(input, input_bytes, output, output_bytes))
    polygeist_v2_fatal("DeviceSegmentedReduce_cub_segmented_sum_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "DeviceSegmentedReduce_cub_segmented_sum_f32", "polygeist_cub_segmented_sum_f32");
  int pv2_k_input = polygeist_v2_map_add(&pv2_map, input, input_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cub_segmented_sum_f32(rows, cols, seed, (const float *)polygeist_v2_map_dev(&pv2_map, pv2_k_input), (float *)polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

extern __typeof__(polygeist_cub_segmented_xor_i32) polygeist_cub_segmented_xor_i32 __attribute__((weak));
void polygeist_v2_DeviceSegmentedReduce_cub_segmented_xor_i32(int32_t rows, int32_t cols, int32_t seed, const int *input, int64_t input_bytes, int32_t input_stride, int *output, int64_t output_bytes, int32_t output_stride) {
  if (!polygeist_cub_segmented_xor_i32)
    polygeist_v2_fatal("DeviceSegmentedReduce_cub_segmented_xor_i32", "optional vendor entry is not linked");
  if (rows < 1 || rows > 2147483647)
    polygeist_v2_fatal("DeviceSegmentedReduce_cub_segmented_xor_i32", "integer outside wrapper bounds");
  if (cols < 1 || cols > 2147483647)
    polygeist_v2_fatal("DeviceSegmentedReduce_cub_segmented_xor_i32", "integer outside wrapper bounds");
  if (seed < -2147483648 || seed > 2147483647)
    polygeist_v2_fatal("DeviceSegmentedReduce_cub_segmented_xor_i32", "integer outside wrapper bounds");
  int64_t pv2_bytes_input = sizeof(int);
  if ((int64_t)rows <= 0 || pv2_bytes_input > INT64_MAX / (int64_t)rows)
    polygeist_v2_fatal("DeviceSegmentedReduce_cub_segmented_xor_i32", "array extent overflow");
  pv2_bytes_input *= (int64_t)rows;
  if ((int64_t)cols <= 0 || pv2_bytes_input > INT64_MAX / (int64_t)cols)
    polygeist_v2_fatal("DeviceSegmentedReduce_cub_segmented_xor_i32", "array extent overflow");
  pv2_bytes_input *= (int64_t)cols;
  if (input_stride != (int64_t)cols || !input || input_bytes < pv2_bytes_input)
    polygeist_v2_fatal("DeviceSegmentedReduce_cub_segmented_xor_i32", "array layout or capacity mismatch");
  int64_t pv2_bytes_output = sizeof(int);
  if ((int64_t)rows <= 0 || pv2_bytes_output > INT64_MAX / (int64_t)rows)
    polygeist_v2_fatal("DeviceSegmentedReduce_cub_segmented_xor_i32", "array extent overflow");
  pv2_bytes_output *= (int64_t)rows;
  if (output_stride != 1 || !output || output_bytes < pv2_bytes_output)
    polygeist_v2_fatal("DeviceSegmentedReduce_cub_segmented_xor_i32", "array layout or capacity mismatch");
  if (polygeist_v2_overlaps(input, input_bytes, output, output_bytes))
    polygeist_v2_fatal("DeviceSegmentedReduce_cub_segmented_xor_i32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "DeviceSegmentedReduce_cub_segmented_xor_i32", "polygeist_cub_segmented_xor_i32");
  int pv2_k_input = polygeist_v2_map_add(&pv2_map, input, input_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cub_segmented_xor_i32(rows, cols, seed, (const int *)polygeist_v2_map_dev(&pv2_map, pv2_k_input), (int *)polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

extern __typeof__(polygeist_ordered_rows_nansum_f32) polygeist_ordered_rows_nansum_f32 __attribute__((weak));
void polygeist_v2_ThreadReduce_aten_nansum_ordered_f32(int32_t rows, int32_t cols, const float *input, int64_t input_bytes, int32_t input_stride, float *output, int64_t output_bytes, int32_t output_stride) {
  if (!polygeist_ordered_rows_nansum_f32)
    polygeist_v2_fatal("ThreadReduce_aten_nansum_ordered_f32", "optional vendor entry is not linked");
  if (rows < 1 || rows > 2147483647)
    polygeist_v2_fatal("ThreadReduce_aten_nansum_ordered_f32", "integer outside wrapper bounds");
  if (cols < 1 || cols > 2147483647)
    polygeist_v2_fatal("ThreadReduce_aten_nansum_ordered_f32", "integer outside wrapper bounds");
  int64_t pv2_bytes_input = sizeof(float);
  if ((int64_t)rows <= 0 || pv2_bytes_input > INT64_MAX / (int64_t)rows)
    polygeist_v2_fatal("ThreadReduce_aten_nansum_ordered_f32", "array extent overflow");
  pv2_bytes_input *= (int64_t)rows;
  if ((int64_t)cols <= 0 || pv2_bytes_input > INT64_MAX / (int64_t)cols)
    polygeist_v2_fatal("ThreadReduce_aten_nansum_ordered_f32", "array extent overflow");
  pv2_bytes_input *= (int64_t)cols;
  if (input_stride != (int64_t)cols || !input || input_bytes < pv2_bytes_input)
    polygeist_v2_fatal("ThreadReduce_aten_nansum_ordered_f32", "array layout or capacity mismatch");
  int64_t pv2_bytes_output = sizeof(float);
  if ((int64_t)rows <= 0 || pv2_bytes_output > INT64_MAX / (int64_t)rows)
    polygeist_v2_fatal("ThreadReduce_aten_nansum_ordered_f32", "array extent overflow");
  pv2_bytes_output *= (int64_t)rows;
  if (output_stride != 1 || !output || output_bytes < pv2_bytes_output)
    polygeist_v2_fatal("ThreadReduce_aten_nansum_ordered_f32", "array layout or capacity mismatch");
  if (polygeist_v2_overlaps(input, input_bytes, output, output_bytes))
    polygeist_v2_fatal("ThreadReduce_aten_nansum_ordered_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "ThreadReduce_aten_nansum_ordered_f32", "polygeist_ordered_rows_nansum_f32");
  int pv2_k_input = polygeist_v2_map_add(&pv2_map, input, input_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_ordered_rows_nansum_f32(rows, cols, (const float *)polygeist_v2_map_dev(&pv2_map, pv2_k_input), (float *)polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

extern __typeof__(polygeist_ordered_row_prefix_sum_f32) polygeist_ordered_row_prefix_sum_f32 __attribute__((weak));
void polygeist_v2_ThreadReduce_cub_ordered_row_prefix_sum_f32(int32_t rows, int32_t cols, const float *input, int64_t input_bytes, int32_t input_stride, const int *length, int64_t length_bytes, int32_t length_stride, float *output, int64_t output_bytes, int32_t output_stride) {
  if (!polygeist_ordered_row_prefix_sum_f32)
    polygeist_v2_fatal("ThreadReduce_cub_ordered_row_prefix_sum_f32", "optional vendor entry is not linked");
  if (rows < 1 || rows > 2147483647)
    polygeist_v2_fatal("ThreadReduce_cub_ordered_row_prefix_sum_f32", "integer outside wrapper bounds");
  if (cols < 1 || cols > 2147483647)
    polygeist_v2_fatal("ThreadReduce_cub_ordered_row_prefix_sum_f32", "integer outside wrapper bounds");
  int64_t pv2_bytes_input = sizeof(float);
  if ((int64_t)rows <= 0 || pv2_bytes_input > INT64_MAX / (int64_t)rows)
    polygeist_v2_fatal("ThreadReduce_cub_ordered_row_prefix_sum_f32", "array extent overflow");
  pv2_bytes_input *= (int64_t)rows;
  if ((int64_t)cols <= 0 || pv2_bytes_input > INT64_MAX / (int64_t)cols)
    polygeist_v2_fatal("ThreadReduce_cub_ordered_row_prefix_sum_f32", "array extent overflow");
  pv2_bytes_input *= (int64_t)cols;
  if (input_stride != (int64_t)cols || !input || input_bytes < pv2_bytes_input)
    polygeist_v2_fatal("ThreadReduce_cub_ordered_row_prefix_sum_f32", "array layout or capacity mismatch");
  int64_t pv2_bytes_length = sizeof(int);
  if ((int64_t)rows <= 0 || pv2_bytes_length > INT64_MAX / (int64_t)rows)
    polygeist_v2_fatal("ThreadReduce_cub_ordered_row_prefix_sum_f32", "array extent overflow");
  pv2_bytes_length *= (int64_t)rows;
  if (length_stride != 1 || !length || length_bytes < pv2_bytes_length)
    polygeist_v2_fatal("ThreadReduce_cub_ordered_row_prefix_sum_f32", "array layout or capacity mismatch");
  int64_t pv2_bytes_output = sizeof(float);
  if ((int64_t)rows <= 0 || pv2_bytes_output > INT64_MAX / (int64_t)rows)
    polygeist_v2_fatal("ThreadReduce_cub_ordered_row_prefix_sum_f32", "array extent overflow");
  pv2_bytes_output *= (int64_t)rows;
  if (output_stride != 1 || !output || output_bytes < pv2_bytes_output)
    polygeist_v2_fatal("ThreadReduce_cub_ordered_row_prefix_sum_f32", "array layout or capacity mismatch");
  if (polygeist_v2_overlaps(input, input_bytes, output, output_bytes))
    polygeist_v2_fatal("ThreadReduce_cub_ordered_row_prefix_sum_f32", "unsupported output overlap");
  if (polygeist_v2_overlaps(length, length_bytes, output, output_bytes))
    polygeist_v2_fatal("ThreadReduce_cub_ordered_row_prefix_sum_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "ThreadReduce_cub_ordered_row_prefix_sum_f32", "polygeist_ordered_row_prefix_sum_f32");
  int pv2_k_input = polygeist_v2_map_add(&pv2_map, input, input_bytes, 0);
  int pv2_k_length = polygeist_v2_map_add(&pv2_map, length, length_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_ordered_row_prefix_sum_f32(rows, cols, (const float *)polygeist_v2_map_dev(&pv2_map, pv2_k_input), (const int *)polygeist_v2_map_dev(&pv2_map, pv2_k_length), (float *)polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

extern __typeof__(polygeist_ordered_rows_divided_sum_f32) polygeist_ordered_rows_divided_sum_f32 __attribute__((weak));
void polygeist_v2_ThreadReduce_cub_ordered_rows_divided_sum_f32(int32_t rows, int32_t cols, float seed, float factor, const float *input, int64_t input_bytes, int32_t input_stride, float *output, int64_t output_bytes, int32_t output_stride) {
  if (!polygeist_ordered_rows_divided_sum_f32)
    polygeist_v2_fatal("ThreadReduce_cub_ordered_rows_divided_sum_f32", "optional vendor entry is not linked");
  if (rows < 1 || rows > 2147483647)
    polygeist_v2_fatal("ThreadReduce_cub_ordered_rows_divided_sum_f32", "integer outside wrapper bounds");
  if (cols < 1 || cols > 2147483647)
    polygeist_v2_fatal("ThreadReduce_cub_ordered_rows_divided_sum_f32", "integer outside wrapper bounds");
  int64_t pv2_bytes_input = sizeof(float);
  if ((int64_t)rows <= 0 || pv2_bytes_input > INT64_MAX / (int64_t)rows)
    polygeist_v2_fatal("ThreadReduce_cub_ordered_rows_divided_sum_f32", "array extent overflow");
  pv2_bytes_input *= (int64_t)rows;
  if ((int64_t)cols <= 0 || pv2_bytes_input > INT64_MAX / (int64_t)cols)
    polygeist_v2_fatal("ThreadReduce_cub_ordered_rows_divided_sum_f32", "array extent overflow");
  pv2_bytes_input *= (int64_t)cols;
  if (input_stride != (int64_t)cols || !input || input_bytes < pv2_bytes_input)
    polygeist_v2_fatal("ThreadReduce_cub_ordered_rows_divided_sum_f32", "array layout or capacity mismatch");
  int64_t pv2_bytes_output = sizeof(float);
  if ((int64_t)rows <= 0 || pv2_bytes_output > INT64_MAX / (int64_t)rows)
    polygeist_v2_fatal("ThreadReduce_cub_ordered_rows_divided_sum_f32", "array extent overflow");
  pv2_bytes_output *= (int64_t)rows;
  if (output_stride != 1 || !output || output_bytes < pv2_bytes_output)
    polygeist_v2_fatal("ThreadReduce_cub_ordered_rows_divided_sum_f32", "array layout or capacity mismatch");
  if (polygeist_v2_overlaps(input, input_bytes, output, output_bytes))
    polygeist_v2_fatal("ThreadReduce_cub_ordered_rows_divided_sum_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "ThreadReduce_cub_ordered_rows_divided_sum_f32", "polygeist_ordered_rows_divided_sum_f32");
  int pv2_k_input = polygeist_v2_map_add(&pv2_map, input, input_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_ordered_rows_divided_sum_f32(rows, cols, seed, factor, (const float *)polygeist_v2_map_dev(&pv2_map, pv2_k_input), (float *)polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

extern __typeof__(polygeist_ordered_rows_scaled_sum_f32) polygeist_ordered_rows_scaled_sum_f32 __attribute__((weak));
void polygeist_v2_ThreadReduce_cub_ordered_rows_scaled_sum_f32(int32_t rows, int32_t cols, float seed, float factor, const float *input, int64_t input_bytes, int32_t input_stride, float *output, int64_t output_bytes, int32_t output_stride) {
  if (!polygeist_ordered_rows_scaled_sum_f32)
    polygeist_v2_fatal("ThreadReduce_cub_ordered_rows_scaled_sum_f32", "optional vendor entry is not linked");
  if (rows < 1 || rows > 2147483647)
    polygeist_v2_fatal("ThreadReduce_cub_ordered_rows_scaled_sum_f32", "integer outside wrapper bounds");
  if (cols < 1 || cols > 2147483647)
    polygeist_v2_fatal("ThreadReduce_cub_ordered_rows_scaled_sum_f32", "integer outside wrapper bounds");
  int64_t pv2_bytes_input = sizeof(float);
  if ((int64_t)rows <= 0 || pv2_bytes_input > INT64_MAX / (int64_t)rows)
    polygeist_v2_fatal("ThreadReduce_cub_ordered_rows_scaled_sum_f32", "array extent overflow");
  pv2_bytes_input *= (int64_t)rows;
  if ((int64_t)cols <= 0 || pv2_bytes_input > INT64_MAX / (int64_t)cols)
    polygeist_v2_fatal("ThreadReduce_cub_ordered_rows_scaled_sum_f32", "array extent overflow");
  pv2_bytes_input *= (int64_t)cols;
  if (input_stride != (int64_t)cols || !input || input_bytes < pv2_bytes_input)
    polygeist_v2_fatal("ThreadReduce_cub_ordered_rows_scaled_sum_f32", "array layout or capacity mismatch");
  int64_t pv2_bytes_output = sizeof(float);
  if ((int64_t)rows <= 0 || pv2_bytes_output > INT64_MAX / (int64_t)rows)
    polygeist_v2_fatal("ThreadReduce_cub_ordered_rows_scaled_sum_f32", "array extent overflow");
  pv2_bytes_output *= (int64_t)rows;
  if (output_stride != 1 || !output || output_bytes < pv2_bytes_output)
    polygeist_v2_fatal("ThreadReduce_cub_ordered_rows_scaled_sum_f32", "array layout or capacity mismatch");
  if (polygeist_v2_overlaps(input, input_bytes, output, output_bytes))
    polygeist_v2_fatal("ThreadReduce_cub_ordered_rows_scaled_sum_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "ThreadReduce_cub_ordered_rows_scaled_sum_f32", "polygeist_ordered_rows_scaled_sum_f32");
  int pv2_k_input = polygeist_v2_map_add(&pv2_map, input, input_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_ordered_rows_scaled_sum_f32(rows, cols, seed, factor, (const float *)polygeist_v2_map_dev(&pv2_map, pv2_k_input), (float *)polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

extern __typeof__(polygeist_ordered_rows_sum_f32) polygeist_ordered_rows_sum_f32 __attribute__((weak));
void polygeist_v2_ThreadReduce_cub_ordered_rows_sum_f32(int32_t rows, int32_t cols, float seed, const float *input, int64_t input_bytes, int32_t input_stride, float *output, int64_t output_bytes, int32_t output_stride) {
  if (!polygeist_ordered_rows_sum_f32)
    polygeist_v2_fatal("ThreadReduce_cub_ordered_rows_sum_f32", "optional vendor entry is not linked");
  if (rows < 1 || rows > 2147483647)
    polygeist_v2_fatal("ThreadReduce_cub_ordered_rows_sum_f32", "integer outside wrapper bounds");
  if (cols < 1 || cols > 2147483647)
    polygeist_v2_fatal("ThreadReduce_cub_ordered_rows_sum_f32", "integer outside wrapper bounds");
  int64_t pv2_bytes_input = sizeof(float);
  if ((int64_t)rows <= 0 || pv2_bytes_input > INT64_MAX / (int64_t)rows)
    polygeist_v2_fatal("ThreadReduce_cub_ordered_rows_sum_f32", "array extent overflow");
  pv2_bytes_input *= (int64_t)rows;
  if ((int64_t)cols <= 0 || pv2_bytes_input > INT64_MAX / (int64_t)cols)
    polygeist_v2_fatal("ThreadReduce_cub_ordered_rows_sum_f32", "array extent overflow");
  pv2_bytes_input *= (int64_t)cols;
  if (input_stride != (int64_t)cols || !input || input_bytes < pv2_bytes_input)
    polygeist_v2_fatal("ThreadReduce_cub_ordered_rows_sum_f32", "array layout or capacity mismatch");
  int64_t pv2_bytes_output = sizeof(float);
  if ((int64_t)rows <= 0 || pv2_bytes_output > INT64_MAX / (int64_t)rows)
    polygeist_v2_fatal("ThreadReduce_cub_ordered_rows_sum_f32", "array extent overflow");
  pv2_bytes_output *= (int64_t)rows;
  if (output_stride != 1 || !output || output_bytes < pv2_bytes_output)
    polygeist_v2_fatal("ThreadReduce_cub_ordered_rows_sum_f32", "array layout or capacity mismatch");
  if (polygeist_v2_overlaps(input, input_bytes, output, output_bytes))
    polygeist_v2_fatal("ThreadReduce_cub_ordered_rows_sum_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "ThreadReduce_cub_ordered_rows_sum_f32", "polygeist_ordered_rows_sum_f32");
  int pv2_k_input = polygeist_v2_map_add(&pv2_map, input, input_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_ordered_rows_sum_f32(rows, cols, seed, (const float *)polygeist_v2_map_dev(&pv2_map, pv2_k_input), (float *)polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cublasDaxpy(int32_t n, double alpha, const double *x, int64_t x_bytes, int32_t incx, double *y, int64_t y_bytes, int32_t incy) {
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cublasDaxpy", "cublasDaxpy_v2");
  int pv2_k_x = polygeist_v2_map_add(&pv2_map, x, x_bytes, 0);
  int pv2_k_y = polygeist_v2_map_add(&pv2_map, y, y_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  POLYGEIST_V2_CUBLAS(cublasDaxpy_v2(polygeist_v2_cublas(), n, &alpha, (const double*)polygeist_v2_map_dev(&pv2_map, pv2_k_x), incx, (double*)polygeist_v2_map_dev(&pv2_map, pv2_k_y), incy), "cublasDaxpy_v2");
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cublasDgeam_openblas_daxpby_geam_f64(int32_t n, double alpha, const double *x, int64_t x_bytes, int32_t x_stride, double beta, double *y, int64_t y_bytes, int32_t y_stride) {
  if (n < 1 || n > 2147483647)
    polygeist_v2_fatal("cublasDgeam_openblas_daxpby_geam_f64", "integer outside wrapper bounds");
  int64_t pv2_bytes_x = sizeof(double);
  if ((int64_t)n <= 0 || pv2_bytes_x > INT64_MAX / (int64_t)n)
    polygeist_v2_fatal("cublasDgeam_openblas_daxpby_geam_f64", "array extent overflow");
  pv2_bytes_x *= (int64_t)n;
  if (x_stride != 1 || !x || x_bytes < pv2_bytes_x)
    polygeist_v2_fatal("cublasDgeam_openblas_daxpby_geam_f64", "array layout or capacity mismatch");
  int64_t pv2_bytes_y = sizeof(double);
  if ((int64_t)n <= 0 || pv2_bytes_y > INT64_MAX / (int64_t)n)
    polygeist_v2_fatal("cublasDgeam_openblas_daxpby_geam_f64", "array extent overflow");
  pv2_bytes_y *= (int64_t)n;
  if (y_stride != 1 || !y || y_bytes < pv2_bytes_y)
    polygeist_v2_fatal("cublasDgeam_openblas_daxpby_geam_f64", "array layout or capacity mismatch");
  if (polygeist_v2_overlaps(x, x_bytes, y, y_bytes))
    polygeist_v2_fatal("cublasDgeam_openblas_daxpby_geam_f64", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cublasDgeam_openblas_daxpby_geam_f64", "polygeist_cublas_geam_vector_f64");
  int pv2_k_x = polygeist_v2_map_add(&pv2_map, x, x_bytes, 0);
  int pv2_k_y = polygeist_v2_map_add(&pv2_map, y, y_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cublas_geam_vector_f64(n, alpha, (const double *)polygeist_v2_map_dev(&pv2_map, pv2_k_x), beta, (double *)polygeist_v2_map_dev(&pv2_map, pv2_k_y));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cublasDgemm(int32_t transa, int32_t transb, int32_t m, int32_t n, int32_t k, double alpha, const double *A, int64_t A_bytes, int32_t lda, const double *B, int64_t B_bytes, int32_t ldb, double beta, double *C, int64_t C_bytes, int32_t ldc) {
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cublasDgemm", "cublasDgemm_v2");
  int pv2_k_A = polygeist_v2_map_add(&pv2_map, A, A_bytes, 0);
  int pv2_k_B = polygeist_v2_map_add(&pv2_map, B, B_bytes, 0);
  int pv2_k_C = polygeist_v2_map_add(&pv2_map, C, C_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  POLYGEIST_V2_CUBLAS(cublasDgemm_v2(polygeist_v2_cublas(), (cublasOperation_t)transa, (cublasOperation_t)transb, m, n, k, &alpha, (const double*)polygeist_v2_map_dev(&pv2_map, pv2_k_A), lda, (const double*)polygeist_v2_map_dev(&pv2_map, pv2_k_B), ldb, &beta, (double*)polygeist_v2_map_dev(&pv2_map, pv2_k_C), ldc), "cublasDgemm_v2");
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cublasDgemv(int32_t trans, int32_t m, int32_t n, double alpha, const double *A, int64_t A_bytes, int32_t lda, const double *x, int64_t x_bytes, int32_t incx, double beta, double *y, int64_t y_bytes, int32_t incy) {
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cublasDgemv", "cublasDgemv_v2");
  int pv2_k_A = polygeist_v2_map_add(&pv2_map, A, A_bytes, 0);
  int pv2_k_x = polygeist_v2_map_add(&pv2_map, x, x_bytes, 0);
  int pv2_k_y = polygeist_v2_map_add(&pv2_map, y, y_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  POLYGEIST_V2_CUBLAS(cublasDgemv_v2(polygeist_v2_cublas(), (cublasOperation_t)trans, m, n, &alpha, (const double*)polygeist_v2_map_dev(&pv2_map, pv2_k_A), lda, (const double*)polygeist_v2_map_dev(&pv2_map, pv2_k_x), incx, &beta, (double*)polygeist_v2_map_dev(&pv2_map, pv2_k_y), incy), "cublasDgemv_v2");
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cublasDger(int32_t m, int32_t n, double alpha, const double *x, int64_t x_bytes, int32_t incx, const double *y, int64_t y_bytes, int32_t incy, double *A, int64_t A_bytes, int32_t lda) {
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cublasDger", "cublasDger_v2");
  int pv2_k_x = polygeist_v2_map_add(&pv2_map, x, x_bytes, 0);
  int pv2_k_y = polygeist_v2_map_add(&pv2_map, y, y_bytes, 0);
  int pv2_k_A = polygeist_v2_map_add(&pv2_map, A, A_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  POLYGEIST_V2_CUBLAS(cublasDger_v2(polygeist_v2_cublas(), m, n, &alpha, (const double*)polygeist_v2_map_dev(&pv2_map, pv2_k_x), incx, (const double*)polygeist_v2_map_dev(&pv2_map, pv2_k_y), incy, (double*)polygeist_v2_map_dev(&pv2_map, pv2_k_A), lda), "cublasDger_v2");
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cublasDscal(int32_t n, double alpha, double *x, int64_t x_bytes, int32_t incx) {
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cublasDscal", "cublasDscal_v2");
  int pv2_k_x = polygeist_v2_map_add(&pv2_map, x, x_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  POLYGEIST_V2_CUBLAS(cublasDscal_v2(polygeist_v2_cublas(), n, &alpha, (double*)polygeist_v2_map_dev(&pv2_map, pv2_k_x), incx), "cublasDscal_v2");
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cublasDsymm(int32_t side, int32_t uplo, int32_t m, int32_t n, double alpha, const double *A, int64_t A_bytes, int32_t lda, const double *B, int64_t B_bytes, int32_t ldb, double beta, double *C, int64_t C_bytes, int32_t ldc) {
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cublasDsymm", "cublasDsymm_v2");
  int pv2_k_A = polygeist_v2_map_add(&pv2_map, A, A_bytes, 0);
  int pv2_k_B = polygeist_v2_map_add(&pv2_map, B, B_bytes, 0);
  int pv2_k_C = polygeist_v2_map_add(&pv2_map, C, C_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  POLYGEIST_V2_CUBLAS(cublasDsymm_v2(polygeist_v2_cublas(), (cublasSideMode_t)side, (cublasFillMode_t)uplo, m, n, &alpha, (const double*)polygeist_v2_map_dev(&pv2_map, pv2_k_A), lda, (const double*)polygeist_v2_map_dev(&pv2_map, pv2_k_B), ldb, &beta, (double*)polygeist_v2_map_dev(&pv2_map, pv2_k_C), ldc), "cublasDsymm_v2");
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cublasDsyr2k(int32_t uplo, int32_t trans, int32_t n, int32_t k, double alpha, const double *A, int64_t A_bytes, int32_t lda, const double *B, int64_t B_bytes, int32_t ldb, double beta, double *C, int64_t C_bytes, int32_t ldc) {
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cublasDsyr2k", "cublasDsyr2k_v2");
  int pv2_k_A = polygeist_v2_map_add(&pv2_map, A, A_bytes, 0);
  int pv2_k_B = polygeist_v2_map_add(&pv2_map, B, B_bytes, 0);
  int pv2_k_C = polygeist_v2_map_add(&pv2_map, C, C_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  POLYGEIST_V2_CUBLAS(cublasDsyr2k_v2(polygeist_v2_cublas(), (cublasFillMode_t)uplo, (cublasOperation_t)trans, n, k, &alpha, (const double*)polygeist_v2_map_dev(&pv2_map, pv2_k_A), lda, (const double*)polygeist_v2_map_dev(&pv2_map, pv2_k_B), ldb, &beta, (double*)polygeist_v2_map_dev(&pv2_map, pv2_k_C), ldc), "cublasDsyr2k_v2");
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cublasDsyrk(int32_t uplo, int32_t trans, int32_t n, int32_t k, double alpha, const double *A, int64_t A_bytes, int32_t lda, double beta, double *C, int64_t C_bytes, int32_t ldc) {
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cublasDsyrk", "cublasDsyrk_v2");
  int pv2_k_A = polygeist_v2_map_add(&pv2_map, A, A_bytes, 0);
  int pv2_k_C = polygeist_v2_map_add(&pv2_map, C, C_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  POLYGEIST_V2_CUBLAS(cublasDsyrk_v2(polygeist_v2_cublas(), (cublasFillMode_t)uplo, (cublasOperation_t)trans, n, k, &alpha, (const double*)polygeist_v2_map_dev(&pv2_map, pv2_k_A), lda, &beta, (double*)polygeist_v2_map_dev(&pv2_map, pv2_k_C), ldc), "cublasDsyrk_v2");
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cublasDtrmm(int32_t side, int32_t uplo, int32_t trans, int32_t diag, int32_t m, int32_t n, double alpha, const double *A, int64_t A_bytes, int32_t lda, double *B, int64_t B_bytes, int32_t ldb, double *C, int64_t C_bytes, int32_t ldc) {
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cublasDtrmm", "cublasDtrmm_v2");
  int pv2_k_A = polygeist_v2_map_add(&pv2_map, A, A_bytes, 0);
  int pv2_k_B = polygeist_v2_map_add(&pv2_map, B, B_bytes, 1);
  int pv2_k_C = polygeist_v2_map_add(&pv2_map, C, C_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  POLYGEIST_V2_CUBLAS(cublasDtrmm_v2(polygeist_v2_cublas(), (cublasSideMode_t)side, (cublasFillMode_t)uplo, (cublasOperation_t)trans, (cublasDiagType_t)diag, m, n, &alpha, (const double*)polygeist_v2_map_dev(&pv2_map, pv2_k_A), lda, (const double*)polygeist_v2_map_dev(&pv2_map, pv2_k_B), ldb, (double*)polygeist_v2_map_dev(&pv2_map, pv2_k_C), ldc), "cublasDtrmm_v2");
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cublasDtrsv(int32_t uplo, int32_t trans, int32_t diag, int32_t n, const double *A, int64_t A_bytes, int32_t lda, double *x, int64_t x_bytes, int32_t incx) {
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cublasDtrsv", "cublasDtrsv_v2");
  int pv2_k_A = polygeist_v2_map_add(&pv2_map, A, A_bytes, 0);
  int pv2_k_x = polygeist_v2_map_add(&pv2_map, x, x_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  POLYGEIST_V2_CUBLAS(cublasDtrsv_v2(polygeist_v2_cublas(), (cublasFillMode_t)uplo, (cublasOperation_t)trans, (cublasDiagType_t)diag, n, (const double*)polygeist_v2_map_dev(&pv2_map, pv2_k_A), lda, (double*)polygeist_v2_map_dev(&pv2_map, pv2_k_x), incx), "cublasDtrsv_v2");
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cublasGemmEx_cutlass_gemm_i8_i32(int32_t m, int32_t n, int32_t k, const signed char *a, int64_t a_bytes, int32_t a_stride, const signed char *b, int64_t b_bytes, int32_t b_stride, int *out, int64_t out_bytes, int32_t out_stride) {
  if (m < 1 || m > 2147483647)
    polygeist_v2_fatal("cublasGemmEx_cutlass_gemm_i8_i32", "integer outside wrapper bounds");
  if (n < 4 || n > 2147483644)
    polygeist_v2_fatal("cublasGemmEx_cutlass_gemm_i8_i32", "integer outside wrapper bounds");
  if (n % 4 != 0)
    polygeist_v2_fatal("cublasGemmEx_cutlass_gemm_i8_i32", "integer violates wrapper divisibility");
  if (k < 4 || k > 131068)
    polygeist_v2_fatal("cublasGemmEx_cutlass_gemm_i8_i32", "integer outside wrapper bounds");
  if (k % 4 != 0)
    polygeist_v2_fatal("cublasGemmEx_cutlass_gemm_i8_i32", "integer violates wrapper divisibility");
  int64_t pv2_bytes_a = sizeof(signed char);
  if ((int64_t)m <= 0 || pv2_bytes_a > INT64_MAX / (int64_t)m)
    polygeist_v2_fatal("cublasGemmEx_cutlass_gemm_i8_i32", "array extent overflow");
  pv2_bytes_a *= (int64_t)m;
  if ((int64_t)k <= 0 || pv2_bytes_a > INT64_MAX / (int64_t)k)
    polygeist_v2_fatal("cublasGemmEx_cutlass_gemm_i8_i32", "array extent overflow");
  pv2_bytes_a *= (int64_t)k;
  if (a_stride != (int64_t)k || !a || a_bytes < pv2_bytes_a)
    polygeist_v2_fatal("cublasGemmEx_cutlass_gemm_i8_i32", "array layout or capacity mismatch");
  int64_t pv2_bytes_b = sizeof(signed char);
  if ((int64_t)k <= 0 || pv2_bytes_b > INT64_MAX / (int64_t)k)
    polygeist_v2_fatal("cublasGemmEx_cutlass_gemm_i8_i32", "array extent overflow");
  pv2_bytes_b *= (int64_t)k;
  if ((int64_t)n <= 0 || pv2_bytes_b > INT64_MAX / (int64_t)n)
    polygeist_v2_fatal("cublasGemmEx_cutlass_gemm_i8_i32", "array extent overflow");
  pv2_bytes_b *= (int64_t)n;
  if (b_stride != (int64_t)n || !b || b_bytes < pv2_bytes_b)
    polygeist_v2_fatal("cublasGemmEx_cutlass_gemm_i8_i32", "array layout or capacity mismatch");
  int64_t pv2_bytes_out = sizeof(int);
  if ((int64_t)m <= 0 || pv2_bytes_out > INT64_MAX / (int64_t)m)
    polygeist_v2_fatal("cublasGemmEx_cutlass_gemm_i8_i32", "array extent overflow");
  pv2_bytes_out *= (int64_t)m;
  if ((int64_t)n <= 0 || pv2_bytes_out > INT64_MAX / (int64_t)n)
    polygeist_v2_fatal("cublasGemmEx_cutlass_gemm_i8_i32", "array extent overflow");
  pv2_bytes_out *= (int64_t)n;
  if (out_stride != (int64_t)n || !out || out_bytes < pv2_bytes_out)
    polygeist_v2_fatal("cublasGemmEx_cutlass_gemm_i8_i32", "array layout or capacity mismatch");
  if (polygeist_v2_overlaps(a, a_bytes, out, out_bytes))
    polygeist_v2_fatal("cublasGemmEx_cutlass_gemm_i8_i32", "unsupported output overlap");
  if (polygeist_v2_overlaps(b, b_bytes, out, out_bytes))
    polygeist_v2_fatal("cublasGemmEx_cutlass_gemm_i8_i32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cublasGemmEx_cutlass_gemm_i8_i32", "polygeist_cublas_gemmex_i8_i32");
  int pv2_k_a = polygeist_v2_map_add(&pv2_map, a, a_bytes, 0);
  int pv2_k_b = polygeist_v2_map_add(&pv2_map, b, b_bytes, 0);
  int pv2_k_out = polygeist_v2_map_add(&pv2_map, out, out_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cublas_gemmex_i8_i32(m, n, k, (const signed char *)polygeist_v2_map_dev(&pv2_map, pv2_k_a), (const signed char *)polygeist_v2_map_dev(&pv2_map, pv2_k_b), (int *)polygeist_v2_map_dev(&pv2_map, pv2_k_out));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cublasScopy(int32_t n, const float *x, int64_t x_bytes, int32_t incx, float *y, int64_t y_bytes, int32_t incy) {
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cublasScopy", "cublasScopy_v2");
  int pv2_k_x = polygeist_v2_map_add(&pv2_map, x, x_bytes, 0);
  int pv2_k_y = polygeist_v2_map_add(&pv2_map, y, y_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  POLYGEIST_V2_CUBLAS(cublasScopy_v2(polygeist_v2_cublas(), n, (const float*)polygeist_v2_map_dev(&pv2_map, pv2_k_x), incx, (float*)polygeist_v2_map_dev(&pv2_map, pv2_k_y), incy), "cublasScopy_v2");
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cublasSgeam(int32_t transa, int32_t transb, int32_t m, int32_t n, float alpha, const float *A, int64_t A_bytes, int32_t lda, float beta, const float *B, int64_t B_bytes, int32_t ldb, float *C, int64_t C_bytes, int32_t ldc) {
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cublasSgeam", "cublasSgeam");
  int pv2_k_A = polygeist_v2_map_add(&pv2_map, A, A_bytes, 0);
  int pv2_k_B = polygeist_v2_map_add(&pv2_map, B, B_bytes, 0);
  int pv2_k_C = polygeist_v2_map_add(&pv2_map, C, C_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  POLYGEIST_V2_CUBLAS(cublasSgeam(polygeist_v2_cublas(), (cublasOperation_t)transa, (cublasOperation_t)transb, m, n, &alpha, (const float*)polygeist_v2_map_dev(&pv2_map, pv2_k_A), lda, &beta, (const float*)polygeist_v2_map_dev(&pv2_map, pv2_k_B), ldb, (float*)polygeist_v2_map_dev(&pv2_map, pv2_k_C), ldc), "cublasSgeam");
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cublasSgemm(int32_t transa, int32_t transb, int32_t m, int32_t n, int32_t k, float alpha, const float *A, int64_t A_bytes, int32_t lda, const float *B, int64_t B_bytes, int32_t ldb, float beta, float *C, int64_t C_bytes, int32_t ldc) {
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cublasSgemm", "cublasSgemm_v2");
  int pv2_k_A = polygeist_v2_map_add(&pv2_map, A, A_bytes, 0);
  int pv2_k_B = polygeist_v2_map_add(&pv2_map, B, B_bytes, 0);
  int pv2_k_C = polygeist_v2_map_add(&pv2_map, C, C_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  POLYGEIST_V2_CUBLAS(cublasSgemm_v2(polygeist_v2_cublas(), (cublasOperation_t)transa, (cublasOperation_t)transb, m, n, k, &alpha, (const float*)polygeist_v2_map_dev(&pv2_map, pv2_k_A), lda, (const float*)polygeist_v2_map_dev(&pv2_map, pv2_k_B), ldb, &beta, (float*)polygeist_v2_map_dev(&pv2_map, pv2_k_C), ldc), "cublasSgemm_v2");
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cublasSgemv(int32_t trans, int32_t m, int32_t n, float alpha, const float *A, int64_t A_bytes, int32_t lda, const float *x, int64_t x_bytes, int32_t incx, float beta, float *y, int64_t y_bytes, int32_t incy) {
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cublasSgemv", "cublasSgemv_v2");
  int pv2_k_A = polygeist_v2_map_add(&pv2_map, A, A_bytes, 0);
  int pv2_k_x = polygeist_v2_map_add(&pv2_map, x, x_bytes, 0);
  int pv2_k_y = polygeist_v2_map_add(&pv2_map, y, y_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  POLYGEIST_V2_CUBLAS(cublasSgemv_v2(polygeist_v2_cublas(), (cublasOperation_t)trans, m, n, &alpha, (const float*)polygeist_v2_map_dev(&pv2_map, pv2_k_A), lda, (const float*)polygeist_v2_map_dev(&pv2_map, pv2_k_x), incx, &beta, (float*)polygeist_v2_map_dev(&pv2_map, pv2_k_y), incy), "cublasSgemv_v2");
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cublasSsyr2k(int32_t uplo, int32_t trans, int32_t n, int32_t k, float alpha, const float *A, int64_t A_bytes, int32_t lda, const float *B, int64_t B_bytes, int32_t ldb, float beta, float *C, int64_t C_bytes, int32_t ldc) {
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cublasSsyr2k", "cublasSsyr2k_v2");
  int pv2_k_A = polygeist_v2_map_add(&pv2_map, A, A_bytes, 0);
  int pv2_k_B = polygeist_v2_map_add(&pv2_map, B, B_bytes, 0);
  int pv2_k_C = polygeist_v2_map_add(&pv2_map, C, C_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  POLYGEIST_V2_CUBLAS(cublasSsyr2k_v2(polygeist_v2_cublas(), (cublasFillMode_t)uplo, (cublasOperation_t)trans, n, k, &alpha, (const float*)polygeist_v2_map_dev(&pv2_map, pv2_k_A), lda, (const float*)polygeist_v2_map_dev(&pv2_map, pv2_k_B), ldb, &beta, (float*)polygeist_v2_map_dev(&pv2_map, pv2_k_C), ldc), "cublasSsyr2k_v2");
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cudnnBackendExecute_aten_abs_f32(int32_t n, const float *x, int64_t x_bytes, int32_t x_stride, float *output, int64_t output_bytes, int32_t output_stride) {
  if (n <= 0) polygeist_v2_fatal("cudnnBackendExecute_aten_abs_f32", "nonpositive dimension");
  if (x_stride <= 0 || !x || x_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * x_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_abs_f32", "invalid strided FP32 row");
  if (output_stride <= 0 || !output || output_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * output_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_abs_f32", "invalid strided FP32 row");
  if (polygeist_v2_overlaps(x, x_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_abs_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cudnnBackendExecute_aten_abs_f32", "cudnnBackendExecute");
  int pv2_k_x = polygeist_v2_map_add(&pv2_map, x, x_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cudnn_pointwise_graph_f32(n, 150994944LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 1, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, x_stride, 1, 1, 1, output_stride, polygeist_v2_map_dev(&pv2_map, pv2_k_x), NULL, NULL, NULL, polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cudnnBackendExecute_aten_add_f32(int32_t n, const float *x, int64_t x_bytes, int32_t x_stride, const float *y, int64_t y_bytes, int32_t y_stride, float *output, int64_t output_bytes, int32_t output_stride) {
  if (n <= 0) polygeist_v2_fatal("cudnnBackendExecute_aten_add_f32", "nonpositive dimension");
  if (x_stride <= 0 || !x || x_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * x_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_add_f32", "invalid strided FP32 row");
  if (y_stride <= 0 || !y || y_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * y_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_add_f32", "invalid strided FP32 row");
  if (output_stride <= 0 || !output || output_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * output_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_add_f32", "invalid strided FP32 row");
  if (polygeist_v2_overlaps(x, x_bytes, output, output_bytes) && !(x == output && x_bytes == output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_add_f32", "unsupported output overlap");
  if (polygeist_v2_overlaps(y, y_bytes, output, output_bytes) && !(y == output && y_bytes == output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_add_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cudnnBackendExecute_aten_add_f32", "cudnnBackendExecute");
  int pv2_k_x = polygeist_v2_map_add(&pv2_map, x, x_bytes, 0);
  int pv2_k_y = polygeist_v2_map_add(&pv2_map, y, y_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cudnn_pointwise_graph_f32(n, 16777472LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 1, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, x_stride, y_stride, 1, 1, output_stride, polygeist_v2_map_dev(&pv2_map, pv2_k_x), polygeist_v2_map_dev(&pv2_map, pv2_k_y), NULL, NULL, polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cudnnBackendExecute_aten_addcdiv_f32(int32_t n, const float *self_val, int64_t self_val_bytes, int32_t self_val_stride, const float *t1_val, int64_t t1_val_bytes, int32_t t1_val_stride, const float *t2_val, int64_t t2_val_bytes, int32_t t2_val_stride, float *output, int64_t output_bytes, int32_t output_stride, float scalar_val) {
  if (n <= 0) polygeist_v2_fatal("cudnnBackendExecute_aten_addcdiv_f32", "nonpositive dimension");
  if (self_val_stride <= 0 || !self_val || self_val_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * self_val_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_addcdiv_f32", "invalid strided FP32 row");
  if (t1_val_stride <= 0 || !t1_val || t1_val_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * t1_val_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_addcdiv_f32", "invalid strided FP32 row");
  if (t2_val_stride <= 0 || !t2_val || t2_val_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * t2_val_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_addcdiv_f32", "invalid strided FP32 row");
  if (output_stride <= 0 || !output || output_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * output_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_addcdiv_f32", "invalid strided FP32 row");
  if (polygeist_v2_overlaps(self_val, self_val_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_addcdiv_f32", "unsupported output overlap");
  if (polygeist_v2_overlaps(t1_val, t1_val_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_addcdiv_f32", "unsupported output overlap");
  if (polygeist_v2_overlaps(t2_val, t2_val_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_addcdiv_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cudnnBackendExecute_aten_addcdiv_f32", "cudnnBackendExecute");
  int pv2_k_self_val = polygeist_v2_map_add(&pv2_map, self_val, self_val_bytes, 0);
  int pv2_k_t1_val = polygeist_v2_map_add(&pv2_map, t1_val, t1_val_bytes, 0);
  int pv2_k_t2_val = polygeist_v2_map_add(&pv2_map, t2_val, t2_val_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cudnn_pointwise_graph_f32(n, 291610274929312000LL, 16780544LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 3, scalar_val, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, self_val_stride, t1_val_stride, t2_val_stride, 1, output_stride, polygeist_v2_map_dev(&pv2_map, pv2_k_self_val), polygeist_v2_map_dev(&pv2_map, pv2_k_t1_val), polygeist_v2_map_dev(&pv2_map, pv2_k_t2_val), NULL, polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cudnnBackendExecute_aten_addcmul_f32(int32_t n, const float *self_val, int64_t self_val_bytes, int32_t self_val_stride, const float *t1_val, int64_t t1_val_bytes, int32_t t1_val_stride, const float *t2_val, int64_t t2_val_bytes, int32_t t2_val_stride, float *output, int64_t output_bytes, int32_t output_stride, float scalar_val) {
  if (n <= 0) polygeist_v2_fatal("cudnnBackendExecute_aten_addcmul_f32", "nonpositive dimension");
  if (self_val_stride <= 0 || !self_val || self_val_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * self_val_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_addcmul_f32", "invalid strided FP32 row");
  if (t1_val_stride <= 0 || !t1_val || t1_val_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * t1_val_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_addcmul_f32", "invalid strided FP32 row");
  if (t2_val_stride <= 0 || !t2_val || t2_val_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * t2_val_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_addcmul_f32", "invalid strided FP32 row");
  if (output_stride <= 0 || !output || output_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * output_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_addcmul_f32", "invalid strided FP32 row");
  if (polygeist_v2_overlaps(self_val, self_val_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_addcmul_f32", "unsupported output overlap");
  if (polygeist_v2_overlaps(t1_val, t1_val_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_addcmul_f32", "unsupported output overlap");
  if (polygeist_v2_overlaps(t2_val, t2_val_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_addcmul_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cudnnBackendExecute_aten_addcmul_f32", "cudnnBackendExecute");
  int pv2_k_self_val = polygeist_v2_map_add(&pv2_map, self_val, self_val_bytes, 0);
  int pv2_k_t1_val = polygeist_v2_map_add(&pv2_map, t1_val, t1_val_bytes, 0);
  int pv2_k_t2_val = polygeist_v2_map_add(&pv2_map, t2_val, t2_val_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cudnn_pointwise_graph_f32(n, 147495086853456128LL, 16780544LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 3, scalar_val, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, self_val_stride, t1_val_stride, t2_val_stride, 1, output_stride, polygeist_v2_map_dev(&pv2_map, pv2_k_self_val), polygeist_v2_map_dev(&pv2_map, pv2_k_t1_val), polygeist_v2_map_dev(&pv2_map, pv2_k_t2_val), NULL, polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cudnnBackendExecute_aten_addr_elementwise_f32(int32_t n, const float *self_val, int64_t self_val_bytes, int32_t self_val_stride, const float *vec1_val, int64_t vec1_val_bytes, int32_t vec1_val_stride, const float *vec2_val, int64_t vec2_val_bytes, int32_t vec2_val_stride, float beta_val, float alpha_val, float *output, int64_t output_bytes, int32_t output_stride) {
  if (n <= 0) polygeist_v2_fatal("cudnnBackendExecute_aten_addr_elementwise_f32", "nonpositive dimension");
  if (self_val_stride <= 0 || !self_val || self_val_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * self_val_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_addr_elementwise_f32", "invalid strided FP32 row");
  if (vec1_val_stride <= 0 || !vec1_val || vec1_val_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * vec1_val_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_addr_elementwise_f32", "invalid strided FP32 row");
  if (vec2_val_stride <= 0 || !vec2_val || vec2_val_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * vec2_val_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_addr_elementwise_f32", "invalid strided FP32 row");
  if (output_stride <= 0 || !output || output_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * output_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_addr_elementwise_f32", "invalid strided FP32 row");
  if (polygeist_v2_overlaps(self_val, self_val_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_addr_elementwise_f32", "unsupported output overlap");
  if (polygeist_v2_overlaps(vec1_val, vec1_val_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_addr_elementwise_f32", "unsupported output overlap");
  if (polygeist_v2_overlaps(vec2_val, vec2_val_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_addr_elementwise_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cudnnBackendExecute_aten_addr_elementwise_f32", "cudnnBackendExecute");
  int pv2_k_self_val = polygeist_v2_map_add(&pv2_map, self_val, self_val_bytes, 0);
  int pv2_k_vec1_val = polygeist_v2_map_add(&pv2_map, vec1_val, vec1_val_bytes, 0);
  int pv2_k_vec2_val = polygeist_v2_map_add(&pv2_map, vec2_val, vec2_val_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cudnn_pointwise_graph_f32(n, 147493987341893632LL, 145522562993226240LL, 76015835932459264LL, 2094751143732446720LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 8, beta_val, alpha_val, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, vec1_val_stride, vec2_val_stride, self_val_stride, 1, output_stride, polygeist_v2_map_dev(&pv2_map, pv2_k_vec1_val), polygeist_v2_map_dev(&pv2_map, pv2_k_vec2_val), polygeist_v2_map_dev(&pv2_map, pv2_k_self_val), NULL, polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cudnnBackendExecute_aten_atan2_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, const float *b, int64_t b_bytes, int32_t b_stride, float *output, int64_t output_bytes, int32_t output_stride) {
  if (n <= 0) polygeist_v2_fatal("cudnnBackendExecute_aten_atan2_f32", "nonpositive dimension");
  if (a_stride <= 0 || !a || a_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * a_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_atan2_f32", "invalid strided FP32 row");
  if (b_stride <= 0 || !b || b_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * b_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_atan2_f32", "invalid strided FP32 row");
  if (output_stride <= 0 || !output || output_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * output_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_atan2_f32", "invalid strided FP32 row");
  if (polygeist_v2_overlaps(a, a_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_atan2_f32", "unsupported output overlap");
  if (polygeist_v2_overlaps(b, b_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_atan2_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cudnnBackendExecute_aten_atan2_f32", "cudnnBackendExecute");
  int pv2_k_a = polygeist_v2_map_add(&pv2_map, a, a_bytes, 0);
  int pv2_k_b = polygeist_v2_map_add(&pv2_map, b, b_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cudnn_pointwise_graph_f32(n, 570425600LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 1, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, a_stride, b_stride, 1, 1, output_stride, polygeist_v2_map_dev(&pv2_map, pv2_k_a), polygeist_v2_map_dev(&pv2_map, pv2_k_b), NULL, NULL, polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cudnnBackendExecute_aten_ceil_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, float *output, int64_t output_bytes, int32_t output_stride) {
  if (n <= 0) polygeist_v2_fatal("cudnnBackendExecute_aten_ceil_f32", "nonpositive dimension");
  if (a_stride <= 0 || !a || a_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * a_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_ceil_f32", "invalid strided FP32 row");
  if (output_stride <= 0 || !output || output_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * output_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_ceil_f32", "invalid strided FP32 row");
  if (polygeist_v2_overlaps(a, a_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_ceil_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cudnnBackendExecute_aten_ceil_f32", "cudnnBackendExecute");
  int pv2_k_a = polygeist_v2_map_add(&pv2_map, a, a_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cudnn_pointwise_graph_f32(n, 285212672LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 1, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, a_stride, 1, 1, 1, output_stride, polygeist_v2_map_dev(&pv2_map, pv2_k_a), NULL, NULL, NULL, polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cudnnBackendExecute_aten_clamp_max_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, float *output, int64_t output_bytes, int32_t output_stride, float max) {
  if (n <= 0) polygeist_v2_fatal("cudnnBackendExecute_aten_clamp_max_f32", "nonpositive dimension");
  if (a_stride <= 0 || !a || a_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * a_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_clamp_max_f32", "invalid strided FP32 row");
  if (output_stride <= 0 || !output || output_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * output_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_clamp_max_f32", "invalid strided FP32 row");
  if (polygeist_v2_overlaps(a, a_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_clamp_max_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cudnnBackendExecute_aten_clamp_max_f32", "cudnnBackendExecute");
  int pv2_k_a = polygeist_v2_map_add(&pv2_map, a, a_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cudnn_pointwise_graph_f32(n, 2093052325320196096LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 2, max, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, a_stride, 1, 1, 1, output_stride, polygeist_v2_map_dev(&pv2_map, pv2_k_a), NULL, NULL, NULL, polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cudnnBackendExecute_aten_clamp_min_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, float *output, int64_t output_bytes, int32_t output_stride, float min) {
  if (n <= 0) polygeist_v2_fatal("cudnnBackendExecute_aten_clamp_min_f32", "nonpositive dimension");
  if (a_stride <= 0 || !a || a_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * a_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_clamp_min_f32", "invalid strided FP32 row");
  if (output_stride <= 0 || !output || output_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * output_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_clamp_min_f32", "invalid strided FP32 row");
  if (polygeist_v2_overlaps(a, a_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_clamp_min_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cudnnBackendExecute_aten_clamp_min_f32", "cudnnBackendExecute");
  int pv2_k_a = polygeist_v2_map_add(&pv2_map, a, a_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cudnn_pointwise_graph_f32(n, 2093052325319934976LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 2, min, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, a_stride, 1, 1, 1, output_stride, polygeist_v2_map_dev(&pv2_map, pv2_k_a), NULL, NULL, NULL, polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cudnnBackendExecute_aten_clamp_scalar_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, float *output, int64_t output_bytes, int32_t output_stride, float min, float max) {
  if (n <= 0) polygeist_v2_fatal("cudnnBackendExecute_aten_clamp_scalar_f32", "nonpositive dimension");
  if (a_stride <= 0 || !a || a_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * a_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_clamp_scalar_f32", "invalid strided FP32 row");
  if (output_stride <= 0 || !output || output_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * output_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_clamp_scalar_f32", "invalid strided FP32 row");
  if (polygeist_v2_overlaps(a, a_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_clamp_scalar_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cudnnBackendExecute_aten_clamp_scalar_f32", "cudnnBackendExecute");
  int pv2_k_a = polygeist_v2_map_add(&pv2_map, a, a_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cudnn_pointwise_graph_f32(n, 2093052325319934976LL, 2093616430619888896LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 4, min, max, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, a_stride, 1, 1, 1, output_stride, polygeist_v2_map_dev(&pv2_map, pv2_k_a), NULL, NULL, NULL, polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cudnnBackendExecute_aten_cross_component_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, const float *b, int64_t b_bytes, int32_t b_stride, const float *c, int64_t c_bytes, int32_t c_stride, const float *d, int64_t d_bytes, int32_t d_stride, float *output, int64_t output_bytes, int32_t output_stride) {
  if (n <= 0) polygeist_v2_fatal("cudnnBackendExecute_aten_cross_component_f32", "nonpositive dimension");
  if (a_stride <= 0 || !a || a_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * a_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_cross_component_f32", "invalid strided FP32 row");
  if (b_stride <= 0 || !b || b_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * b_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_cross_component_f32", "invalid strided FP32 row");
  if (c_stride <= 0 || !c || c_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * c_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_cross_component_f32", "invalid strided FP32 row");
  if (d_stride <= 0 || !d || d_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * d_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_cross_component_f32", "invalid strided FP32 row");
  if (output_stride <= 0 || !output || output_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * output_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_cross_component_f32", "invalid strided FP32 row");
  if (polygeist_v2_overlaps(a, a_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_cross_component_f32", "unsupported output overlap");
  if (polygeist_v2_overlaps(b, b_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_cross_component_f32", "unsupported output overlap");
  if (polygeist_v2_overlaps(c, c_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_cross_component_f32", "unsupported output overlap");
  if (polygeist_v2_overlaps(d, d_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_cross_component_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cudnnBackendExecute_aten_cross_component_f32", "cudnnBackendExecute");
  int pv2_k_a = polygeist_v2_map_add(&pv2_map, a, a_bytes, 0);
  int pv2_k_b = polygeist_v2_map_add(&pv2_map, b, b_bytes, 0);
  int pv2_k_c = polygeist_v2_map_add(&pv2_map, c, c_bytes, 0);
  int pv2_k_d = polygeist_v2_map_add(&pv2_map, d, d_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cudnn_pointwise_graph_f32(n, 144681436597715200LL, 51121408LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 3, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, a_stride, b_stride, c_stride, d_stride, output_stride, polygeist_v2_map_dev(&pv2_map, pv2_k_a), polygeist_v2_map_dev(&pv2_map, pv2_k_b), polygeist_v2_map_dev(&pv2_map, pv2_k_c), polygeist_v2_map_dev(&pv2_map, pv2_k_d), polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cudnnBackendExecute_aten_div_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, const float *b, int64_t b_bytes, int32_t b_stride, float *output, int64_t output_bytes, int32_t output_stride) {
  if (n <= 0) polygeist_v2_fatal("cudnnBackendExecute_aten_div_f32", "nonpositive dimension");
  if (a_stride <= 0 || !a || a_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * a_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_div_f32", "invalid strided FP32 row");
  if (b_stride <= 0 || !b || b_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * b_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_div_f32", "invalid strided FP32 row");
  if (output_stride <= 0 || !output || output_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * output_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_div_f32", "invalid strided FP32 row");
  if (polygeist_v2_overlaps(a, a_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_div_f32", "unsupported output overlap");
  if (polygeist_v2_overlaps(b, b_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_div_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cudnnBackendExecute_aten_div_f32", "cudnnBackendExecute");
  int pv2_k_a = polygeist_v2_map_add(&pv2_map, a, a_bytes, 0);
  int pv2_k_b = polygeist_v2_map_add(&pv2_map, b, b_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cudnn_pointwise_graph_f32(n, 67109120LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 1, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, a_stride, b_stride, 1, 1, output_stride, polygeist_v2_map_dev(&pv2_map, pv2_k_a), polygeist_v2_map_dev(&pv2_map, pv2_k_b), NULL, NULL, polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cudnnBackendExecute_aten_eq_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, const float *b, int64_t b_bytes, int32_t b_stride, float *output, int64_t output_bytes, int32_t output_stride) {
  if (n <= 0) polygeist_v2_fatal("cudnnBackendExecute_aten_eq_f32", "nonpositive dimension");
  if (a_stride <= 0 || !a || a_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * a_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_eq_f32", "invalid strided FP32 row");
  if (b_stride <= 0 || !b || b_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * b_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_eq_f32", "invalid strided FP32 row");
  if (output_stride <= 0 || !output || output_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * output_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_eq_f32", "invalid strided FP32 row");
  if (polygeist_v2_overlaps(a, a_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_eq_f32", "unsupported output overlap");
  if (polygeist_v2_overlaps(b, b_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_eq_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cudnnBackendExecute_aten_eq_f32", "cudnnBackendExecute");
  int pv2_k_a = polygeist_v2_map_add(&pv2_map, a, a_bytes, 0);
  int pv2_k_b = polygeist_v2_map_add(&pv2_map, b, b_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cudnn_pointwise_graph_f32(n, 2093052346727661824LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 2, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, a_stride, b_stride, 1, 1, output_stride, polygeist_v2_map_dev(&pv2_map, pv2_k_a), polygeist_v2_map_dev(&pv2_map, pv2_k_b), NULL, NULL, polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cudnnBackendExecute_aten_erf_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, float *output, int64_t output_bytes, int32_t output_stride) {
  if (n <= 0) polygeist_v2_fatal("cudnnBackendExecute_aten_erf_f32", "nonpositive dimension");
  if (a_stride <= 0 || !a || a_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * a_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_erf_f32", "invalid strided FP32 row");
  if (output_stride <= 0 || !output || output_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * output_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_erf_f32", "invalid strided FP32 row");
  if (polygeist_v2_overlaps(a, a_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_erf_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cudnnBackendExecute_aten_erf_f32", "cudnnBackendExecute");
  int pv2_k_a = polygeist_v2_map_add(&pv2_map, a, a_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cudnn_pointwise_graph_f32(n, 301989888LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 1, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, a_stride, 1, 1, 1, output_stride, polygeist_v2_map_dev(&pv2_map, pv2_k_a), NULL, NULL, NULL, polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cudnnBackendExecute_aten_exp_f32(int32_t n, const float *x, int64_t x_bytes, int32_t x_stride, float *output, int64_t output_bytes, int32_t output_stride) {
  if (n <= 0) polygeist_v2_fatal("cudnnBackendExecute_aten_exp_f32", "nonpositive dimension");
  if (x_stride <= 0 || !x || x_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * x_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_exp_f32", "invalid strided FP32 row");
  if (output_stride <= 0 || !output || output_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * output_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_exp_f32", "invalid strided FP32 row");
  if (polygeist_v2_overlaps(x, x_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_exp_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cudnnBackendExecute_aten_exp_f32", "cudnnBackendExecute");
  int pv2_k_x = polygeist_v2_map_add(&pv2_map, x, x_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cudnn_pointwise_graph_f32(n, 117440512LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 1, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, x_stride, 1, 1, 1, output_stride, polygeist_v2_map_dev(&pv2_map, pv2_k_x), NULL, NULL, NULL, polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cudnnBackendExecute_aten_floor_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, float *output, int64_t output_bytes, int32_t output_stride) {
  if (n <= 0) polygeist_v2_fatal("cudnnBackendExecute_aten_floor_f32", "nonpositive dimension");
  if (a_stride <= 0 || !a || a_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * a_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_floor_f32", "invalid strided FP32 row");
  if (output_stride <= 0 || !output || output_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * output_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_floor_f32", "invalid strided FP32 row");
  if (polygeist_v2_overlaps(a, a_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_floor_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cudnnBackendExecute_aten_floor_f32", "cudnnBackendExecute");
  int pv2_k_a = polygeist_v2_map_add(&pv2_map, a, a_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cudnn_pointwise_graph_f32(n, 268435456LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 1, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, a_stride, 1, 1, 1, output_stride, polygeist_v2_map_dev(&pv2_map, pv2_k_a), NULL, NULL, NULL, polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cudnnBackendExecute_aten_frac_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, float *output, int64_t output_bytes, int32_t output_stride) {
  if (n <= 0) polygeist_v2_fatal("cudnnBackendExecute_aten_frac_f32", "nonpositive dimension");
  if (a_stride <= 0 || !a || a_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * a_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_frac_f32", "invalid strided FP32 row");
  if (output_stride <= 0 || !output || output_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * output_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_frac_f32", "invalid strided FP32 row");
  if (polygeist_v2_overlaps(a, a_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_frac_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cudnnBackendExecute_aten_frac_f32", "cudnnBackendExecute");
  int pv2_k_a = polygeist_v2_map_add(&pv2_map, a, a_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cudnn_pointwise_graph_f32(n, 1224979099097760768LL, 2093062280869576704LL, 50335488LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 5, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, a_stride, 1, 1, 1, output_stride, polygeist_v2_map_dev(&pv2_map, pv2_k_a), NULL, NULL, NULL, polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cudnnBackendExecute_aten_ge_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, const float *b, int64_t b_bytes, int32_t b_stride, float *output, int64_t output_bytes, int32_t output_stride) {
  if (n <= 0) polygeist_v2_fatal("cudnnBackendExecute_aten_ge_f32", "nonpositive dimension");
  if (a_stride <= 0 || !a || a_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * a_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_ge_f32", "invalid strided FP32 row");
  if (b_stride <= 0 || !b || b_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * b_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_ge_f32", "invalid strided FP32 row");
  if (output_stride <= 0 || !output || output_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * output_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_ge_f32", "invalid strided FP32 row");
  if (polygeist_v2_overlaps(a, a_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_ge_f32", "unsupported output overlap");
  if (polygeist_v2_overlaps(b, b_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_ge_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cudnnBackendExecute_aten_ge_f32", "cudnnBackendExecute");
  int pv2_k_a = polygeist_v2_map_add(&pv2_map, a, a_bytes, 0);
  int pv2_k_b = polygeist_v2_map_add(&pv2_map, b, b_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cudnn_pointwise_graph_f32(n, 2093052346777993472LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 2, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, a_stride, b_stride, 1, 1, output_stride, polygeist_v2_map_dev(&pv2_map, pv2_k_a), polygeist_v2_map_dev(&pv2_map, pv2_k_b), NULL, NULL, polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cudnnBackendExecute_aten_gelu_exact_f32(int32_t n, const float *x, int64_t x_bytes, int32_t x_stride, float *output, int64_t output_bytes, int32_t output_stride) {
  if (n <= 0) polygeist_v2_fatal("cudnnBackendExecute_aten_gelu_exact_f32", "nonpositive dimension");
  if (x_stride <= 0 || !x || x_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * x_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_gelu_exact_f32", "invalid strided FP32 row");
  if (output_stride <= 0 || !output || output_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * output_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_gelu_exact_f32", "invalid strided FP32 row");
  if (polygeist_v2_overlaps(x, x_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_gelu_exact_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cudnnBackendExecute_aten_gelu_exact_f32", "cudnnBackendExecute");
  int pv2_k_x = polygeist_v2_map_add(&pv2_map, x, x_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cudnn_pointwise_graph_f32(n, 144120685667550208LL, 76004841084485632LL, 34344704LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 5, 0.5f, 0.707106769f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, x_stride, 1, 1, 1, output_stride, polygeist_v2_map_dev(&pv2_map, pv2_k_x), NULL, NULL, NULL, polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cudnnBackendExecute_aten_gelu_tanh_f32(int32_t n, const float *x, int64_t x_bytes, int32_t x_stride, float *output, int64_t output_bytes, int32_t output_stride) {
  if (n <= 0) polygeist_v2_fatal("cudnnBackendExecute_aten_gelu_tanh_f32", "nonpositive dimension");
  if (x_stride <= 0 || !x || x_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * x_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_gelu_tanh_f32", "invalid strided FP32 row");
  if (output_stride <= 0 || !output || output_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * output_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_gelu_tanh_f32", "invalid strided FP32 row");
  if (polygeist_v2_overlaps(x, x_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_gelu_tanh_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cudnnBackendExecute_aten_gelu_tanh_f32", "cudnnBackendExecute");
  int pv2_k_x = polygeist_v2_map_add(&pv2_map, x, x_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cudnn_pointwise_graph_f32(n, 147492887829938176LL, 72072987235124224LL, 144120685668011776LL, 77130740790198272LL, 34673408LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 9, 0.044715f, 0.5f, 1.0f, 0.7978845834732056f, 0.0f, 0.0f, 0.0f, 0.0f, x_stride, 1, 1, 1, output_stride, polygeist_v2_map_dev(&pv2_map, pv2_k_x), NULL, NULL, NULL, polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cudnnBackendExecute_aten_gt_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, const float *b, int64_t b_bytes, int32_t b_stride, float *output, int64_t output_bytes, int32_t output_stride) {
  if (n <= 0) polygeist_v2_fatal("cudnnBackendExecute_aten_gt_f32", "nonpositive dimension");
  if (a_stride <= 0 || !a || a_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * a_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_gt_f32", "invalid strided FP32 row");
  if (b_stride <= 0 || !b || b_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * b_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_gt_f32", "invalid strided FP32 row");
  if (output_stride <= 0 || !output || output_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * output_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_gt_f32", "invalid strided FP32 row");
  if (polygeist_v2_overlaps(a, a_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_gt_f32", "unsupported output overlap");
  if (polygeist_v2_overlaps(b, b_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_gt_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cudnnBackendExecute_aten_gt_f32", "cudnnBackendExecute");
  int pv2_k_a = polygeist_v2_map_add(&pv2_map, a, a_bytes, 0);
  int pv2_k_b = polygeist_v2_map_add(&pv2_map, b, b_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cudnn_pointwise_graph_f32(n, 2093052346761216256LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 2, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, a_stride, b_stride, 1, 1, output_stride, polygeist_v2_map_dev(&pv2_map, pv2_k_a), polygeist_v2_map_dev(&pv2_map, pv2_k_b), NULL, NULL, polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cudnnBackendExecute_aten_hardsigmoid_backward_f32(int32_t n, const float *grad_val, int64_t grad_val_bytes, int32_t grad_val_stride, const float *self_val, int64_t self_val_bytes, int32_t self_val_stride, float *output, int64_t output_bytes, int32_t output_stride) {
  if (n <= 0) polygeist_v2_fatal("cudnnBackendExecute_aten_hardsigmoid_backward_f32", "nonpositive dimension");
  if (grad_val_stride <= 0 || !grad_val || grad_val_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * grad_val_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_hardsigmoid_backward_f32", "invalid strided FP32 row");
  if (self_val_stride <= 0 || !self_val || self_val_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * self_val_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_hardsigmoid_backward_f32", "invalid strided FP32 row");
  if (output_stride <= 0 || !output || output_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * output_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_hardsigmoid_backward_f32", "invalid strided FP32 row");
  if (polygeist_v2_overlaps(grad_val, grad_val_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_hardsigmoid_backward_f32", "unsupported output overlap");
  if (polygeist_v2_overlaps(self_val, self_val_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_hardsigmoid_backward_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cudnnBackendExecute_aten_hardsigmoid_backward_f32", "cudnnBackendExecute");
  int pv2_k_grad_val = polygeist_v2_map_add(&pv2_map, grad_val, grad_val_bytes, 0);
  int pv2_k_self_val = polygeist_v2_map_add(&pv2_map, self_val, self_val_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cudnn_pointwise_graph_f32(n, 1945560537001624576LL, 144403260626439424LL, 487460615LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 5, -3.0f, 3.0f, 0.166666672f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, self_val_stride, grad_val_stride, 1, 1, output_stride, polygeist_v2_map_dev(&pv2_map, pv2_k_self_val), polygeist_v2_map_dev(&pv2_map, pv2_k_grad_val), NULL, NULL, polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cudnnBackendExecute_aten_hardsigmoid_f32(int32_t n, const float *self_val, int64_t self_val_bytes, int32_t self_val_stride, float *output, int64_t output_bytes, int32_t output_stride) {
  if (n <= 0) polygeist_v2_fatal("cudnnBackendExecute_aten_hardsigmoid_f32", "nonpositive dimension");
  if (self_val_stride <= 0 || !self_val || self_val_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * self_val_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_hardsigmoid_f32", "invalid strided FP32 row");
  if (output_stride <= 0 || !output || output_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * output_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_hardsigmoid_f32", "invalid strided FP32 row");
  if (polygeist_v2_overlaps(self_val, self_val_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_hardsigmoid_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cudnnBackendExecute_aten_hardsigmoid_f32", "cudnnBackendExecute");
  int pv2_k_self_val = polygeist_v2_map_add(&pv2_map, self_val, self_val_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cudnn_pointwise_graph_f32(n, 1948938236319499264LL, 1947259282534499596LL, 292740573336372750LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 6, 3.0f, 0.0f, 6.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, self_val_stride, 1, 1, 1, output_stride, polygeist_v2_map_dev(&pv2_map, pv2_k_self_val), NULL, NULL, NULL, polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cudnnBackendExecute_aten_hardswish_backward_f32(int32_t n, const float *grad_val, int64_t grad_val_bytes, int32_t grad_val_stride, const float *self_val, int64_t self_val_bytes, int32_t self_val_stride, float *output, int64_t output_bytes, int32_t output_stride) {
  if (n <= 0) polygeist_v2_fatal("cudnnBackendExecute_aten_hardswish_backward_f32", "nonpositive dimension");
  if (grad_val_stride <= 0 || !grad_val || grad_val_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * grad_val_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_hardswish_backward_f32", "invalid strided FP32 row");
  if (self_val_stride <= 0 || !self_val || self_val_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * self_val_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_hardswish_backward_f32", "invalid strided FP32 row");
  if (output_stride <= 0 || !output || output_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * output_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_hardswish_backward_f32", "invalid strided FP32 row");
  if (polygeist_v2_overlaps(grad_val, grad_val_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_hardswish_backward_f32", "unsupported output overlap");
  if (polygeist_v2_overlaps(self_val, self_val_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_hardswish_backward_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cudnnBackendExecute_aten_hardswish_backward_f32", "cudnnBackendExecute");
  int pv2_k_grad_val = polygeist_v2_map_add(&pv2_map, grad_val, grad_val_bytes, 0);
  int pv2_k_self_val = polygeist_v2_map_add(&pv2_map, self_val, self_val_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cudnn_pointwise_graph_f32(n, 1945560537051956224LL, 76004840848753920LL, 2093346998311784192LL, 487327505LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 7, -3.0f, 3.0f, 0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, self_val_stride, grad_val_stride, 1, 1, output_stride, polygeist_v2_map_dev(&pv2_map, pv2_k_self_val), polygeist_v2_map_dev(&pv2_map, pv2_k_grad_val), NULL, NULL, polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cudnnBackendExecute_aten_hardswish_f32(int32_t n, const float *x, int64_t x_bytes, int32_t x_stride, float *output, int64_t output_bytes, int32_t output_stride) {
  if (n <= 0) polygeist_v2_fatal("cudnnBackendExecute_aten_hardswish_f32", "nonpositive dimension");
  if (x_stride <= 0 || !x || x_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * x_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_hardswish_f32", "invalid strided FP32 row");
  if (output_stride <= 0 || !output || output_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * output_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_hardswish_f32", "invalid strided FP32 row");
  if (polygeist_v2_overlaps(x, x_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_hardswish_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cudnnBackendExecute_aten_hardswish_f32", "cudnnBackendExecute");
  int pv2_k_x = polygeist_v2_map_add(&pv2_map, x, x_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cudnn_pointwise_graph_f32(n, 1948938236319499264LL, 1947259282534499596LL, 144132780749424142LL, 68224512LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 7, 3.0f, 0.0f, 6.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, x_stride, 1, 1, 1, output_stride, polygeist_v2_map_dev(&pv2_map, pv2_k_x), NULL, NULL, NULL, polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cudnnBackendExecute_aten_hardtanh_backward_f32(int32_t n, const float *grad_val, int64_t grad_val_bytes, int32_t grad_val_stride, const float *self_val, int64_t self_val_bytes, int32_t self_val_stride, float *output, int64_t output_bytes, int32_t output_stride, float min_val, float max_val) {
  if (n <= 0) polygeist_v2_fatal("cudnnBackendExecute_aten_hardtanh_backward_f32", "nonpositive dimension");
  if (grad_val_stride <= 0 || !grad_val || grad_val_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * grad_val_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_hardtanh_backward_f32", "invalid strided FP32 row");
  if (self_val_stride <= 0 || !self_val || self_val_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * self_val_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_hardtanh_backward_f32", "invalid strided FP32 row");
  if (output_stride <= 0 || !output || output_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * output_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_hardtanh_backward_f32", "invalid strided FP32 row");
  if (polygeist_v2_overlaps(grad_val, grad_val_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_hardtanh_backward_f32", "unsupported output overlap");
  if (polygeist_v2_overlaps(self_val, self_val_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_hardtanh_backward_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cudnnBackendExecute_aten_hardtanh_backward_f32", "cudnnBackendExecute");
  int pv2_k_grad_val = polygeist_v2_map_add(&pv2_map, grad_val, grad_val_bytes, 0);
  int pv2_k_self_val = polygeist_v2_map_add(&pv2_map, self_val, self_val_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cudnn_pointwise_graph_f32(n, 1873504042525656320LL, 2093615279636221184LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 4, 0.0f, min_val, max_val, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, self_val_stride, grad_val_stride, 1, 1, output_stride, polygeist_v2_map_dev(&pv2_map, pv2_k_self_val), polygeist_v2_map_dev(&pv2_map, pv2_k_grad_val), NULL, NULL, polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cudnnBackendExecute_aten_le_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, const float *b, int64_t b_bytes, int32_t b_stride, float *output, int64_t output_bytes, int32_t output_stride) {
  if (n <= 0) polygeist_v2_fatal("cudnnBackendExecute_aten_le_f32", "nonpositive dimension");
  if (a_stride <= 0 || !a || a_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * a_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_le_f32", "invalid strided FP32 row");
  if (b_stride <= 0 || !b || b_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * b_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_le_f32", "invalid strided FP32 row");
  if (output_stride <= 0 || !output || output_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * output_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_le_f32", "invalid strided FP32 row");
  if (polygeist_v2_overlaps(a, a_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_le_f32", "unsupported output overlap");
  if (polygeist_v2_overlaps(b, b_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_le_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cudnnBackendExecute_aten_le_f32", "cudnnBackendExecute");
  int pv2_k_a = polygeist_v2_map_add(&pv2_map, a, a_bytes, 0);
  int pv2_k_b = polygeist_v2_map_add(&pv2_map, b, b_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cudnn_pointwise_graph_f32(n, 2093052346811547904LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 2, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, a_stride, b_stride, 1, 1, output_stride, polygeist_v2_map_dev(&pv2_map, pv2_k_a), polygeist_v2_map_dev(&pv2_map, pv2_k_b), NULL, NULL, polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cudnnBackendExecute_aten_leaky_relu_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, float *output, int64_t output_bytes, int32_t output_stride, float negval) {
  if (n <= 0) polygeist_v2_fatal("cudnnBackendExecute_aten_leaky_relu_f32", "nonpositive dimension");
  if (a_stride <= 0 || !a || a_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * a_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_leaky_relu_f32", "invalid strided FP32 row");
  if (output_stride <= 0 || !output || output_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * output_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_leaky_relu_f32", "invalid strided FP32 row");
  if (polygeist_v2_overlaps(a, a_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_leaky_relu_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cudnnBackendExecute_aten_leaky_relu_f32", "cudnnBackendExecute");
  int pv2_k_a = polygeist_v2_map_add(&pv2_map, a, a_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cudnn_pointwise_graph_f32(n, 144120686053426176LL, 487325709LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 3, 0.0f, negval, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, a_stride, 1, 1, 1, output_stride, polygeist_v2_map_dev(&pv2_map, pv2_k_a), NULL, NULL, NULL, polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cudnnBackendExecute_aten_log_f32(int32_t n, const float *x, int64_t x_bytes, int32_t x_stride, float *output, int64_t output_bytes, int32_t output_stride) {
  if (n <= 0) polygeist_v2_fatal("cudnnBackendExecute_aten_log_f32", "nonpositive dimension");
  if (x_stride <= 0 || !x || x_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * x_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_log_f32", "invalid strided FP32 row");
  if (output_stride <= 0 || !output || output_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * output_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_log_f32", "invalid strided FP32 row");
  if (polygeist_v2_overlaps(x, x_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_log_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cudnnBackendExecute_aten_log_f32", "cudnnBackendExecute");
  int pv2_k_x = polygeist_v2_map_add(&pv2_map, x, x_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cudnn_pointwise_graph_f32(n, 201326592LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 1, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, x_stride, 1, 1, 1, output_stride, polygeist_v2_map_dev(&pv2_map, pv2_k_x), NULL, NULL, NULL, polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cudnnBackendExecute_aten_log_sigmoid_backward_f32(int32_t n, const float *input, int64_t input_bytes, int32_t input_stride, const float *buffer, int64_t buffer_bytes, int32_t buffer_stride, const float *grad, int64_t grad_bytes, int32_t grad_stride, float *output, int64_t output_bytes, int32_t output_stride) {
  if (n <= 0) polygeist_v2_fatal("cudnnBackendExecute_aten_log_sigmoid_backward_f32", "nonpositive dimension");
  if (input_stride <= 0 || !input || input_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * input_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_log_sigmoid_backward_f32", "invalid strided FP32 row");
  if (buffer_stride <= 0 || !buffer || buffer_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * buffer_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_log_sigmoid_backward_f32", "invalid strided FP32 row");
  if (grad_stride <= 0 || !grad || grad_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * grad_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_log_sigmoid_backward_f32", "invalid strided FP32 row");
  if (output_stride <= 0 || !output || output_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * output_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_log_sigmoid_backward_f32", "invalid strided FP32 row");
  if (polygeist_v2_overlaps(input, input_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_log_sigmoid_backward_f32", "unsupported output overlap");
  if (polygeist_v2_overlaps(buffer, buffer_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_log_sigmoid_backward_f32", "unsupported output overlap");
  if (polygeist_v2_overlaps(grad, grad_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_log_sigmoid_backward_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cudnnBackendExecute_aten_log_sigmoid_backward_f32", "cudnnBackendExecute");
  int pv2_k_input = polygeist_v2_map_add(&pv2_map, input, input_bytes, 0);
  int pv2_k_buffer = polygeist_v2_map_add(&pv2_map, buffer, buffer_bytes, 0);
  int pv2_k_grad = polygeist_v2_map_add(&pv2_map, grad, grad_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cudnn_pointwise_graph_f32(n, 2093053442011431936LL, 72344567060104454LL, 148073430003027712LL, 149183936731091200LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 8, 0.0f, 1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, input_stride, buffer_stride, grad_stride, 1, output_stride, polygeist_v2_map_dev(&pv2_map, pv2_k_input), polygeist_v2_map_dev(&pv2_map, pv2_k_buffer), polygeist_v2_map_dev(&pv2_map, pv2_k_grad), NULL, polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cudnnBackendExecute_aten_logical_and_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, const float *b, int64_t b_bytes, int32_t b_stride, float *output, int64_t output_bytes, int32_t output_stride) {
  if (n <= 0) polygeist_v2_fatal("cudnnBackendExecute_aten_logical_and_f32", "nonpositive dimension");
  if (a_stride <= 0 || !a || a_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * a_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_logical_and_f32", "invalid strided FP32 row");
  if (b_stride <= 0 || !b || b_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * b_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_logical_and_f32", "invalid strided FP32 row");
  if (output_stride <= 0 || !output || output_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * output_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_logical_and_f32", "invalid strided FP32 row");
  if (polygeist_v2_overlaps(a, a_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_logical_and_f32", "unsupported output overlap");
  if (polygeist_v2_overlaps(b, b_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_logical_and_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cudnnBackendExecute_aten_logical_and_f32", "cudnnBackendExecute");
  int pv2_k_a = polygeist_v2_map_add(&pv2_map, a, a_bytes, 0);
  int pv2_k_b = polygeist_v2_map_add(&pv2_map, b, b_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cudnn_pointwise_graph_f32(n, 1729668130336146432LL, 2093616392015973632LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 4, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, a_stride, b_stride, 1, 1, output_stride, polygeist_v2_map_dev(&pv2_map, pv2_k_a), polygeist_v2_map_dev(&pv2_map, pv2_k_b), NULL, NULL, polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cudnnBackendExecute_aten_logical_not_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, float *output, int64_t output_bytes, int32_t output_stride) {
  if (n <= 0) polygeist_v2_fatal("cudnnBackendExecute_aten_logical_not_f32", "nonpositive dimension");
  if (a_stride <= 0 || !a || a_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * a_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_logical_not_f32", "invalid strided FP32 row");
  if (output_stride <= 0 || !output || output_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * output_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_logical_not_f32", "invalid strided FP32 row");
  if (polygeist_v2_overlaps(a, a_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_logical_not_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cudnnBackendExecute_aten_logical_not_f32", "cudnnBackendExecute");
  int pv2_k_a = polygeist_v2_map_add(&pv2_map, a, a_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cudnn_pointwise_graph_f32(n, 2309220709336876032LL, 487392516LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 3, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, a_stride, 1, 1, 1, output_stride, polygeist_v2_map_dev(&pv2_map, pv2_k_a), NULL, NULL, NULL, polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cudnnBackendExecute_aten_logical_or_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, const float *b, int64_t b_bytes, int32_t b_stride, float *output, int64_t output_bytes, int32_t output_stride) {
  if (n <= 0) polygeist_v2_fatal("cudnnBackendExecute_aten_logical_or_f32", "nonpositive dimension");
  if (a_stride <= 0 || !a || a_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * a_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_logical_or_f32", "invalid strided FP32 row");
  if (b_stride <= 0 || !b || b_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * b_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_logical_or_f32", "invalid strided FP32 row");
  if (output_stride <= 0 || !output || output_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * output_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_logical_or_f32", "invalid strided FP32 row");
  if (polygeist_v2_overlaps(a, a_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_logical_or_f32", "unsupported output overlap");
  if (polygeist_v2_overlaps(b, b_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_logical_or_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cudnnBackendExecute_aten_logical_or_f32", "cudnnBackendExecute");
  int pv2_k_a = polygeist_v2_map_add(&pv2_map, a, a_bytes, 0);
  int pv2_k_b = polygeist_v2_map_add(&pv2_map, b, b_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cudnn_pointwise_graph_f32(n, 1729668130336146432LL, 2093616392032750848LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 4, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, a_stride, b_stride, 1, 1, output_stride, polygeist_v2_map_dev(&pv2_map, pv2_k_a), polygeist_v2_map_dev(&pv2_map, pv2_k_b), NULL, NULL, polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cudnnBackendExecute_aten_logical_xor_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, const float *b, int64_t b_bytes, int32_t b_stride, float *output, int64_t output_bytes, int32_t output_stride) {
  if (n <= 0) polygeist_v2_fatal("cudnnBackendExecute_aten_logical_xor_f32", "nonpositive dimension");
  if (a_stride <= 0 || !a || a_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * a_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_logical_xor_f32", "invalid strided FP32 row");
  if (b_stride <= 0 || !b || b_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * b_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_logical_xor_f32", "invalid strided FP32 row");
  if (output_stride <= 0 || !output || output_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * output_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_logical_xor_f32", "invalid strided FP32 row");
  if (polygeist_v2_overlaps(a, a_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_logical_xor_f32", "unsupported output overlap");
  if (polygeist_v2_overlaps(b, b_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_logical_xor_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cudnnBackendExecute_aten_logical_xor_f32", "cudnnBackendExecute");
  int pv2_k_a = polygeist_v2_map_add(&pv2_map, a, a_bytes, 0);
  int pv2_k_b = polygeist_v2_map_add(&pv2_map, b, b_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cudnn_pointwise_graph_f32(n, 1729668130336146432LL, 2309220709471944704LL, 2165964239943765504LL, 2094742291939856640LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 8, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, a_stride, b_stride, 1, 1, output_stride, polygeist_v2_map_dev(&pv2_map, pv2_k_a), polygeist_v2_map_dev(&pv2_map, pv2_k_b), NULL, NULL, polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cudnnBackendExecute_aten_lt_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, const float *b, int64_t b_bytes, int32_t b_stride, float *output, int64_t output_bytes, int32_t output_stride) {
  if (n <= 0) polygeist_v2_fatal("cudnnBackendExecute_aten_lt_f32", "nonpositive dimension");
  if (a_stride <= 0 || !a || a_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * a_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_lt_f32", "invalid strided FP32 row");
  if (b_stride <= 0 || !b || b_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * b_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_lt_f32", "invalid strided FP32 row");
  if (output_stride <= 0 || !output || output_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * output_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_lt_f32", "invalid strided FP32 row");
  if (polygeist_v2_overlaps(a, a_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_lt_f32", "unsupported output overlap");
  if (polygeist_v2_overlaps(b, b_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_lt_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cudnnBackendExecute_aten_lt_f32", "cudnnBackendExecute");
  int pv2_k_a = polygeist_v2_map_add(&pv2_map, a, a_bytes, 0);
  int pv2_k_b = polygeist_v2_map_add(&pv2_map, b, b_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cudnn_pointwise_graph_f32(n, 2093052346794770688LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 2, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, a_stride, b_stride, 1, 1, output_stride, polygeist_v2_map_dev(&pv2_map, pv2_k_a), polygeist_v2_map_dev(&pv2_map, pv2_k_b), NULL, NULL, polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cudnnBackendExecute_aten_mul_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, const float *b, int64_t b_bytes, int32_t b_stride, float *output, int64_t output_bytes, int32_t output_stride) {
  if (n <= 0) polygeist_v2_fatal("cudnnBackendExecute_aten_mul_f32", "nonpositive dimension");
  if (a_stride <= 0 || !a || a_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * a_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_mul_f32", "invalid strided FP32 row");
  if (b_stride <= 0 || !b || b_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * b_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_mul_f32", "invalid strided FP32 row");
  if (output_stride <= 0 || !output || output_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * output_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_mul_f32", "invalid strided FP32 row");
  if (polygeist_v2_overlaps(a, a_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_mul_f32", "unsupported output overlap");
  if (polygeist_v2_overlaps(b, b_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_mul_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cudnnBackendExecute_aten_mul_f32", "cudnnBackendExecute");
  int pv2_k_a = polygeist_v2_map_add(&pv2_map, a, a_bytes, 0);
  int pv2_k_b = polygeist_v2_map_add(&pv2_map, b, b_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cudnn_pointwise_graph_f32(n, 33554688LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 1, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, a_stride, b_stride, 1, 1, output_stride, polygeist_v2_map_dev(&pv2_map, pv2_k_a), polygeist_v2_map_dev(&pv2_map, pv2_k_b), NULL, NULL, polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cudnnBackendExecute_aten_ne_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, const float *b, int64_t b_bytes, int32_t b_stride, float *output, int64_t output_bytes, int32_t output_stride) {
  if (n <= 0) polygeist_v2_fatal("cudnnBackendExecute_aten_ne_f32", "nonpositive dimension");
  if (a_stride <= 0 || !a || a_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * a_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_ne_f32", "invalid strided FP32 row");
  if (b_stride <= 0 || !b || b_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * b_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_ne_f32", "invalid strided FP32 row");
  if (output_stride <= 0 || !output || output_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * output_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_ne_f32", "invalid strided FP32 row");
  if (polygeist_v2_overlaps(a, a_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_ne_f32", "unsupported output overlap");
  if (polygeist_v2_overlaps(b, b_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_ne_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cudnnBackendExecute_aten_ne_f32", "cudnnBackendExecute");
  int pv2_k_a = polygeist_v2_map_add(&pv2_map, a, a_bytes, 0);
  int pv2_k_b = polygeist_v2_map_add(&pv2_map, b, b_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cudnn_pointwise_graph_f32(n, 2093052346744439040LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 2, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, a_stride, b_stride, 1, 1, output_stride, polygeist_v2_map_dev(&pv2_map, pv2_k_a), polygeist_v2_map_dev(&pv2_map, pv2_k_b), NULL, NULL, polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cudnnBackendExecute_aten_neg_f32(int32_t n, const float *x, int64_t x_bytes, int32_t x_stride, float *output, int64_t output_bytes, int32_t output_stride) {
  if (n <= 0) polygeist_v2_fatal("cudnnBackendExecute_aten_neg_f32", "nonpositive dimension");
  if (x_stride <= 0 || !x || x_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * x_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_neg_f32", "invalid strided FP32 row");
  if (output_stride <= 0 || !output || output_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * output_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_neg_f32", "invalid strided FP32 row");
  if (polygeist_v2_overlaps(x, x_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_neg_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cudnnBackendExecute_aten_neg_f32", "cudnnBackendExecute");
  int pv2_k_x = polygeist_v2_map_add(&pv2_map, x, x_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cudnn_pointwise_graph_f32(n, 352321536LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 1, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, x_stride, 1, 1, 1, output_stride, polygeist_v2_map_dev(&pv2_map, pv2_k_x), NULL, NULL, NULL, polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cudnnBackendExecute_aten_pow_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, const float *b, int64_t b_bytes, int32_t b_stride, float *output, int64_t output_bytes, int32_t output_stride) {
  if (n <= 0) polygeist_v2_fatal("cudnnBackendExecute_aten_pow_f32", "nonpositive dimension");
  if (a_stride <= 0 || !a || a_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * a_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_pow_f32", "invalid strided FP32 row");
  if (b_stride <= 0 || !b || b_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * b_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_pow_f32", "invalid strided FP32 row");
  if (output_stride <= 0 || !output || output_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * output_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_pow_f32", "invalid strided FP32 row");
  if (polygeist_v2_overlaps(a, a_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_pow_f32", "unsupported output overlap");
  if (polygeist_v2_overlaps(b, b_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_pow_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cudnnBackendExecute_aten_pow_f32", "cudnnBackendExecute");
  int pv2_k_a = polygeist_v2_map_add(&pv2_map, a, a_bytes, 0);
  int pv2_k_b = polygeist_v2_map_add(&pv2_map, b, b_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cudnn_pointwise_graph_f32(n, 318767360LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 1, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, a_stride, b_stride, 1, 1, output_stride, polygeist_v2_map_dev(&pv2_map, pv2_k_a), polygeist_v2_map_dev(&pv2_map, pv2_k_b), NULL, NULL, polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cudnnBackendExecute_aten_pow_uniform_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, float exponent, float *output, int64_t output_bytes, int32_t output_stride) {
  if (n <= 0) polygeist_v2_fatal("cudnnBackendExecute_aten_pow_uniform_f32", "nonpositive dimension");
  if (a_stride <= 0 || !a || a_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * a_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_pow_uniform_f32", "invalid strided FP32 row");
  if (output_stride <= 0 || !output || output_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * output_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_pow_uniform_f32", "invalid strided FP32 row");
  if (polygeist_v2_overlaps(a, a_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_pow_uniform_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cudnnBackendExecute_aten_pow_uniform_f32", "cudnnBackendExecute");
  int pv2_k_a = polygeist_v2_map_add(&pv2_map, a, a_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cudnn_pointwise_graph_f32(n, 318768128LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 1, exponent, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, a_stride, 1, 1, 1, output_stride, polygeist_v2_map_dev(&pv2_map, pv2_k_a), NULL, NULL, NULL, polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cudnnBackendExecute_aten_reciprocal_f32(int32_t n, const float *x, int64_t x_bytes, int32_t x_stride, float *output, int64_t output_bytes, int32_t output_stride) {
  if (n <= 0) polygeist_v2_fatal("cudnnBackendExecute_aten_reciprocal_f32", "nonpositive dimension");
  if (x_stride <= 0 || !x || x_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * x_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_reciprocal_f32", "invalid strided FP32 row");
  if (output_stride <= 0 || !output || output_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * output_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_reciprocal_f32", "invalid strided FP32 row");
  if (polygeist_v2_overlaps(x, x_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_reciprocal_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cudnnBackendExecute_aten_reciprocal_f32", "cudnnBackendExecute");
  int pv2_k_x = polygeist_v2_map_add(&pv2_map, x, x_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cudnn_pointwise_graph_f32(n, 67371008LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 1, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, x_stride, 1, 1, 1, output_stride, polygeist_v2_map_dev(&pv2_map, pv2_k_x), NULL, NULL, NULL, polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cudnnBackendExecute_aten_renorm_scale_f32(int32_t n, const float *norm, int64_t norm_bytes, int32_t norm_stride, float maxnorm, float *output, int64_t output_bytes, int32_t output_stride) {
  if (n <= 0) polygeist_v2_fatal("cudnnBackendExecute_aten_renorm_scale_f32", "nonpositive dimension");
  if (norm_stride <= 0 || !norm || norm_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * norm_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_renorm_scale_f32", "invalid strided FP32 row");
  if (output_stride <= 0 || !output || output_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * output_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_renorm_scale_f32", "invalid strided FP32 row");
  if (polygeist_v2_overlaps(norm, norm_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_renorm_scale_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cudnnBackendExecute_aten_renorm_scale_f32", "cudnnBackendExecute");
  int pv2_k_norm = polygeist_v2_map_add(&pv2_map, norm, norm_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cudnn_pointwise_graph_f32(n, 72061992503870976LL, 2093063341525568768LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 4, 1e-07f, 1.0f, maxnorm, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, norm_stride, 1, 1, 1, output_stride, polygeist_v2_map_dev(&pv2_map, pv2_k_norm), NULL, NULL, NULL, polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cudnnBackendExecute_aten_rsqrt_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, float *output, int64_t output_bytes, int32_t output_stride) {
  if (n <= 0) polygeist_v2_fatal("cudnnBackendExecute_aten_rsqrt_f32", "nonpositive dimension");
  if (a_stride <= 0 || !a || a_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * a_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_rsqrt_f32", "invalid strided FP32 row");
  if (output_stride <= 0 || !output || output_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * output_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_rsqrt_f32", "invalid strided FP32 row");
  if (polygeist_v2_overlaps(a, a_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_rsqrt_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cudnnBackendExecute_aten_rsqrt_f32", "cudnnBackendExecute");
  int pv2_k_a = polygeist_v2_map_add(&pv2_map, a, a_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cudnn_pointwise_graph_f32(n, 289369470332305408LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 2, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, a_stride, 1, 1, 1, output_stride, polygeist_v2_map_dev(&pv2_map, pv2_k_a), NULL, NULL, NULL, polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cudnnBackendExecute_aten_scaled_add_f32(int32_t n, const float *self, int64_t self_bytes, int32_t self_stride, const float *other, int64_t other_bytes, int32_t other_stride, float *output, int64_t output_bytes, int32_t output_stride, float alpha) {
  if (n <= 0) polygeist_v2_fatal("cudnnBackendExecute_aten_scaled_add_f32", "nonpositive dimension");
  if (self_stride <= 0 || !self || self_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * self_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_scaled_add_f32", "invalid strided FP32 row");
  if (other_stride <= 0 || !other || other_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * other_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_scaled_add_f32", "invalid strided FP32 row");
  if (output_stride <= 0 || !output || output_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * output_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_scaled_add_f32", "invalid strided FP32 row");
  if (polygeist_v2_overlaps(self, self_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_scaled_add_f32", "unsupported output overlap");
  if (polygeist_v2_overlaps(other, other_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_scaled_add_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cudnnBackendExecute_aten_scaled_add_f32", "cudnnBackendExecute");
  int pv2_k_self = polygeist_v2_map_add(&pv2_map, self, self_bytes, 0);
  int pv2_k_other = polygeist_v2_map_add(&pv2_map, other, other_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cudnn_pointwise_graph_f32(n, 72070788211278080LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 2, alpha, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, self_stride, other_stride, 1, 1, output_stride, polygeist_v2_map_dev(&pv2_map, pv2_k_self), polygeist_v2_map_dev(&pv2_map, pv2_k_other), NULL, NULL, polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cudnnBackendExecute_aten_shrink_backward_f32(int32_t n, const float *grad_val, int64_t grad_val_bytes, int32_t grad_val_stride, const float *self_val, int64_t self_val_bytes, int32_t self_val_stride, float *output, int64_t output_bytes, int32_t output_stride, float lambd_val) {
  if (n <= 0) polygeist_v2_fatal("cudnnBackendExecute_aten_shrink_backward_f32", "nonpositive dimension");
  if (grad_val_stride <= 0 || !grad_val || grad_val_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * grad_val_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_shrink_backward_f32", "invalid strided FP32 row");
  if (self_val_stride <= 0 || !self_val || self_val_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * self_val_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_shrink_backward_f32", "invalid strided FP32 row");
  if (output_stride <= 0 || !output || output_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * output_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_shrink_backward_f32", "invalid strided FP32 row");
  if (polygeist_v2_overlaps(grad_val, grad_val_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_shrink_backward_f32", "unsupported output overlap");
  if (polygeist_v2_overlaps(self_val, self_val_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_shrink_backward_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cudnnBackendExecute_aten_shrink_backward_f32", "cudnnBackendExecute");
  int pv2_k_grad_val = polygeist_v2_map_add(&pv2_map, grad_val, grad_val_bytes, 0);
  int pv2_k_self_val = polygeist_v2_map_add(&pv2_map, self_val, self_val_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cudnn_pointwise_graph_f32(n, 1873510639478308864LL, 2165402389467628800LL, 487523329LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 5, 0.0f, lambd_val, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, self_val_stride, grad_val_stride, 1, 1, output_stride, polygeist_v2_map_dev(&pv2_map, pv2_k_self_val), polygeist_v2_map_dev(&pv2_map, pv2_k_grad_val), NULL, NULL, polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cudnnBackendExecute_aten_sigmoid_backward_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, const float *b, int64_t b_bytes, int32_t b_stride, float *output, int64_t output_bytes, int32_t output_stride) {
  if (n <= 0) polygeist_v2_fatal("cudnnBackendExecute_aten_sigmoid_backward_f32", "nonpositive dimension");
  if (a_stride <= 0 || !a || a_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * a_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_sigmoid_backward_f32", "invalid strided FP32 row");
  if (b_stride <= 0 || !b || b_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * b_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_sigmoid_backward_f32", "invalid strided FP32 row");
  if (output_stride <= 0 || !output || output_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * output_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_sigmoid_backward_f32", "invalid strided FP32 row");
  if (polygeist_v2_overlaps(a, a_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_sigmoid_backward_f32", "unsupported output overlap");
  if (polygeist_v2_overlaps(b, b_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_sigmoid_backward_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cudnnBackendExecute_aten_sigmoid_backward_f32", "cudnnBackendExecute");
  int pv2_k_a = polygeist_v2_map_add(&pv2_map, a, a_bytes, 0);
  int pv2_k_b = polygeist_v2_map_add(&pv2_map, b, b_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cudnn_pointwise_graph_f32(n, 144128382265983232LL, 34406656LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 3, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, a_stride, b_stride, 1, 1, output_stride, polygeist_v2_map_dev(&pv2_map, pv2_k_a), polygeist_v2_map_dev(&pv2_map, pv2_k_b), NULL, NULL, polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cudnnBackendExecute_aten_sigmoid_f32(int32_t n, const float *x, int64_t x_bytes, int32_t x_stride, float *output, int64_t output_bytes, int32_t output_stride) {
  if (n <= 0) polygeist_v2_fatal("cudnnBackendExecute_aten_sigmoid_f32", "nonpositive dimension");
  if (x_stride <= 0 || !x || x_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * x_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_sigmoid_f32", "invalid strided FP32 row");
  if (output_stride <= 0 || !output || output_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * output_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_sigmoid_f32", "invalid strided FP32 row");
  if (polygeist_v2_overlaps(x, x_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_sigmoid_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cudnnBackendExecute_aten_sigmoid_f32", "cudnnBackendExecute");
  int pv2_k_x = polygeist_v2_map_add(&pv2_map, x, x_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cudnn_pointwise_graph_f32(n, 507780858338344960LL, 289371669238973440LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 4, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, x_stride, 1, 1, 1, output_stride, polygeist_v2_map_dev(&pv2_map, pv2_k_x), NULL, NULL, NULL, polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cudnnBackendExecute_aten_sign_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, float *output, int64_t output_bytes, int32_t output_stride) {
  if (n <= 0) polygeist_v2_fatal("cudnnBackendExecute_aten_sign_f32", "nonpositive dimension");
  if (a_stride <= 0 || !a || a_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * a_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_sign_f32", "invalid strided FP32 row");
  if (output_stride <= 0 || !output || output_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * output_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_sign_f32", "invalid strided FP32 row");
  if (polygeist_v2_overlaps(a, a_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_sign_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cudnnBackendExecute_aten_sign_f32", "cudnnBackendExecute");
  int pv2_k_a = polygeist_v2_map_add(&pv2_map, a, a_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cudnn_pointwise_graph_f32(n, 1945559437523812352LL, 2093334917022483716LL, 51252992LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 5, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, a_stride, 1, 1, 1, output_stride, polygeist_v2_map_dev(&pv2_map, pv2_k_a), NULL, NULL, NULL, polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cudnnBackendExecute_aten_silu_backward_f32(int32_t n, const float *dy, int64_t dy_bytes, int32_t dy_stride, const float *x, int64_t x_bytes, int32_t x_stride, float *output, int64_t output_bytes, int32_t output_stride) {
  if (n <= 0) polygeist_v2_fatal("cudnnBackendExecute_aten_silu_backward_f32", "nonpositive dimension");
  if (dy_stride <= 0 || !dy || dy_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * dy_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_silu_backward_f32", "invalid strided FP32 row");
  if (x_stride <= 0 || !x || x_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * x_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_silu_backward_f32", "invalid strided FP32 row");
  if (output_stride <= 0 || !output || output_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * output_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_silu_backward_f32", "invalid strided FP32 row");
  if (polygeist_v2_overlaps(dy, dy_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_silu_backward_f32", "unsupported output overlap");
  if (polygeist_v2_overlaps(x, x_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_silu_backward_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cudnnBackendExecute_aten_silu_backward_f32", "cudnnBackendExecute");
  int pv2_k_dy = polygeist_v2_map_add(&pv2_map, dy, dy_bytes, 0);
  int pv2_k_x = polygeist_v2_map_add(&pv2_map, x, x_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cudnn_pointwise_graph_f32(n, 507780858338344960LL, 289371669238973440LL, 217315174728666880LL, 77128541698789632LL, 34607872LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 9, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, x_stride, dy_stride, 1, 1, output_stride, polygeist_v2_map_dev(&pv2_map, pv2_k_x), polygeist_v2_map_dev(&pv2_map, pv2_k_dy), NULL, NULL, polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cudnnBackendExecute_aten_silu_f32(int32_t n, const float *x, int64_t x_bytes, int32_t x_stride, float *output, int64_t output_bytes, int32_t output_stride) {
  if (n <= 0) polygeist_v2_fatal("cudnnBackendExecute_aten_silu_f32", "nonpositive dimension");
  if (x_stride <= 0 || !x || x_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * x_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_silu_f32", "invalid strided FP32 row");
  if (output_stride <= 0 || !output || output_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * output_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_silu_f32", "invalid strided FP32 row");
  if (polygeist_v2_overlaps(x, x_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_silu_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cudnnBackendExecute_aten_silu_f32", "cudnnBackendExecute");
  int pv2_k_x = polygeist_v2_map_add(&pv2_map, x, x_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cudnn_pointwise_graph_f32(n, 507780858338344960LL, 288245769332130816LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 4, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, x_stride, 1, 1, 1, output_stride, polygeist_v2_map_dev(&pv2_map, pv2_k_x), NULL, NULL, NULL, polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cudnnBackendExecute_aten_softshrink_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, float *output, int64_t output_bytes, int32_t output_stride, float lambd_val) {
  if (n <= 0) polygeist_v2_fatal("cudnnBackendExecute_aten_softshrink_f32", "nonpositive dimension");
  if (a_stride <= 0 || !a || a_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * a_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_softshrink_f32", "invalid strided FP32 row");
  if (output_stride <= 0 || !output || output_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * output_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_softshrink_f32", "invalid strided FP32 row");
  if (polygeist_v2_overlaps(a, a_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_softshrink_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cudnnBackendExecute_aten_softshrink_f32", "cudnnBackendExecute");
  int pv2_k_a = polygeist_v2_map_add(&pv2_map, a, a_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cudnn_pointwise_graph_f32(n, 216178280091354368LL, 1945570432539492352LL, 144119586139145472LL, 2093062298268536849LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 8, 0.0f, lambd_val, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, a_stride, 1, 1, 1, output_stride, polygeist_v2_map_dev(&pv2_map, pv2_k_a), NULL, NULL, NULL, polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cudnnBackendExecute_aten_sqrt_f32(int32_t n, const float *x, int64_t x_bytes, int32_t x_stride, float *output, int64_t output_bytes, int32_t output_stride) {
  if (n <= 0) polygeist_v2_fatal("cudnnBackendExecute_aten_sqrt_f32", "nonpositive dimension");
  if (x_stride <= 0 || !x || x_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * x_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_sqrt_f32", "invalid strided FP32 row");
  if (output_stride <= 0 || !output || output_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * output_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_sqrt_f32", "invalid strided FP32 row");
  if (polygeist_v2_overlaps(x, x_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_sqrt_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cudnnBackendExecute_aten_sqrt_f32", "cudnnBackendExecute");
  int pv2_k_x = polygeist_v2_map_add(&pv2_map, x, x_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cudnn_pointwise_graph_f32(n, 134217728LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 1, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, x_stride, 1, 1, 1, output_stride, polygeist_v2_map_dev(&pv2_map, pv2_k_x), NULL, NULL, NULL, polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cudnnBackendExecute_aten_tanh_backward_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, const float *b, int64_t b_bytes, int32_t b_stride, float *output, int64_t output_bytes, int32_t output_stride) {
  if (n <= 0) polygeist_v2_fatal("cudnnBackendExecute_aten_tanh_backward_f32", "nonpositive dimension");
  if (a_stride <= 0 || !a || a_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * a_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_tanh_backward_f32", "invalid strided FP32 row");
  if (b_stride <= 0 || !b || b_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * b_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_tanh_backward_f32", "invalid strided FP32 row");
  if (output_stride <= 0 || !output || output_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * output_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_tanh_backward_f32", "invalid strided FP32 row");
  if (polygeist_v2_overlaps(a, a_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_tanh_backward_f32", "unsupported output overlap");
  if (polygeist_v2_overlaps(b, b_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_tanh_backward_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cudnnBackendExecute_aten_tanh_backward_f32", "cudnnBackendExecute");
  int pv2_k_a = polygeist_v2_map_add(&pv2_map, a, a_bytes, 0);
  int pv2_k_b = polygeist_v2_map_add(&pv2_map, b, b_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cudnn_pointwise_graph_f32(n, 217311876193779968LL, 33557760LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 3, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, a_stride, b_stride, 1, 1, output_stride, polygeist_v2_map_dev(&pv2_map, pv2_k_a), polygeist_v2_map_dev(&pv2_map, pv2_k_b), NULL, NULL, polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cudnnBackendExecute_aten_tanh_f32(int32_t n, const float *x, int64_t x_bytes, int32_t x_stride, float *output, int64_t output_bytes, int32_t output_stride) {
  if (n <= 0) polygeist_v2_fatal("cudnnBackendExecute_aten_tanh_f32", "nonpositive dimension");
  if (x_stride <= 0 || !x || x_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * x_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_tanh_f32", "invalid strided FP32 row");
  if (output_stride <= 0 || !output || output_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * output_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_tanh_f32", "invalid strided FP32 row");
  if (polygeist_v2_overlaps(x, x_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_tanh_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cudnnBackendExecute_aten_tanh_f32", "cudnnBackendExecute");
  int pv2_k_x = polygeist_v2_map_add(&pv2_map, x, x_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cudnn_pointwise_graph_f32(n, 100663296LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 1, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, x_stride, 1, 1, 1, output_stride, polygeist_v2_map_dev(&pv2_map, pv2_k_x), NULL, NULL, NULL, polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cudnnBackendExecute_aten_threshold_f32(int32_t n, const float *x, int64_t x_bytes, int32_t x_stride, const float *other, int64_t other_bytes, int32_t other_stride, float *output, int64_t output_bytes, int32_t output_stride, float threshold, float value) {
  if (n <= 0) polygeist_v2_fatal("cudnnBackendExecute_aten_threshold_f32", "nonpositive dimension");
  if (x_stride <= 0 || !x || x_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * x_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_threshold_f32", "invalid strided FP32 row");
  if (other_stride <= 0 || !other || other_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * other_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_threshold_f32", "invalid strided FP32 row");
  if (output_stride <= 0 || !output || output_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * output_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_threshold_f32", "invalid strided FP32 row");
  if (polygeist_v2_overlaps(x, x_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_threshold_f32", "unsupported output overlap");
  if (polygeist_v2_overlaps(other, other_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_threshold_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cudnnBackendExecute_aten_threshold_f32", "cudnnBackendExecute");
  int pv2_k_x = polygeist_v2_map_add(&pv2_map, x, x_bytes, 0);
  int pv2_k_other = polygeist_v2_map_add(&pv2_map, other, other_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cudnn_pointwise_graph_f32(n, 2093053429143307264LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 2, threshold, value, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, x_stride, other_stride, 1, 1, output_stride, polygeist_v2_map_dev(&pv2_map, pv2_k_x), polygeist_v2_map_dev(&pv2_map, pv2_k_other), NULL, NULL, polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cudnnBackendExecute_aten_trunc_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, float *output, int64_t output_bytes, int32_t output_stride) {
  if (n <= 0) polygeist_v2_fatal("cudnnBackendExecute_aten_trunc_f32", "nonpositive dimension");
  if (a_stride <= 0 || !a || a_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * a_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_trunc_f32", "invalid strided FP32 row");
  if (output_stride <= 0 || !output || output_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * output_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_aten_trunc_f32", "invalid strided FP32 row");
  if (polygeist_v2_overlaps(a, a_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_aten_trunc_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cudnnBackendExecute_aten_trunc_f32", "cudnnBackendExecute");
  int pv2_k_a = polygeist_v2_map_add(&pv2_map, a, a_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cudnn_pointwise_graph_f32(n, 1224979099097760768LL, 2093062280869576704LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 4, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, a_stride, 1, 1, 1, output_stride, polygeist_v2_map_dev(&pv2_map, pv2_k_a), NULL, NULL, NULL, polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cudnnBackendExecute_cutlass_rmsnorm_f32(int32_t n, const float *input, int64_t input_bytes, int32_t input_stride, const float *weight, int64_t weight_bytes, int32_t weight_stride, float *output, int64_t output_bytes, int32_t output_stride) {
  if (n <= 0) polygeist_v2_fatal("cudnnBackendExecute_cutlass_rmsnorm_f32", "nonpositive dimension");
  if (input_stride != 1 || !input || input_bytes < (int64_t)n * sizeof(float))
    polygeist_v2_fatal("cudnnBackendExecute_cutlass_rmsnorm_f32", "runtime adapter requires a contiguous FP32 row");
  if (weight_stride != 1 || !weight || weight_bytes < (int64_t)n * sizeof(float))
    polygeist_v2_fatal("cudnnBackendExecute_cutlass_rmsnorm_f32", "runtime adapter requires a contiguous FP32 row");
  if (output_stride != 1 || !output || output_bytes < (int64_t)n * sizeof(float))
    polygeist_v2_fatal("cudnnBackendExecute_cutlass_rmsnorm_f32", "runtime adapter requires a contiguous FP32 row");
  if (polygeist_v2_overlaps(input, input_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_cutlass_rmsnorm_f32", "unsupported output overlap");
  if (polygeist_v2_overlaps(weight, weight_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_cutlass_rmsnorm_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cudnnBackendExecute_cutlass_rmsnorm_f32", "cudnnBackendExecute");
  int pv2_k_input = polygeist_v2_map_add(&pv2_map, input, input_bytes, 0);
  int pv2_k_weight = polygeist_v2_map_add(&pv2_map, weight, weight_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_rmsnorm_f32(n, polygeist_v2_map_dev(&pv2_map, pv2_k_input), polygeist_v2_map_dev(&pv2_map, pv2_k_weight), polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cudnnBackendExecute_cutlass_rmsnorm_parameterized_f32(int32_t n, const float *input, int64_t input_bytes, int32_t input_stride, const float *weight, int64_t weight_bytes, int32_t weight_stride, float *output, int64_t output_bytes, int32_t output_stride, float epsilon) {
  if (n <= 0) polygeist_v2_fatal("cudnnBackendExecute_cutlass_rmsnorm_parameterized_f32", "nonpositive dimension");
  if (input_stride != 1 || !input || input_bytes < (int64_t)n * sizeof(float))
    polygeist_v2_fatal("cudnnBackendExecute_cutlass_rmsnorm_parameterized_f32", "runtime adapter requires a contiguous FP32 row");
  if (weight_stride != 1 || !weight || weight_bytes < (int64_t)n * sizeof(float))
    polygeist_v2_fatal("cudnnBackendExecute_cutlass_rmsnorm_parameterized_f32", "runtime adapter requires a contiguous FP32 row");
  if (output_stride != 1 || !output || output_bytes < (int64_t)n * sizeof(float))
    polygeist_v2_fatal("cudnnBackendExecute_cutlass_rmsnorm_parameterized_f32", "runtime adapter requires a contiguous FP32 row");
  if (polygeist_v2_overlaps(input, input_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_cutlass_rmsnorm_parameterized_f32", "unsupported output overlap");
  if (polygeist_v2_overlaps(weight, weight_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_cutlass_rmsnorm_parameterized_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cudnnBackendExecute_cutlass_rmsnorm_parameterized_f32", "cudnnBackendExecute");
  int pv2_k_input = polygeist_v2_map_add(&pv2_map, input, input_bytes, 0);
  int pv2_k_weight = polygeist_v2_map_add(&pv2_map, weight, weight_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_rmsnorm_f32_epsilon(n, polygeist_v2_map_dev(&pv2_map, pv2_k_input), polygeist_v2_map_dev(&pv2_map, pv2_k_weight), polygeist_v2_map_dev(&pv2_map, pv2_k_output), epsilon);
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cudnnBackendExecute_ggml_swiglu_f32(int32_t n, const float *x, int64_t x_bytes, int32_t x_stride, const float *gate, int64_t gate_bytes, int32_t gate_stride, float *output, int64_t output_bytes, int32_t output_stride) {
  if (n <= 0) polygeist_v2_fatal("cudnnBackendExecute_ggml_swiglu_f32", "nonpositive dimension");
  if (x_stride <= 0 || !x || x_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * x_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_ggml_swiglu_f32", "invalid strided FP32 row");
  if (gate_stride <= 0 || !gate || gate_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * gate_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_ggml_swiglu_f32", "invalid strided FP32 row");
  if (output_stride <= 0 || !output || output_bytes / (int64_t)sizeof(float) < ((int64_t)n-1) * output_stride + 1)
    polygeist_v2_fatal("cudnnBackendExecute_ggml_swiglu_f32", "invalid strided FP32 row");
  if (polygeist_v2_overlaps(x, x_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_ggml_swiglu_f32", "unsupported output overlap");
  if (polygeist_v2_overlaps(gate, gate_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnBackendExecute_ggml_swiglu_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cudnnBackendExecute_ggml_swiglu_f32", "cudnnBackendExecute");
  int pv2_k_x = polygeist_v2_map_add(&pv2_map, x, x_bytes, 0);
  int pv2_k_gate = polygeist_v2_map_add(&pv2_map, gate, gate_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cudnn_pointwise_graph_f32(n, 507780858338344960LL, 288245769332130816LL, 34537728LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 5, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, x_stride, gate_stride, 1, 1, output_stride, polygeist_v2_map_dev(&pv2_map, pv2_k_x), polygeist_v2_map_dev(&pv2_map, pv2_k_gate), NULL, NULL, polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cudnnConvolutionForward_cudnn_conv2d_valid_d1_f32(int32_t N, int32_t IC, int32_t IH, int32_t IW, int32_t OC, int32_t KH, int32_t KW, int32_t OH, int32_t OW, const float *input, int64_t input_bytes, int32_t input_stride, const float *filter, int64_t filter_bytes, int32_t filter_stride, float *output, int64_t output_bytes, int32_t output_stride) {
  if (N < 1 || N > 2147483647)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv2d_valid_d1_f32", "integer outside wrapper bounds");
  if (IC < 1 || IC > 2147483647)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv2d_valid_d1_f32", "integer outside wrapper bounds");
  if (IH < 1 || IH > 2147483647)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv2d_valid_d1_f32", "integer outside wrapper bounds");
  if (IW < 1 || IW > 2147483647)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv2d_valid_d1_f32", "integer outside wrapper bounds");
  if (OC < 1 || OC > 2147483647)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv2d_valid_d1_f32", "integer outside wrapper bounds");
  if (KH < 1 || KH > 2147483647)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv2d_valid_d1_f32", "integer outside wrapper bounds");
  if (KW < 1 || KW > 2147483647)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv2d_valid_d1_f32", "integer outside wrapper bounds");
  if (OH < 1 || OH > 2147483647)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv2d_valid_d1_f32", "integer outside wrapper bounds");
  if (OW < 1 || OW > 2147483647)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv2d_valid_d1_f32", "integer outside wrapper bounds");
  int64_t pv2_bytes_input = sizeof(float);
  if ((int64_t)N <= 0 || pv2_bytes_input > INT64_MAX / (int64_t)N)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv2d_valid_d1_f32", "array extent overflow");
  pv2_bytes_input *= (int64_t)N;
  if ((int64_t)IC <= 0 || pv2_bytes_input > INT64_MAX / (int64_t)IC)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv2d_valid_d1_f32", "array extent overflow");
  pv2_bytes_input *= (int64_t)IC;
  if ((int64_t)IH <= 0 || pv2_bytes_input > INT64_MAX / (int64_t)IH)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv2d_valid_d1_f32", "array extent overflow");
  pv2_bytes_input *= (int64_t)IH;
  if ((int64_t)IW <= 0 || pv2_bytes_input > INT64_MAX / (int64_t)IW)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv2d_valid_d1_f32", "array extent overflow");
  pv2_bytes_input *= (int64_t)IW;
  if (input_stride != 1 || !input || input_bytes < pv2_bytes_input)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv2d_valid_d1_f32", "array layout or capacity mismatch");
  int64_t pv2_bytes_filter = sizeof(float);
  if ((int64_t)OC <= 0 || pv2_bytes_filter > INT64_MAX / (int64_t)OC)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv2d_valid_d1_f32", "array extent overflow");
  pv2_bytes_filter *= (int64_t)OC;
  if ((int64_t)IC <= 0 || pv2_bytes_filter > INT64_MAX / (int64_t)IC)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv2d_valid_d1_f32", "array extent overflow");
  pv2_bytes_filter *= (int64_t)IC;
  if ((int64_t)KH <= 0 || pv2_bytes_filter > INT64_MAX / (int64_t)KH)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv2d_valid_d1_f32", "array extent overflow");
  pv2_bytes_filter *= (int64_t)KH;
  if ((int64_t)KW <= 0 || pv2_bytes_filter > INT64_MAX / (int64_t)KW)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv2d_valid_d1_f32", "array extent overflow");
  pv2_bytes_filter *= (int64_t)KW;
  if (filter_stride != 1 || !filter || filter_bytes < pv2_bytes_filter)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv2d_valid_d1_f32", "array layout or capacity mismatch");
  int64_t pv2_bytes_output = sizeof(float);
  if ((int64_t)N <= 0 || pv2_bytes_output > INT64_MAX / (int64_t)N)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv2d_valid_d1_f32", "array extent overflow");
  pv2_bytes_output *= (int64_t)N;
  if ((int64_t)OC <= 0 || pv2_bytes_output > INT64_MAX / (int64_t)OC)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv2d_valid_d1_f32", "array extent overflow");
  pv2_bytes_output *= (int64_t)OC;
  if ((int64_t)OH <= 0 || pv2_bytes_output > INT64_MAX / (int64_t)OH)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv2d_valid_d1_f32", "array extent overflow");
  pv2_bytes_output *= (int64_t)OH;
  if ((int64_t)OW <= 0 || pv2_bytes_output > INT64_MAX / (int64_t)OW)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv2d_valid_d1_f32", "array extent overflow");
  pv2_bytes_output *= (int64_t)OW;
  if (output_stride != 1 || !output || output_bytes < pv2_bytes_output)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv2d_valid_d1_f32", "array layout or capacity mismatch");
  if (polygeist_v2_overlaps(input, input_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv2d_valid_d1_f32", "unsupported output overlap");
  if (polygeist_v2_overlaps(filter, filter_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv2d_valid_d1_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cudnnConvolutionForward_cudnn_conv2d_valid_d1_f32", "polygeist_reference_conv2d_d1_f32");
  int pv2_k_input = polygeist_v2_map_add(&pv2_map, input, input_bytes, 0);
  int pv2_k_filter = polygeist_v2_map_add(&pv2_map, filter, filter_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_reference_conv2d_d1_f32(N, IC, IH, IW, OC, KH, KW, OH, OW, (const float *)polygeist_v2_map_dev(&pv2_map, pv2_k_input), (const float *)polygeist_v2_map_dev(&pv2_map, pv2_k_filter), (float *)polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cudnnConvolutionForward_cudnn_conv2d_valid_d2_f32(int32_t N, int32_t IC, int32_t IH, int32_t IW, int32_t OC, int32_t KH, int32_t KW, int32_t OH, int32_t OW, const float *input, int64_t input_bytes, int32_t input_stride, const float *filter, int64_t filter_bytes, int32_t filter_stride, float *output, int64_t output_bytes, int32_t output_stride) {
  if (N < 1 || N > 2147483647)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv2d_valid_d2_f32", "integer outside wrapper bounds");
  if (IC < 1 || IC > 2147483647)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv2d_valid_d2_f32", "integer outside wrapper bounds");
  if (IH < 1 || IH > 2147483647)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv2d_valid_d2_f32", "integer outside wrapper bounds");
  if (IW < 1 || IW > 2147483647)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv2d_valid_d2_f32", "integer outside wrapper bounds");
  if (OC < 1 || OC > 2147483647)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv2d_valid_d2_f32", "integer outside wrapper bounds");
  if (KH < 1 || KH > 2147483647)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv2d_valid_d2_f32", "integer outside wrapper bounds");
  if (KW < 1 || KW > 2147483647)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv2d_valid_d2_f32", "integer outside wrapper bounds");
  if (OH < 1 || OH > 2147483647)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv2d_valid_d2_f32", "integer outside wrapper bounds");
  if (OW < 1 || OW > 2147483647)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv2d_valid_d2_f32", "integer outside wrapper bounds");
  int64_t pv2_bytes_input = sizeof(float);
  if ((int64_t)N <= 0 || pv2_bytes_input > INT64_MAX / (int64_t)N)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv2d_valid_d2_f32", "array extent overflow");
  pv2_bytes_input *= (int64_t)N;
  if ((int64_t)IC <= 0 || pv2_bytes_input > INT64_MAX / (int64_t)IC)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv2d_valid_d2_f32", "array extent overflow");
  pv2_bytes_input *= (int64_t)IC;
  if ((int64_t)IH <= 0 || pv2_bytes_input > INT64_MAX / (int64_t)IH)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv2d_valid_d2_f32", "array extent overflow");
  pv2_bytes_input *= (int64_t)IH;
  if ((int64_t)IW <= 0 || pv2_bytes_input > INT64_MAX / (int64_t)IW)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv2d_valid_d2_f32", "array extent overflow");
  pv2_bytes_input *= (int64_t)IW;
  if (input_stride != 1 || !input || input_bytes < pv2_bytes_input)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv2d_valid_d2_f32", "array layout or capacity mismatch");
  int64_t pv2_bytes_filter = sizeof(float);
  if ((int64_t)OC <= 0 || pv2_bytes_filter > INT64_MAX / (int64_t)OC)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv2d_valid_d2_f32", "array extent overflow");
  pv2_bytes_filter *= (int64_t)OC;
  if ((int64_t)IC <= 0 || pv2_bytes_filter > INT64_MAX / (int64_t)IC)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv2d_valid_d2_f32", "array extent overflow");
  pv2_bytes_filter *= (int64_t)IC;
  if ((int64_t)KH <= 0 || pv2_bytes_filter > INT64_MAX / (int64_t)KH)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv2d_valid_d2_f32", "array extent overflow");
  pv2_bytes_filter *= (int64_t)KH;
  if ((int64_t)KW <= 0 || pv2_bytes_filter > INT64_MAX / (int64_t)KW)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv2d_valid_d2_f32", "array extent overflow");
  pv2_bytes_filter *= (int64_t)KW;
  if (filter_stride != 1 || !filter || filter_bytes < pv2_bytes_filter)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv2d_valid_d2_f32", "array layout or capacity mismatch");
  int64_t pv2_bytes_output = sizeof(float);
  if ((int64_t)N <= 0 || pv2_bytes_output > INT64_MAX / (int64_t)N)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv2d_valid_d2_f32", "array extent overflow");
  pv2_bytes_output *= (int64_t)N;
  if ((int64_t)OC <= 0 || pv2_bytes_output > INT64_MAX / (int64_t)OC)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv2d_valid_d2_f32", "array extent overflow");
  pv2_bytes_output *= (int64_t)OC;
  if ((int64_t)OH <= 0 || pv2_bytes_output > INT64_MAX / (int64_t)OH)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv2d_valid_d2_f32", "array extent overflow");
  pv2_bytes_output *= (int64_t)OH;
  if ((int64_t)OW <= 0 || pv2_bytes_output > INT64_MAX / (int64_t)OW)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv2d_valid_d2_f32", "array extent overflow");
  pv2_bytes_output *= (int64_t)OW;
  if (output_stride != 1 || !output || output_bytes < pv2_bytes_output)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv2d_valid_d2_f32", "array layout or capacity mismatch");
  if (polygeist_v2_overlaps(input, input_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv2d_valid_d2_f32", "unsupported output overlap");
  if (polygeist_v2_overlaps(filter, filter_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv2d_valid_d2_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cudnnConvolutionForward_cudnn_conv2d_valid_d2_f32", "polygeist_reference_conv2d_d2_f32");
  int pv2_k_input = polygeist_v2_map_add(&pv2_map, input, input_bytes, 0);
  int pv2_k_filter = polygeist_v2_map_add(&pv2_map, filter, filter_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_reference_conv2d_d2_f32(N, IC, IH, IW, OC, KH, KW, OH, OW, (const float *)polygeist_v2_map_dev(&pv2_map, pv2_k_input), (const float *)polygeist_v2_map_dev(&pv2_map, pv2_k_filter), (float *)polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cudnnConvolutionForward_cudnn_conv2d_valid_unbatched_d2_f32(int32_t IC, int32_t IH, int32_t IW, int32_t OC, int32_t KH, int32_t KW, int32_t OH, int32_t OW, const float *input, int64_t input_bytes, int32_t input_stride, const float *filter, int64_t filter_bytes, int32_t filter_stride, float *output, int64_t output_bytes, int32_t output_stride) {
  if (IC < 1 || IC > 2147483647)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv2d_valid_unbatched_d2_f32", "integer outside wrapper bounds");
  if (IH < 1 || IH > 2147483647)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv2d_valid_unbatched_d2_f32", "integer outside wrapper bounds");
  if (IW < 1 || IW > 2147483647)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv2d_valid_unbatched_d2_f32", "integer outside wrapper bounds");
  if (OC < 1 || OC > 2147483647)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv2d_valid_unbatched_d2_f32", "integer outside wrapper bounds");
  if (KH < 1 || KH > 2147483647)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv2d_valid_unbatched_d2_f32", "integer outside wrapper bounds");
  if (KW < 1 || KW > 2147483647)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv2d_valid_unbatched_d2_f32", "integer outside wrapper bounds");
  if (OH < 1 || OH > 2147483647)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv2d_valid_unbatched_d2_f32", "integer outside wrapper bounds");
  if (OW < 1 || OW > 2147483647)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv2d_valid_unbatched_d2_f32", "integer outside wrapper bounds");
  int64_t pv2_bytes_input = sizeof(float);
  if ((int64_t)IC <= 0 || pv2_bytes_input > INT64_MAX / (int64_t)IC)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv2d_valid_unbatched_d2_f32", "array extent overflow");
  pv2_bytes_input *= (int64_t)IC;
  if ((int64_t)IH <= 0 || pv2_bytes_input > INT64_MAX / (int64_t)IH)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv2d_valid_unbatched_d2_f32", "array extent overflow");
  pv2_bytes_input *= (int64_t)IH;
  if ((int64_t)IW <= 0 || pv2_bytes_input > INT64_MAX / (int64_t)IW)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv2d_valid_unbatched_d2_f32", "array extent overflow");
  pv2_bytes_input *= (int64_t)IW;
  if (input_stride != 1 || !input || input_bytes < pv2_bytes_input)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv2d_valid_unbatched_d2_f32", "array layout or capacity mismatch");
  int64_t pv2_bytes_filter = sizeof(float);
  if ((int64_t)OC <= 0 || pv2_bytes_filter > INT64_MAX / (int64_t)OC)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv2d_valid_unbatched_d2_f32", "array extent overflow");
  pv2_bytes_filter *= (int64_t)OC;
  if ((int64_t)IC <= 0 || pv2_bytes_filter > INT64_MAX / (int64_t)IC)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv2d_valid_unbatched_d2_f32", "array extent overflow");
  pv2_bytes_filter *= (int64_t)IC;
  if ((int64_t)KH <= 0 || pv2_bytes_filter > INT64_MAX / (int64_t)KH)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv2d_valid_unbatched_d2_f32", "array extent overflow");
  pv2_bytes_filter *= (int64_t)KH;
  if ((int64_t)KW <= 0 || pv2_bytes_filter > INT64_MAX / (int64_t)KW)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv2d_valid_unbatched_d2_f32", "array extent overflow");
  pv2_bytes_filter *= (int64_t)KW;
  if (filter_stride != 1 || !filter || filter_bytes < pv2_bytes_filter)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv2d_valid_unbatched_d2_f32", "array layout or capacity mismatch");
  int64_t pv2_bytes_output = sizeof(float);
  if ((int64_t)OC <= 0 || pv2_bytes_output > INT64_MAX / (int64_t)OC)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv2d_valid_unbatched_d2_f32", "array extent overflow");
  pv2_bytes_output *= (int64_t)OC;
  if ((int64_t)OH <= 0 || pv2_bytes_output > INT64_MAX / (int64_t)OH)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv2d_valid_unbatched_d2_f32", "array extent overflow");
  pv2_bytes_output *= (int64_t)OH;
  if ((int64_t)OW <= 0 || pv2_bytes_output > INT64_MAX / (int64_t)OW)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv2d_valid_unbatched_d2_f32", "array extent overflow");
  pv2_bytes_output *= (int64_t)OW;
  if (output_stride != 1 || !output || output_bytes < pv2_bytes_output)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv2d_valid_unbatched_d2_f32", "array layout or capacity mismatch");
  if (polygeist_v2_overlaps(input, input_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv2d_valid_unbatched_d2_f32", "unsupported output overlap");
  if (polygeist_v2_overlaps(filter, filter_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv2d_valid_unbatched_d2_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cudnnConvolutionForward_cudnn_conv2d_valid_unbatched_d2_f32", "polygeist_reference_conv2d_unbatched_d2_f32");
  int pv2_k_input = polygeist_v2_map_add(&pv2_map, input, input_bytes, 0);
  int pv2_k_filter = polygeist_v2_map_add(&pv2_map, filter, filter_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_reference_conv2d_unbatched_d2_f32(IC, IH, IW, OC, KH, KW, OH, OW, (const float *)polygeist_v2_map_dev(&pv2_map, pv2_k_input), (const float *)polygeist_v2_map_dev(&pv2_map, pv2_k_filter), (float *)polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cudnnConvolutionForward_cudnn_conv3d_valid_f32(int32_t IC, int32_t ID, int32_t IH, int32_t IW, int32_t OC, int32_t KD, int32_t KH, int32_t KW, int32_t OD, int32_t OH, int32_t OW, const float *input, int64_t input_bytes, int32_t input_stride, const float *filter, int64_t filter_bytes, int32_t filter_stride, float *output, int64_t output_bytes, int32_t output_stride) {
  if (IC < 1 || IC > 2147483647)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv3d_valid_f32", "integer outside wrapper bounds");
  if (ID < 1 || ID > 2147483647)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv3d_valid_f32", "integer outside wrapper bounds");
  if (IH < 1 || IH > 2147483647)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv3d_valid_f32", "integer outside wrapper bounds");
  if (IW < 1 || IW > 2147483647)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv3d_valid_f32", "integer outside wrapper bounds");
  if (OC < 1 || OC > 2147483647)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv3d_valid_f32", "integer outside wrapper bounds");
  if (KD < 1 || KD > 2147483647)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv3d_valid_f32", "integer outside wrapper bounds");
  if (KH < 1 || KH > 2147483647)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv3d_valid_f32", "integer outside wrapper bounds");
  if (KW < 1 || KW > 2147483647)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv3d_valid_f32", "integer outside wrapper bounds");
  if (OD < 1 || OD > 2147483647)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv3d_valid_f32", "integer outside wrapper bounds");
  if (OH < 1 || OH > 2147483647)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv3d_valid_f32", "integer outside wrapper bounds");
  if (OW < 1 || OW > 2147483647)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv3d_valid_f32", "integer outside wrapper bounds");
  int64_t pv2_bytes_input = sizeof(float);
  if ((int64_t)IC <= 0 || pv2_bytes_input > INT64_MAX / (int64_t)IC)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv3d_valid_f32", "array extent overflow");
  pv2_bytes_input *= (int64_t)IC;
  if ((int64_t)ID <= 0 || pv2_bytes_input > INT64_MAX / (int64_t)ID)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv3d_valid_f32", "array extent overflow");
  pv2_bytes_input *= (int64_t)ID;
  if ((int64_t)IH <= 0 || pv2_bytes_input > INT64_MAX / (int64_t)IH)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv3d_valid_f32", "array extent overflow");
  pv2_bytes_input *= (int64_t)IH;
  if ((int64_t)IW <= 0 || pv2_bytes_input > INT64_MAX / (int64_t)IW)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv3d_valid_f32", "array extent overflow");
  pv2_bytes_input *= (int64_t)IW;
  if (input_stride != 1 || !input || input_bytes < pv2_bytes_input)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv3d_valid_f32", "array layout or capacity mismatch");
  int64_t pv2_bytes_filter = sizeof(float);
  if ((int64_t)OC <= 0 || pv2_bytes_filter > INT64_MAX / (int64_t)OC)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv3d_valid_f32", "array extent overflow");
  pv2_bytes_filter *= (int64_t)OC;
  if ((int64_t)IC <= 0 || pv2_bytes_filter > INT64_MAX / (int64_t)IC)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv3d_valid_f32", "array extent overflow");
  pv2_bytes_filter *= (int64_t)IC;
  if ((int64_t)KD <= 0 || pv2_bytes_filter > INT64_MAX / (int64_t)KD)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv3d_valid_f32", "array extent overflow");
  pv2_bytes_filter *= (int64_t)KD;
  if ((int64_t)KH <= 0 || pv2_bytes_filter > INT64_MAX / (int64_t)KH)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv3d_valid_f32", "array extent overflow");
  pv2_bytes_filter *= (int64_t)KH;
  if ((int64_t)KW <= 0 || pv2_bytes_filter > INT64_MAX / (int64_t)KW)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv3d_valid_f32", "array extent overflow");
  pv2_bytes_filter *= (int64_t)KW;
  if (filter_stride != 1 || !filter || filter_bytes < pv2_bytes_filter)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv3d_valid_f32", "array layout or capacity mismatch");
  int64_t pv2_bytes_output = sizeof(float);
  if ((int64_t)OC <= 0 || pv2_bytes_output > INT64_MAX / (int64_t)OC)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv3d_valid_f32", "array extent overflow");
  pv2_bytes_output *= (int64_t)OC;
  if ((int64_t)OD <= 0 || pv2_bytes_output > INT64_MAX / (int64_t)OD)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv3d_valid_f32", "array extent overflow");
  pv2_bytes_output *= (int64_t)OD;
  if ((int64_t)OH <= 0 || pv2_bytes_output > INT64_MAX / (int64_t)OH)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv3d_valid_f32", "array extent overflow");
  pv2_bytes_output *= (int64_t)OH;
  if ((int64_t)OW <= 0 || pv2_bytes_output > INT64_MAX / (int64_t)OW)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv3d_valid_f32", "array extent overflow");
  pv2_bytes_output *= (int64_t)OW;
  if (output_stride != 1 || !output || output_bytes < pv2_bytes_output)
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv3d_valid_f32", "array layout or capacity mismatch");
  if (polygeist_v2_overlaps(input, input_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv3d_valid_f32", "unsupported output overlap");
  if (polygeist_v2_overlaps(filter, filter_bytes, output, output_bytes))
    polygeist_v2_fatal("cudnnConvolutionForward_cudnn_conv3d_valid_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cudnnConvolutionForward_cudnn_conv3d_valid_f32", "polygeist_reference_conv3d_f32");
  int pv2_k_input = polygeist_v2_map_add(&pv2_map, input, input_bytes, 0);
  int pv2_k_filter = polygeist_v2_map_add(&pv2_map, filter, filter_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_reference_conv3d_f32(IC, ID, IH, IW, OC, KD, KH, KW, OD, OH, OW, (const float *)polygeist_v2_map_dev(&pv2_map, pv2_k_input), (const float *)polygeist_v2_map_dev(&pv2_map, pv2_k_filter), (float *)polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cudnnSoftmaxBackward(int32_t n, const float *dy, int64_t dy_bytes, int32_t incdy, const float *y, int64_t y_bytes, int32_t incy, float *dx, int64_t dx_bytes, int32_t incdx) {
  if (n <= 0) polygeist_v2_fatal("cudnnSoftmaxBackward", "nonpositive descriptor dimension");
  if (incdy != 1 || !dy || dy_bytes < (int64_t)n * sizeof(float))
    polygeist_v2_fatal("cudnnSoftmaxBackward", "descriptor requires a sufficiently large contiguous FP32 row");
  if (incy != 1 || !y || y_bytes < (int64_t)n * sizeof(float))
    polygeist_v2_fatal("cudnnSoftmaxBackward", "descriptor requires a sufficiently large contiguous FP32 row");
  if (incdx != 1 || !dx || dx_bytes < (int64_t)n * sizeof(float))
    polygeist_v2_fatal("cudnnSoftmaxBackward", "descriptor requires a sufficiently large contiguous FP32 row");
  if (polygeist_v2_overlaps(dy, dy_bytes, y, y_bytes))
    polygeist_v2_fatal("cudnnSoftmaxBackward", "descriptor adapter requires nonoverlapping arrays");
  if (polygeist_v2_overlaps(dy, dy_bytes, dx, dx_bytes))
    polygeist_v2_fatal("cudnnSoftmaxBackward", "descriptor adapter requires nonoverlapping arrays");
  if (polygeist_v2_overlaps(y, y_bytes, dx, dx_bytes))
    polygeist_v2_fatal("cudnnSoftmaxBackward", "descriptor adapter requires nonoverlapping arrays");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cudnnSoftmaxBackward", "cudnnSoftmaxBackward");
  int pv2_k_dy = polygeist_v2_map_add(&pv2_map, dy, dy_bytes, 0);
  int pv2_k_y = polygeist_v2_map_add(&pv2_map, y, y_bytes, 0);
  int pv2_k_dx = polygeist_v2_map_add(&pv2_map, dx, dx_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  const float pv2_alpha = 1.0f;
  cudnnTensorDescriptor_t pv2_yDesc = polygeist_v2_tensor4d(CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, n);
  cudnnTensorDescriptor_t pv2_dyDesc = polygeist_v2_tensor4d(CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, n);
  const float pv2_beta = 0.0f;
  cudnnTensorDescriptor_t pv2_dxDesc = polygeist_v2_tensor4d(CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, n);
  POLYGEIST_V2_CUDNN(cudnnSoftmaxBackward(polygeist_v2_cudnn(), CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE, &pv2_alpha, pv2_yDesc, polygeist_v2_map_dev(&pv2_map, pv2_k_y), pv2_dyDesc, polygeist_v2_map_dev(&pv2_map, pv2_k_dy), &pv2_beta, pv2_dxDesc, polygeist_v2_map_dev(&pv2_map, pv2_k_dx)));
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cudnnSoftmaxForward(int32_t n, const float *x, int64_t x_bytes, int32_t incx, float *y, int64_t y_bytes, int32_t incy) {
  if (n <= 0) polygeist_v2_fatal("cudnnSoftmaxForward", "nonpositive descriptor dimension");
  if (incx != 1 || !x || x_bytes < (int64_t)n * sizeof(float))
    polygeist_v2_fatal("cudnnSoftmaxForward", "descriptor requires a sufficiently large contiguous FP32 row");
  if (incy != 1 || !y || y_bytes < (int64_t)n * sizeof(float))
    polygeist_v2_fatal("cudnnSoftmaxForward", "descriptor requires a sufficiently large contiguous FP32 row");
  if (polygeist_v2_overlaps(x, x_bytes, y, y_bytes))
    polygeist_v2_fatal("cudnnSoftmaxForward", "descriptor adapter requires nonoverlapping arrays");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cudnnSoftmaxForward", "cudnnSoftmaxForward");
  int pv2_k_x = polygeist_v2_map_add(&pv2_map, x, x_bytes, 0);
  int pv2_k_y = polygeist_v2_map_add(&pv2_map, y, y_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  const float pv2_alpha = 1.0f;
  cudnnTensorDescriptor_t pv2_xDesc = polygeist_v2_tensor4d(CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, n);
  const float pv2_beta = 0.0f;
  cudnnTensorDescriptor_t pv2_yDesc = polygeist_v2_tensor4d(CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, n);
  POLYGEIST_V2_CUDNN(cudnnSoftmaxForward(polygeist_v2_cudnn(), CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE, &pv2_alpha, pv2_xDesc, polygeist_v2_map_dev(&pv2_map, pv2_k_x), &pv2_beta, pv2_yDesc, polygeist_v2_map_dev(&pv2_map, pv2_k_y)));
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cusolverDnDpotrf(int32_t uplo, int32_t n, double *A, int64_t A_bytes, int32_t lda) {
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cusolverDnDpotrf", "cusolverDnDpotrf");
  int pv2_k_A = polygeist_v2_map_add(&pv2_map, A, A_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  int Lwork = 0;
  int *devInfo = polygeist_v2_devinfo();
  POLYGEIST_V2_CUSOLVER(cusolverDnDpotrf_bufferSize(polygeist_v2_cusolver(), (cublasFillMode_t)uplo, n, (double*)polygeist_v2_map_dev(&pv2_map, pv2_k_A), lda, &Lwork), "cusolverDnDpotrf_bufferSize");
  double* Workspace = (double*)polygeist_v2_workspace((size_t)Lwork * sizeof(double));
  POLYGEIST_V2_CUSOLVER(cusolverDnDpotrf(polygeist_v2_cusolver(), (cublasFillMode_t)uplo, n, (double*)polygeist_v2_map_dev(&pv2_map, pv2_k_A), lda, Workspace, Lwork, devInfo), "cusolverDnDpotrf");
  polygeist_v2_map_to_host(&pv2_map);
  polygeist_v2_check_devinfo("cusolverDnDpotrf", devInfo);
}

void polygeist_v2_cutensorPermute_aten_acos_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, float *output, int64_t output_bytes, int32_t output_stride) {
  if (n <= 0) polygeist_v2_fatal("cutensorPermute_aten_acos_f32", "nonpositive dimension");
  if (a_stride != 1 || !a || a_bytes < (int64_t)n * sizeof(float))
    polygeist_v2_fatal("cutensorPermute_aten_acos_f32", "runtime adapter requires a contiguous FP32 row");
  if (output_stride != 1 || !output || output_bytes < (int64_t)n * sizeof(float))
    polygeist_v2_fatal("cutensorPermute_aten_acos_f32", "runtime adapter requires a contiguous FP32 row");
  if (polygeist_v2_overlaps(a, a_bytes, output, output_bytes))
    polygeist_v2_fatal("cutensorPermute_aten_acos_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cutensorPermute_aten_acos_f32", "cutensorPermute");
  int pv2_k_a = polygeist_v2_map_add(&pv2_map, a, a_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cutensor_unary_f32(POLYGEIST_CUTENSOR_UNARY_ACOS, n, polygeist_v2_map_dev(&pv2_map, pv2_k_a), polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cutensorPermute_aten_acosh_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, float *output, int64_t output_bytes, int32_t output_stride) {
  if (n <= 0) polygeist_v2_fatal("cutensorPermute_aten_acosh_f32", "nonpositive dimension");
  if (a_stride != 1 || !a || a_bytes < (int64_t)n * sizeof(float))
    polygeist_v2_fatal("cutensorPermute_aten_acosh_f32", "runtime adapter requires a contiguous FP32 row");
  if (output_stride != 1 || !output || output_bytes < (int64_t)n * sizeof(float))
    polygeist_v2_fatal("cutensorPermute_aten_acosh_f32", "runtime adapter requires a contiguous FP32 row");
  if (polygeist_v2_overlaps(a, a_bytes, output, output_bytes))
    polygeist_v2_fatal("cutensorPermute_aten_acosh_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cutensorPermute_aten_acosh_f32", "cutensorPermute");
  int pv2_k_a = polygeist_v2_map_add(&pv2_map, a, a_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cutensor_unary_f32(POLYGEIST_CUTENSOR_UNARY_ACOSH, n, polygeist_v2_map_dev(&pv2_map, pv2_k_a), polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cutensorPermute_aten_asin_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, float *output, int64_t output_bytes, int32_t output_stride) {
  if (n <= 0) polygeist_v2_fatal("cutensorPermute_aten_asin_f32", "nonpositive dimension");
  if (a_stride != 1 || !a || a_bytes < (int64_t)n * sizeof(float))
    polygeist_v2_fatal("cutensorPermute_aten_asin_f32", "runtime adapter requires a contiguous FP32 row");
  if (output_stride != 1 || !output || output_bytes < (int64_t)n * sizeof(float))
    polygeist_v2_fatal("cutensorPermute_aten_asin_f32", "runtime adapter requires a contiguous FP32 row");
  if (polygeist_v2_overlaps(a, a_bytes, output, output_bytes))
    polygeist_v2_fatal("cutensorPermute_aten_asin_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cutensorPermute_aten_asin_f32", "cutensorPermute");
  int pv2_k_a = polygeist_v2_map_add(&pv2_map, a, a_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cutensor_unary_f32(POLYGEIST_CUTENSOR_UNARY_ASIN, n, polygeist_v2_map_dev(&pv2_map, pv2_k_a), polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cutensorPermute_aten_asinh_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, float *output, int64_t output_bytes, int32_t output_stride) {
  if (n <= 0) polygeist_v2_fatal("cutensorPermute_aten_asinh_f32", "nonpositive dimension");
  if (a_stride != 1 || !a || a_bytes < (int64_t)n * sizeof(float))
    polygeist_v2_fatal("cutensorPermute_aten_asinh_f32", "runtime adapter requires a contiguous FP32 row");
  if (output_stride != 1 || !output || output_bytes < (int64_t)n * sizeof(float))
    polygeist_v2_fatal("cutensorPermute_aten_asinh_f32", "runtime adapter requires a contiguous FP32 row");
  if (polygeist_v2_overlaps(a, a_bytes, output, output_bytes))
    polygeist_v2_fatal("cutensorPermute_aten_asinh_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cutensorPermute_aten_asinh_f32", "cutensorPermute");
  int pv2_k_a = polygeist_v2_map_add(&pv2_map, a, a_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cutensor_unary_f32(POLYGEIST_CUTENSOR_UNARY_ASINH, n, polygeist_v2_map_dev(&pv2_map, pv2_k_a), polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cutensorPermute_aten_atan_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, float *output, int64_t output_bytes, int32_t output_stride) {
  if (n <= 0) polygeist_v2_fatal("cutensorPermute_aten_atan_f32", "nonpositive dimension");
  if (a_stride != 1 || !a || a_bytes < (int64_t)n * sizeof(float))
    polygeist_v2_fatal("cutensorPermute_aten_atan_f32", "runtime adapter requires a contiguous FP32 row");
  if (output_stride != 1 || !output || output_bytes < (int64_t)n * sizeof(float))
    polygeist_v2_fatal("cutensorPermute_aten_atan_f32", "runtime adapter requires a contiguous FP32 row");
  if (polygeist_v2_overlaps(a, a_bytes, output, output_bytes))
    polygeist_v2_fatal("cutensorPermute_aten_atan_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cutensorPermute_aten_atan_f32", "cutensorPermute");
  int pv2_k_a = polygeist_v2_map_add(&pv2_map, a, a_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cutensor_unary_f32(POLYGEIST_CUTENSOR_UNARY_ATAN, n, polygeist_v2_map_dev(&pv2_map, pv2_k_a), polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cutensorPermute_aten_atanh_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, float *output, int64_t output_bytes, int32_t output_stride) {
  if (n <= 0) polygeist_v2_fatal("cutensorPermute_aten_atanh_f32", "nonpositive dimension");
  if (a_stride != 1 || !a || a_bytes < (int64_t)n * sizeof(float))
    polygeist_v2_fatal("cutensorPermute_aten_atanh_f32", "runtime adapter requires a contiguous FP32 row");
  if (output_stride != 1 || !output || output_bytes < (int64_t)n * sizeof(float))
    polygeist_v2_fatal("cutensorPermute_aten_atanh_f32", "runtime adapter requires a contiguous FP32 row");
  if (polygeist_v2_overlaps(a, a_bytes, output, output_bytes))
    polygeist_v2_fatal("cutensorPermute_aten_atanh_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cutensorPermute_aten_atanh_f32", "cutensorPermute");
  int pv2_k_a = polygeist_v2_map_add(&pv2_map, a, a_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cutensor_unary_f32(POLYGEIST_CUTENSOR_UNARY_ATANH, n, polygeist_v2_map_dev(&pv2_map, pv2_k_a), polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cutensorPermute_aten_cosh_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, float *output, int64_t output_bytes, int32_t output_stride) {
  if (n <= 0) polygeist_v2_fatal("cutensorPermute_aten_cosh_f32", "nonpositive dimension");
  if (a_stride != 1 || !a || a_bytes < (int64_t)n * sizeof(float))
    polygeist_v2_fatal("cutensorPermute_aten_cosh_f32", "runtime adapter requires a contiguous FP32 row");
  if (output_stride != 1 || !output || output_bytes < (int64_t)n * sizeof(float))
    polygeist_v2_fatal("cutensorPermute_aten_cosh_f32", "runtime adapter requires a contiguous FP32 row");
  if (polygeist_v2_overlaps(a, a_bytes, output, output_bytes))
    polygeist_v2_fatal("cutensorPermute_aten_cosh_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cutensorPermute_aten_cosh_f32", "cutensorPermute");
  int pv2_k_a = polygeist_v2_map_add(&pv2_map, a, a_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cutensor_unary_f32(POLYGEIST_CUTENSOR_UNARY_COSH, n, polygeist_v2_map_dev(&pv2_map, pv2_k_a), polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cutensorPermute_aten_sinh_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, float *output, int64_t output_bytes, int32_t output_stride) {
  if (n <= 0) polygeist_v2_fatal("cutensorPermute_aten_sinh_f32", "nonpositive dimension");
  if (a_stride != 1 || !a || a_bytes < (int64_t)n * sizeof(float))
    polygeist_v2_fatal("cutensorPermute_aten_sinh_f32", "runtime adapter requires a contiguous FP32 row");
  if (output_stride != 1 || !output || output_bytes < (int64_t)n * sizeof(float))
    polygeist_v2_fatal("cutensorPermute_aten_sinh_f32", "runtime adapter requires a contiguous FP32 row");
  if (polygeist_v2_overlaps(a, a_bytes, output, output_bytes))
    polygeist_v2_fatal("cutensorPermute_aten_sinh_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cutensorPermute_aten_sinh_f32", "cutensorPermute");
  int pv2_k_a = polygeist_v2_map_add(&pv2_map, a, a_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cutensor_unary_f32(POLYGEIST_CUTENSOR_UNARY_SINH, n, polygeist_v2_map_dev(&pv2_map, pv2_k_a), polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cutensorPermute_aten_tan_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, float *output, int64_t output_bytes, int32_t output_stride) {
  if (n <= 0) polygeist_v2_fatal("cutensorPermute_aten_tan_f32", "nonpositive dimension");
  if (a_stride != 1 || !a || a_bytes < (int64_t)n * sizeof(float))
    polygeist_v2_fatal("cutensorPermute_aten_tan_f32", "runtime adapter requires a contiguous FP32 row");
  if (output_stride != 1 || !output || output_bytes < (int64_t)n * sizeof(float))
    polygeist_v2_fatal("cutensorPermute_aten_tan_f32", "runtime adapter requires a contiguous FP32 row");
  if (polygeist_v2_overlaps(a, a_bytes, output, output_bytes))
    polygeist_v2_fatal("cutensorPermute_aten_tan_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cutensorPermute_aten_tan_f32", "cutensorPermute");
  int pv2_k_a = polygeist_v2_map_add(&pv2_map, a, a_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_cutensor_unary_f32(POLYGEIST_CUTENSOR_UNARY_TAN, n, polygeist_v2_map_dev(&pv2_map, pv2_k_a), polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_fill_n_thrust_fill_f32(int32_t n, float value, float *output, int64_t output_bytes, int32_t output_stride) {
  if (n < 1 || n > 2147483647)
    polygeist_v2_fatal("fill_n_thrust_fill_f32", "integer outside wrapper bounds");
  int64_t pv2_bytes_output = sizeof(float);
  if ((int64_t)n <= 0 || pv2_bytes_output > INT64_MAX / (int64_t)n)
    polygeist_v2_fatal("fill_n_thrust_fill_f32", "array extent overflow");
  pv2_bytes_output *= (int64_t)n;
  if (output_stride != 1 || !output || output_bytes < pv2_bytes_output)
    polygeist_v2_fatal("fill_n_thrust_fill_f32", "array layout or capacity mismatch");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "fill_n_thrust_fill_f32", "polygeist_thrust_fill_f32");
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_thrust_fill_f32(n, value, (float *)polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

extern __typeof__(polygeist_thrust_cos_f32) polygeist_thrust_cos_f32 __attribute__((weak));
void polygeist_v2_transform_aten_cos_f32(int32_t n, const float *x, int64_t x_bytes, int32_t x_stride, float *output, int64_t output_bytes, int32_t output_stride) {
  if (!polygeist_thrust_cos_f32)
    polygeist_v2_fatal("transform_aten_cos_f32", "optional vendor entry is not linked");
  if (n < 1 || n > 2147483647)
    polygeist_v2_fatal("transform_aten_cos_f32", "integer outside wrapper bounds");
  int64_t pv2_bytes_x = sizeof(float);
  if ((int64_t)n <= 0 || pv2_bytes_x > INT64_MAX / (int64_t)n)
    polygeist_v2_fatal("transform_aten_cos_f32", "array extent overflow");
  pv2_bytes_x *= (int64_t)n;
  if (x_stride != 1 || !x || x_bytes < pv2_bytes_x)
    polygeist_v2_fatal("transform_aten_cos_f32", "array layout or capacity mismatch");
  int64_t pv2_bytes_output = sizeof(float);
  if ((int64_t)n <= 0 || pv2_bytes_output > INT64_MAX / (int64_t)n)
    polygeist_v2_fatal("transform_aten_cos_f32", "array extent overflow");
  pv2_bytes_output *= (int64_t)n;
  if (output_stride != 1 || !output || output_bytes < pv2_bytes_output)
    polygeist_v2_fatal("transform_aten_cos_f32", "array layout or capacity mismatch");
  if (polygeist_v2_overlaps(x, x_bytes, output, output_bytes))
    polygeist_v2_fatal("transform_aten_cos_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "transform_aten_cos_f32", "polygeist_thrust_cos_f32");
  int pv2_k_x = polygeist_v2_map_add(&pv2_map, x, x_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_thrust_cos_f32(n, (const float *)polygeist_v2_map_dev(&pv2_map, pv2_k_x), (float *)polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

extern __typeof__(polygeist_thrust_sin_f32) polygeist_thrust_sin_f32 __attribute__((weak));
void polygeist_v2_transform_aten_sin_f32(int32_t n, const float *x, int64_t x_bytes, int32_t x_stride, float *output, int64_t output_bytes, int32_t output_stride) {
  if (!polygeist_thrust_sin_f32)
    polygeist_v2_fatal("transform_aten_sin_f32", "optional vendor entry is not linked");
  if (n < 1 || n > 2147483647)
    polygeist_v2_fatal("transform_aten_sin_f32", "integer outside wrapper bounds");
  int64_t pv2_bytes_x = sizeof(float);
  if ((int64_t)n <= 0 || pv2_bytes_x > INT64_MAX / (int64_t)n)
    polygeist_v2_fatal("transform_aten_sin_f32", "array extent overflow");
  pv2_bytes_x *= (int64_t)n;
  if (x_stride != 1 || !x || x_bytes < pv2_bytes_x)
    polygeist_v2_fatal("transform_aten_sin_f32", "array layout or capacity mismatch");
  int64_t pv2_bytes_output = sizeof(float);
  if ((int64_t)n <= 0 || pv2_bytes_output > INT64_MAX / (int64_t)n)
    polygeist_v2_fatal("transform_aten_sin_f32", "array extent overflow");
  pv2_bytes_output *= (int64_t)n;
  if (output_stride != 1 || !output || output_bytes < pv2_bytes_output)
    polygeist_v2_fatal("transform_aten_sin_f32", "array layout or capacity mismatch");
  if (polygeist_v2_overlaps(x, x_bytes, output, output_bytes))
    polygeist_v2_fatal("transform_aten_sin_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "transform_aten_sin_f32", "polygeist_thrust_sin_f32");
  int pv2_k_x = polygeist_v2_map_add(&pv2_map, x, x_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_thrust_sin_f32(n, (const float *)polygeist_v2_map_dev(&pv2_map, pv2_k_x), (float *)polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_transform_thrust_divide_f32(int32_t n, const float *x, int64_t x_bytes, int32_t x_stride, const float *y, int64_t y_bytes, int32_t y_stride, float *output, int64_t output_bytes, int32_t output_stride) {
  if (n < 1 || n > 2147483647)
    polygeist_v2_fatal("transform_thrust_divide_f32", "integer outside wrapper bounds");
  int64_t pv2_bytes_x = sizeof(float);
  if ((int64_t)n <= 0 || pv2_bytes_x > INT64_MAX / (int64_t)n)
    polygeist_v2_fatal("transform_thrust_divide_f32", "array extent overflow");
  pv2_bytes_x *= (int64_t)n;
  if (x_stride != 1 || !x || x_bytes < pv2_bytes_x)
    polygeist_v2_fatal("transform_thrust_divide_f32", "array layout or capacity mismatch");
  int64_t pv2_bytes_y = sizeof(float);
  if ((int64_t)n <= 0 || pv2_bytes_y > INT64_MAX / (int64_t)n)
    polygeist_v2_fatal("transform_thrust_divide_f32", "array extent overflow");
  pv2_bytes_y *= (int64_t)n;
  if (y_stride != 1 || !y || y_bytes < pv2_bytes_y)
    polygeist_v2_fatal("transform_thrust_divide_f32", "array layout or capacity mismatch");
  int64_t pv2_bytes_output = sizeof(float);
  if ((int64_t)n <= 0 || pv2_bytes_output > INT64_MAX / (int64_t)n)
    polygeist_v2_fatal("transform_thrust_divide_f32", "array extent overflow");
  pv2_bytes_output *= (int64_t)n;
  if (output_stride != 1 || !output || output_bytes < pv2_bytes_output)
    polygeist_v2_fatal("transform_thrust_divide_f32", "array layout or capacity mismatch");
  if (polygeist_v2_overlaps(x, x_bytes, output, output_bytes))
    polygeist_v2_fatal("transform_thrust_divide_f32", "unsupported output overlap");
  if (polygeist_v2_overlaps(y, y_bytes, output, output_bytes))
    polygeist_v2_fatal("transform_thrust_divide_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "transform_thrust_divide_f32", "polygeist_thrust_divide_f32");
  int pv2_k_x = polygeist_v2_map_add(&pv2_map, x, x_bytes, 0);
  int pv2_k_y = polygeist_v2_map_add(&pv2_map, y, y_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_thrust_divide_f32(n, (const float *)polygeist_v2_map_dev(&pv2_map, pv2_k_x), (const float *)polygeist_v2_map_dev(&pv2_map, pv2_k_y), (float *)polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_transform_thrust_divide_scalar_left_f32(int32_t n, float x, const float *y, int64_t y_bytes, int32_t y_stride, float *output, int64_t output_bytes, int32_t output_stride) {
  if (n < 1 || n > 2147483647)
    polygeist_v2_fatal("transform_thrust_divide_scalar_left_f32", "integer outside wrapper bounds");
  int64_t pv2_bytes_y = sizeof(float);
  if ((int64_t)n <= 0 || pv2_bytes_y > INT64_MAX / (int64_t)n)
    polygeist_v2_fatal("transform_thrust_divide_scalar_left_f32", "array extent overflow");
  pv2_bytes_y *= (int64_t)n;
  if (y_stride != 1 || !y || y_bytes < pv2_bytes_y)
    polygeist_v2_fatal("transform_thrust_divide_scalar_left_f32", "array layout or capacity mismatch");
  int64_t pv2_bytes_output = sizeof(float);
  if ((int64_t)n <= 0 || pv2_bytes_output > INT64_MAX / (int64_t)n)
    polygeist_v2_fatal("transform_thrust_divide_scalar_left_f32", "array extent overflow");
  pv2_bytes_output *= (int64_t)n;
  if (output_stride != 1 || !output || output_bytes < pv2_bytes_output)
    polygeist_v2_fatal("transform_thrust_divide_scalar_left_f32", "array layout or capacity mismatch");
  if (polygeist_v2_overlaps(y, y_bytes, output, output_bytes))
    polygeist_v2_fatal("transform_thrust_divide_scalar_left_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "transform_thrust_divide_scalar_left_f32", "polygeist_thrust_divide_scalar_left_f32");
  int pv2_k_y = polygeist_v2_map_add(&pv2_map, y, y_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_thrust_divide_scalar_left_f32(n, x, (const float *)polygeist_v2_map_dev(&pv2_map, pv2_k_y), (float *)polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_transform_thrust_divide_scalar_right_f32(int32_t n, const float *x, int64_t x_bytes, int32_t x_stride, float y, float *output, int64_t output_bytes, int32_t output_stride) {
  if (n < 1 || n > 2147483647)
    polygeist_v2_fatal("transform_thrust_divide_scalar_right_f32", "integer outside wrapper bounds");
  int64_t pv2_bytes_x = sizeof(float);
  if ((int64_t)n <= 0 || pv2_bytes_x > INT64_MAX / (int64_t)n)
    polygeist_v2_fatal("transform_thrust_divide_scalar_right_f32", "array extent overflow");
  pv2_bytes_x *= (int64_t)n;
  if (x_stride != 1 || !x || x_bytes < pv2_bytes_x)
    polygeist_v2_fatal("transform_thrust_divide_scalar_right_f32", "array layout or capacity mismatch");
  int64_t pv2_bytes_output = sizeof(float);
  if ((int64_t)n <= 0 || pv2_bytes_output > INT64_MAX / (int64_t)n)
    polygeist_v2_fatal("transform_thrust_divide_scalar_right_f32", "array extent overflow");
  pv2_bytes_output *= (int64_t)n;
  if (output_stride != 1 || !output || output_bytes < pv2_bytes_output)
    polygeist_v2_fatal("transform_thrust_divide_scalar_right_f32", "array layout or capacity mismatch");
  if (polygeist_v2_overlaps(x, x_bytes, output, output_bytes))
    polygeist_v2_fatal("transform_thrust_divide_scalar_right_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "transform_thrust_divide_scalar_right_f32", "polygeist_thrust_divide_scalar_right_f32");
  int pv2_k_x = polygeist_v2_map_add(&pv2_map, x, x_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_thrust_divide_scalar_right_f32(n, (const float *)polygeist_v2_map_dev(&pv2_map, pv2_k_x), y, (float *)polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_transform_thrust_divide_scalars_f32(int32_t n, float x, float y, float *output, int64_t output_bytes, int32_t output_stride) {
  if (n < 1 || n > 2147483647)
    polygeist_v2_fatal("transform_thrust_divide_scalars_f32", "integer outside wrapper bounds");
  int64_t pv2_bytes_output = sizeof(float);
  if ((int64_t)n <= 0 || pv2_bytes_output > INT64_MAX / (int64_t)n)
    polygeist_v2_fatal("transform_thrust_divide_scalars_f32", "array extent overflow");
  pv2_bytes_output *= (int64_t)n;
  if (output_stride != 1 || !output || output_bytes < pv2_bytes_output)
    polygeist_v2_fatal("transform_thrust_divide_scalars_f32", "array layout or capacity mismatch");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "transform_thrust_divide_scalars_f32", "polygeist_thrust_divide_scalars_f32");
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_thrust_divide_scalars_f32(n, x, y, (float *)polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_transform_thrust_log1p_f32(int32_t n, const float *input, int64_t input_bytes, int32_t input_stride, float *output, int64_t output_bytes, int32_t output_stride) {
  if (n < 1 || n > 2147483647)
    polygeist_v2_fatal("transform_thrust_log1p_f32", "integer outside wrapper bounds");
  int64_t pv2_bytes_input = sizeof(float);
  if ((int64_t)n <= 0 || pv2_bytes_input > INT64_MAX / (int64_t)n)
    polygeist_v2_fatal("transform_thrust_log1p_f32", "array extent overflow");
  pv2_bytes_input *= (int64_t)n;
  if (input_stride != 1 || !input || input_bytes < pv2_bytes_input)
    polygeist_v2_fatal("transform_thrust_log1p_f32", "array layout or capacity mismatch");
  int64_t pv2_bytes_output = sizeof(float);
  if ((int64_t)n <= 0 || pv2_bytes_output > INT64_MAX / (int64_t)n)
    polygeist_v2_fatal("transform_thrust_log1p_f32", "array extent overflow");
  pv2_bytes_output *= (int64_t)n;
  if (output_stride != 1 || !output || output_bytes < pv2_bytes_output)
    polygeist_v2_fatal("transform_thrust_log1p_f32", "array layout or capacity mismatch");
  if (polygeist_v2_overlaps(input, input_bytes, output, output_bytes))
    polygeist_v2_fatal("transform_thrust_log1p_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "transform_thrust_log1p_f32", "polygeist_thrust_log1p_f32");
  int pv2_k_input = polygeist_v2_map_add(&pv2_map, input, input_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_thrust_log1p_f32(n, (const float *)polygeist_v2_map_dev(&pv2_map, pv2_k_input), (float *)polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_transform_thrust_minus_f32(int32_t n, const float *x, int64_t x_bytes, int32_t x_stride, const float *y, int64_t y_bytes, int32_t y_stride, float *output, int64_t output_bytes, int32_t output_stride) {
  if (n < 1 || n > 2147483647)
    polygeist_v2_fatal("transform_thrust_minus_f32", "integer outside wrapper bounds");
  int64_t pv2_bytes_x = sizeof(float);
  if ((int64_t)n <= 0 || pv2_bytes_x > INT64_MAX / (int64_t)n)
    polygeist_v2_fatal("transform_thrust_minus_f32", "array extent overflow");
  pv2_bytes_x *= (int64_t)n;
  if (x_stride != 1 || !x || x_bytes < pv2_bytes_x)
    polygeist_v2_fatal("transform_thrust_minus_f32", "array layout or capacity mismatch");
  int64_t pv2_bytes_y = sizeof(float);
  if ((int64_t)n <= 0 || pv2_bytes_y > INT64_MAX / (int64_t)n)
    polygeist_v2_fatal("transform_thrust_minus_f32", "array extent overflow");
  pv2_bytes_y *= (int64_t)n;
  if (y_stride != 1 || !y || y_bytes < pv2_bytes_y)
    polygeist_v2_fatal("transform_thrust_minus_f32", "array layout or capacity mismatch");
  int64_t pv2_bytes_output = sizeof(float);
  if ((int64_t)n <= 0 || pv2_bytes_output > INT64_MAX / (int64_t)n)
    polygeist_v2_fatal("transform_thrust_minus_f32", "array extent overflow");
  pv2_bytes_output *= (int64_t)n;
  if (output_stride != 1 || !output || output_bytes < pv2_bytes_output)
    polygeist_v2_fatal("transform_thrust_minus_f32", "array layout or capacity mismatch");
  if (polygeist_v2_overlaps(x, x_bytes, output, output_bytes))
    polygeist_v2_fatal("transform_thrust_minus_f32", "unsupported output overlap");
  if (polygeist_v2_overlaps(y, y_bytes, output, output_bytes))
    polygeist_v2_fatal("transform_thrust_minus_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "transform_thrust_minus_f32", "polygeist_thrust_minus_f32");
  int pv2_k_x = polygeist_v2_map_add(&pv2_map, x, x_bytes, 0);
  int pv2_k_y = polygeist_v2_map_add(&pv2_map, y, y_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_thrust_minus_f32(n, (const float *)polygeist_v2_map_dev(&pv2_map, pv2_k_x), (const float *)polygeist_v2_map_dev(&pv2_map, pv2_k_y), (float *)polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_transform_thrust_minus_scalar_left_f32(int32_t n, float x, const float *y, int64_t y_bytes, int32_t y_stride, float *output, int64_t output_bytes, int32_t output_stride) {
  if (n < 1 || n > 2147483647)
    polygeist_v2_fatal("transform_thrust_minus_scalar_left_f32", "integer outside wrapper bounds");
  int64_t pv2_bytes_y = sizeof(float);
  if ((int64_t)n <= 0 || pv2_bytes_y > INT64_MAX / (int64_t)n)
    polygeist_v2_fatal("transform_thrust_minus_scalar_left_f32", "array extent overflow");
  pv2_bytes_y *= (int64_t)n;
  if (y_stride != 1 || !y || y_bytes < pv2_bytes_y)
    polygeist_v2_fatal("transform_thrust_minus_scalar_left_f32", "array layout or capacity mismatch");
  int64_t pv2_bytes_output = sizeof(float);
  if ((int64_t)n <= 0 || pv2_bytes_output > INT64_MAX / (int64_t)n)
    polygeist_v2_fatal("transform_thrust_minus_scalar_left_f32", "array extent overflow");
  pv2_bytes_output *= (int64_t)n;
  if (output_stride != 1 || !output || output_bytes < pv2_bytes_output)
    polygeist_v2_fatal("transform_thrust_minus_scalar_left_f32", "array layout or capacity mismatch");
  if (polygeist_v2_overlaps(y, y_bytes, output, output_bytes))
    polygeist_v2_fatal("transform_thrust_minus_scalar_left_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "transform_thrust_minus_scalar_left_f32", "polygeist_thrust_minus_scalar_left_f32");
  int pv2_k_y = polygeist_v2_map_add(&pv2_map, y, y_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_thrust_minus_scalar_left_f32(n, x, (const float *)polygeist_v2_map_dev(&pv2_map, pv2_k_y), (float *)polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_transform_thrust_minus_scalar_right_f32(int32_t n, const float *x, int64_t x_bytes, int32_t x_stride, float y, float *output, int64_t output_bytes, int32_t output_stride) {
  if (n < 1 || n > 2147483647)
    polygeist_v2_fatal("transform_thrust_minus_scalar_right_f32", "integer outside wrapper bounds");
  int64_t pv2_bytes_x = sizeof(float);
  if ((int64_t)n <= 0 || pv2_bytes_x > INT64_MAX / (int64_t)n)
    polygeist_v2_fatal("transform_thrust_minus_scalar_right_f32", "array extent overflow");
  pv2_bytes_x *= (int64_t)n;
  if (x_stride != 1 || !x || x_bytes < pv2_bytes_x)
    polygeist_v2_fatal("transform_thrust_minus_scalar_right_f32", "array layout or capacity mismatch");
  int64_t pv2_bytes_output = sizeof(float);
  if ((int64_t)n <= 0 || pv2_bytes_output > INT64_MAX / (int64_t)n)
    polygeist_v2_fatal("transform_thrust_minus_scalar_right_f32", "array extent overflow");
  pv2_bytes_output *= (int64_t)n;
  if (output_stride != 1 || !output || output_bytes < pv2_bytes_output)
    polygeist_v2_fatal("transform_thrust_minus_scalar_right_f32", "array layout or capacity mismatch");
  if (polygeist_v2_overlaps(x, x_bytes, output, output_bytes))
    polygeist_v2_fatal("transform_thrust_minus_scalar_right_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "transform_thrust_minus_scalar_right_f32", "polygeist_thrust_minus_scalar_right_f32");
  int pv2_k_x = polygeist_v2_map_add(&pv2_map, x, x_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_thrust_minus_scalar_right_f32(n, (const float *)polygeist_v2_map_dev(&pv2_map, pv2_k_x), y, (float *)polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_transform_thrust_minus_scalar_right_i32(int32_t n, const int *x, int64_t x_bytes, int32_t x_stride, int32_t y, int *output, int64_t output_bytes, int32_t output_stride) {
  if (n < 1 || n > 2147483647)
    polygeist_v2_fatal("transform_thrust_minus_scalar_right_i32", "integer outside wrapper bounds");
  if (y < -2147483648 || y > 2147483647)
    polygeist_v2_fatal("transform_thrust_minus_scalar_right_i32", "integer outside wrapper bounds");
  int64_t pv2_bytes_x = sizeof(int);
  if ((int64_t)n <= 0 || pv2_bytes_x > INT64_MAX / (int64_t)n)
    polygeist_v2_fatal("transform_thrust_minus_scalar_right_i32", "array extent overflow");
  pv2_bytes_x *= (int64_t)n;
  if (x_stride != 1 || !x || x_bytes < pv2_bytes_x)
    polygeist_v2_fatal("transform_thrust_minus_scalar_right_i32", "array layout or capacity mismatch");
  int64_t pv2_bytes_output = sizeof(int);
  if ((int64_t)n <= 0 || pv2_bytes_output > INT64_MAX / (int64_t)n)
    polygeist_v2_fatal("transform_thrust_minus_scalar_right_i32", "array extent overflow");
  pv2_bytes_output *= (int64_t)n;
  if (output_stride != 1 || !output || output_bytes < pv2_bytes_output)
    polygeist_v2_fatal("transform_thrust_minus_scalar_right_i32", "array layout or capacity mismatch");
  if (polygeist_v2_overlaps(x, x_bytes, output, output_bytes))
    polygeist_v2_fatal("transform_thrust_minus_scalar_right_i32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "transform_thrust_minus_scalar_right_i32", "polygeist_thrust_minus_scalar_right_i32");
  int pv2_k_x = polygeist_v2_map_add(&pv2_map, x, x_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_thrust_minus_scalar_right_i32(n, (const int *)polygeist_v2_map_dev(&pv2_map, pv2_k_x), y, (int *)polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_transform_thrust_minus_scalars_f32(int32_t n, float x, float y, float *output, int64_t output_bytes, int32_t output_stride) {
  if (n < 1 || n > 2147483647)
    polygeist_v2_fatal("transform_thrust_minus_scalars_f32", "integer outside wrapper bounds");
  int64_t pv2_bytes_output = sizeof(float);
  if ((int64_t)n <= 0 || pv2_bytes_output > INT64_MAX / (int64_t)n)
    polygeist_v2_fatal("transform_thrust_minus_scalars_f32", "array extent overflow");
  pv2_bytes_output *= (int64_t)n;
  if (output_stride != 1 || !output || output_bytes < pv2_bytes_output)
    polygeist_v2_fatal("transform_thrust_minus_scalars_f32", "array layout or capacity mismatch");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "transform_thrust_minus_scalars_f32", "polygeist_thrust_minus_scalars_f32");
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_thrust_minus_scalars_f32(n, x, y, (float *)polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_transform_thrust_multiplies_f32(int32_t n, const float *x, int64_t x_bytes, int32_t x_stride, const float *y, int64_t y_bytes, int32_t y_stride, float *output, int64_t output_bytes, int32_t output_stride) {
  if (n < 1 || n > 2147483647)
    polygeist_v2_fatal("transform_thrust_multiplies_f32", "integer outside wrapper bounds");
  int64_t pv2_bytes_x = sizeof(float);
  if ((int64_t)n <= 0 || pv2_bytes_x > INT64_MAX / (int64_t)n)
    polygeist_v2_fatal("transform_thrust_multiplies_f32", "array extent overflow");
  pv2_bytes_x *= (int64_t)n;
  if (x_stride != 1 || !x || x_bytes < pv2_bytes_x)
    polygeist_v2_fatal("transform_thrust_multiplies_f32", "array layout or capacity mismatch");
  int64_t pv2_bytes_y = sizeof(float);
  if ((int64_t)n <= 0 || pv2_bytes_y > INT64_MAX / (int64_t)n)
    polygeist_v2_fatal("transform_thrust_multiplies_f32", "array extent overflow");
  pv2_bytes_y *= (int64_t)n;
  if (y_stride != 1 || !y || y_bytes < pv2_bytes_y)
    polygeist_v2_fatal("transform_thrust_multiplies_f32", "array layout or capacity mismatch");
  int64_t pv2_bytes_output = sizeof(float);
  if ((int64_t)n <= 0 || pv2_bytes_output > INT64_MAX / (int64_t)n)
    polygeist_v2_fatal("transform_thrust_multiplies_f32", "array extent overflow");
  pv2_bytes_output *= (int64_t)n;
  if (output_stride != 1 || !output || output_bytes < pv2_bytes_output)
    polygeist_v2_fatal("transform_thrust_multiplies_f32", "array layout or capacity mismatch");
  if (polygeist_v2_overlaps(x, x_bytes, output, output_bytes))
    polygeist_v2_fatal("transform_thrust_multiplies_f32", "unsupported output overlap");
  if (polygeist_v2_overlaps(y, y_bytes, output, output_bytes))
    polygeist_v2_fatal("transform_thrust_multiplies_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "transform_thrust_multiplies_f32", "polygeist_thrust_multiplies_f32");
  int pv2_k_x = polygeist_v2_map_add(&pv2_map, x, x_bytes, 0);
  int pv2_k_y = polygeist_v2_map_add(&pv2_map, y, y_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_thrust_multiplies_f32(n, (const float *)polygeist_v2_map_dev(&pv2_map, pv2_k_x), (const float *)polygeist_v2_map_dev(&pv2_map, pv2_k_y), (float *)polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_transform_thrust_multiplies_scalar_left_f32(int32_t n, float x, const float *y, int64_t y_bytes, int32_t y_stride, float *output, int64_t output_bytes, int32_t output_stride) {
  if (n < 1 || n > 2147483647)
    polygeist_v2_fatal("transform_thrust_multiplies_scalar_left_f32", "integer outside wrapper bounds");
  int64_t pv2_bytes_y = sizeof(float);
  if ((int64_t)n <= 0 || pv2_bytes_y > INT64_MAX / (int64_t)n)
    polygeist_v2_fatal("transform_thrust_multiplies_scalar_left_f32", "array extent overflow");
  pv2_bytes_y *= (int64_t)n;
  if (y_stride != 1 || !y || y_bytes < pv2_bytes_y)
    polygeist_v2_fatal("transform_thrust_multiplies_scalar_left_f32", "array layout or capacity mismatch");
  int64_t pv2_bytes_output = sizeof(float);
  if ((int64_t)n <= 0 || pv2_bytes_output > INT64_MAX / (int64_t)n)
    polygeist_v2_fatal("transform_thrust_multiplies_scalar_left_f32", "array extent overflow");
  pv2_bytes_output *= (int64_t)n;
  if (output_stride != 1 || !output || output_bytes < pv2_bytes_output)
    polygeist_v2_fatal("transform_thrust_multiplies_scalar_left_f32", "array layout or capacity mismatch");
  if (polygeist_v2_overlaps(y, y_bytes, output, output_bytes))
    polygeist_v2_fatal("transform_thrust_multiplies_scalar_left_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "transform_thrust_multiplies_scalar_left_f32", "polygeist_thrust_multiplies_scalar_left_f32");
  int pv2_k_y = polygeist_v2_map_add(&pv2_map, y, y_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_thrust_multiplies_scalar_left_f32(n, x, (const float *)polygeist_v2_map_dev(&pv2_map, pv2_k_y), (float *)polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_transform_thrust_multiplies_scalar_right_f32(int32_t n, const float *x, int64_t x_bytes, int32_t x_stride, float y, float *output, int64_t output_bytes, int32_t output_stride) {
  if (n < 1 || n > 2147483647)
    polygeist_v2_fatal("transform_thrust_multiplies_scalar_right_f32", "integer outside wrapper bounds");
  int64_t pv2_bytes_x = sizeof(float);
  if ((int64_t)n <= 0 || pv2_bytes_x > INT64_MAX / (int64_t)n)
    polygeist_v2_fatal("transform_thrust_multiplies_scalar_right_f32", "array extent overflow");
  pv2_bytes_x *= (int64_t)n;
  if (x_stride != 1 || !x || x_bytes < pv2_bytes_x)
    polygeist_v2_fatal("transform_thrust_multiplies_scalar_right_f32", "array layout or capacity mismatch");
  int64_t pv2_bytes_output = sizeof(float);
  if ((int64_t)n <= 0 || pv2_bytes_output > INT64_MAX / (int64_t)n)
    polygeist_v2_fatal("transform_thrust_multiplies_scalar_right_f32", "array extent overflow");
  pv2_bytes_output *= (int64_t)n;
  if (output_stride != 1 || !output || output_bytes < pv2_bytes_output)
    polygeist_v2_fatal("transform_thrust_multiplies_scalar_right_f32", "array layout or capacity mismatch");
  if (polygeist_v2_overlaps(x, x_bytes, output, output_bytes))
    polygeist_v2_fatal("transform_thrust_multiplies_scalar_right_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "transform_thrust_multiplies_scalar_right_f32", "polygeist_thrust_multiplies_scalar_right_f32");
  int pv2_k_x = polygeist_v2_map_add(&pv2_map, x, x_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_thrust_multiplies_scalar_right_f32(n, (const float *)polygeist_v2_map_dev(&pv2_map, pv2_k_x), y, (float *)polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_transform_thrust_multiplies_scalars_f32(int32_t n, float x, float y, float *output, int64_t output_bytes, int32_t output_stride) {
  if (n < 1 || n > 2147483647)
    polygeist_v2_fatal("transform_thrust_multiplies_scalars_f32", "integer outside wrapper bounds");
  int64_t pv2_bytes_output = sizeof(float);
  if ((int64_t)n <= 0 || pv2_bytes_output > INT64_MAX / (int64_t)n)
    polygeist_v2_fatal("transform_thrust_multiplies_scalars_f32", "array extent overflow");
  pv2_bytes_output *= (int64_t)n;
  if (output_stride != 1 || !output || output_bytes < pv2_bytes_output)
    polygeist_v2_fatal("transform_thrust_multiplies_scalars_f32", "array layout or capacity mismatch");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "transform_thrust_multiplies_scalars_f32", "polygeist_thrust_multiplies_scalars_f32");
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_thrust_multiplies_scalars_f32(n, x, y, (float *)polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_transform_thrust_plus_f32(int32_t n, const float *x, int64_t x_bytes, int32_t x_stride, const float *y, int64_t y_bytes, int32_t y_stride, float *output, int64_t output_bytes, int32_t output_stride) {
  if (n < 1 || n > 2147483647)
    polygeist_v2_fatal("transform_thrust_plus_f32", "integer outside wrapper bounds");
  int64_t pv2_bytes_x = sizeof(float);
  if ((int64_t)n <= 0 || pv2_bytes_x > INT64_MAX / (int64_t)n)
    polygeist_v2_fatal("transform_thrust_plus_f32", "array extent overflow");
  pv2_bytes_x *= (int64_t)n;
  if (x_stride != 1 || !x || x_bytes < pv2_bytes_x)
    polygeist_v2_fatal("transform_thrust_plus_f32", "array layout or capacity mismatch");
  int64_t pv2_bytes_y = sizeof(float);
  if ((int64_t)n <= 0 || pv2_bytes_y > INT64_MAX / (int64_t)n)
    polygeist_v2_fatal("transform_thrust_plus_f32", "array extent overflow");
  pv2_bytes_y *= (int64_t)n;
  if (y_stride != 1 || !y || y_bytes < pv2_bytes_y)
    polygeist_v2_fatal("transform_thrust_plus_f32", "array layout or capacity mismatch");
  int64_t pv2_bytes_output = sizeof(float);
  if ((int64_t)n <= 0 || pv2_bytes_output > INT64_MAX / (int64_t)n)
    polygeist_v2_fatal("transform_thrust_plus_f32", "array extent overflow");
  pv2_bytes_output *= (int64_t)n;
  if (output_stride != 1 || !output || output_bytes < pv2_bytes_output)
    polygeist_v2_fatal("transform_thrust_plus_f32", "array layout or capacity mismatch");
  if (polygeist_v2_overlaps(x, x_bytes, output, output_bytes))
    polygeist_v2_fatal("transform_thrust_plus_f32", "unsupported output overlap");
  if (polygeist_v2_overlaps(y, y_bytes, output, output_bytes))
    polygeist_v2_fatal("transform_thrust_plus_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "transform_thrust_plus_f32", "polygeist_thrust_plus_f32");
  int pv2_k_x = polygeist_v2_map_add(&pv2_map, x, x_bytes, 0);
  int pv2_k_y = polygeist_v2_map_add(&pv2_map, y, y_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_thrust_plus_f32(n, (const float *)polygeist_v2_map_dev(&pv2_map, pv2_k_x), (const float *)polygeist_v2_map_dev(&pv2_map, pv2_k_y), (float *)polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_transform_thrust_plus_scalar_left_f32(int32_t n, float x, const float *y, int64_t y_bytes, int32_t y_stride, float *output, int64_t output_bytes, int32_t output_stride) {
  if (n < 1 || n > 2147483647)
    polygeist_v2_fatal("transform_thrust_plus_scalar_left_f32", "integer outside wrapper bounds");
  int64_t pv2_bytes_y = sizeof(float);
  if ((int64_t)n <= 0 || pv2_bytes_y > INT64_MAX / (int64_t)n)
    polygeist_v2_fatal("transform_thrust_plus_scalar_left_f32", "array extent overflow");
  pv2_bytes_y *= (int64_t)n;
  if (y_stride != 1 || !y || y_bytes < pv2_bytes_y)
    polygeist_v2_fatal("transform_thrust_plus_scalar_left_f32", "array layout or capacity mismatch");
  int64_t pv2_bytes_output = sizeof(float);
  if ((int64_t)n <= 0 || pv2_bytes_output > INT64_MAX / (int64_t)n)
    polygeist_v2_fatal("transform_thrust_plus_scalar_left_f32", "array extent overflow");
  pv2_bytes_output *= (int64_t)n;
  if (output_stride != 1 || !output || output_bytes < pv2_bytes_output)
    polygeist_v2_fatal("transform_thrust_plus_scalar_left_f32", "array layout or capacity mismatch");
  if (polygeist_v2_overlaps(y, y_bytes, output, output_bytes))
    polygeist_v2_fatal("transform_thrust_plus_scalar_left_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "transform_thrust_plus_scalar_left_f32", "polygeist_thrust_plus_scalar_left_f32");
  int pv2_k_y = polygeist_v2_map_add(&pv2_map, y, y_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_thrust_plus_scalar_left_f32(n, x, (const float *)polygeist_v2_map_dev(&pv2_map, pv2_k_y), (float *)polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_transform_thrust_plus_scalar_right_f32(int32_t n, const float *x, int64_t x_bytes, int32_t x_stride, float y, float *output, int64_t output_bytes, int32_t output_stride) {
  if (n < 1 || n > 2147483647)
    polygeist_v2_fatal("transform_thrust_plus_scalar_right_f32", "integer outside wrapper bounds");
  int64_t pv2_bytes_x = sizeof(float);
  if ((int64_t)n <= 0 || pv2_bytes_x > INT64_MAX / (int64_t)n)
    polygeist_v2_fatal("transform_thrust_plus_scalar_right_f32", "array extent overflow");
  pv2_bytes_x *= (int64_t)n;
  if (x_stride != 1 || !x || x_bytes < pv2_bytes_x)
    polygeist_v2_fatal("transform_thrust_plus_scalar_right_f32", "array layout or capacity mismatch");
  int64_t pv2_bytes_output = sizeof(float);
  if ((int64_t)n <= 0 || pv2_bytes_output > INT64_MAX / (int64_t)n)
    polygeist_v2_fatal("transform_thrust_plus_scalar_right_f32", "array extent overflow");
  pv2_bytes_output *= (int64_t)n;
  if (output_stride != 1 || !output || output_bytes < pv2_bytes_output)
    polygeist_v2_fatal("transform_thrust_plus_scalar_right_f32", "array layout or capacity mismatch");
  if (polygeist_v2_overlaps(x, x_bytes, output, output_bytes))
    polygeist_v2_fatal("transform_thrust_plus_scalar_right_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "transform_thrust_plus_scalar_right_f32", "polygeist_thrust_plus_scalar_right_f32");
  int pv2_k_x = polygeist_v2_map_add(&pv2_map, x, x_bytes, 0);
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_thrust_plus_scalar_right_f32(n, (const float *)polygeist_v2_map_dev(&pv2_map, pv2_k_x), y, (float *)polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_transform_thrust_plus_scalars_f32(int32_t n, float x, float y, float *output, int64_t output_bytes, int32_t output_stride) {
  if (n < 1 || n > 2147483647)
    polygeist_v2_fatal("transform_thrust_plus_scalars_f32", "integer outside wrapper bounds");
  int64_t pv2_bytes_output = sizeof(float);
  if ((int64_t)n <= 0 || pv2_bytes_output > INT64_MAX / (int64_t)n)
    polygeist_v2_fatal("transform_thrust_plus_scalars_f32", "array extent overflow");
  pv2_bytes_output *= (int64_t)n;
  if (output_stride != 1 || !output || output_bytes < pv2_bytes_output)
    polygeist_v2_fatal("transform_thrust_plus_scalars_f32", "array layout or capacity mismatch");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "transform_thrust_plus_scalars_f32", "polygeist_thrust_plus_scalars_f32");
  int pv2_k_output = polygeist_v2_map_add(&pv2_map, output, output_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_thrust_plus_scalars_f32(n, x, y, (float *)polygeist_v2_map_dev(&pv2_map, pv2_k_output));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_transform_thrust_polar_components_f32(int32_t n, const float *magnitude, int64_t magnitude_bytes, int32_t magnitude_stride, const float *angle, int64_t angle_bytes, int32_t angle_stride, float *real, int64_t real_bytes, int32_t real_stride, float *imag, int64_t imag_bytes, int32_t imag_stride) {
  if (n < 1 || n > 2147483647)
    polygeist_v2_fatal("transform_thrust_polar_components_f32", "integer outside wrapper bounds");
  int64_t pv2_bytes_magnitude = sizeof(float);
  if ((int64_t)n <= 0 || pv2_bytes_magnitude > INT64_MAX / (int64_t)n)
    polygeist_v2_fatal("transform_thrust_polar_components_f32", "array extent overflow");
  pv2_bytes_magnitude *= (int64_t)n;
  if (magnitude_stride != 1 || !magnitude || magnitude_bytes < pv2_bytes_magnitude)
    polygeist_v2_fatal("transform_thrust_polar_components_f32", "array layout or capacity mismatch");
  int64_t pv2_bytes_angle = sizeof(float);
  if ((int64_t)n <= 0 || pv2_bytes_angle > INT64_MAX / (int64_t)n)
    polygeist_v2_fatal("transform_thrust_polar_components_f32", "array extent overflow");
  pv2_bytes_angle *= (int64_t)n;
  if (angle_stride != 1 || !angle || angle_bytes < pv2_bytes_angle)
    polygeist_v2_fatal("transform_thrust_polar_components_f32", "array layout or capacity mismatch");
  int64_t pv2_bytes_real = sizeof(float);
  if ((int64_t)n <= 0 || pv2_bytes_real > INT64_MAX / (int64_t)n)
    polygeist_v2_fatal("transform_thrust_polar_components_f32", "array extent overflow");
  pv2_bytes_real *= (int64_t)n;
  if (real_stride != 1 || !real || real_bytes < pv2_bytes_real)
    polygeist_v2_fatal("transform_thrust_polar_components_f32", "array layout or capacity mismatch");
  int64_t pv2_bytes_imag = sizeof(float);
  if ((int64_t)n <= 0 || pv2_bytes_imag > INT64_MAX / (int64_t)n)
    polygeist_v2_fatal("transform_thrust_polar_components_f32", "array extent overflow");
  pv2_bytes_imag *= (int64_t)n;
  if (imag_stride != 1 || !imag || imag_bytes < pv2_bytes_imag)
    polygeist_v2_fatal("transform_thrust_polar_components_f32", "array layout or capacity mismatch");
  if (polygeist_v2_overlaps(magnitude, magnitude_bytes, real, real_bytes))
    polygeist_v2_fatal("transform_thrust_polar_components_f32", "unsupported output overlap");
  if (polygeist_v2_overlaps(magnitude, magnitude_bytes, imag, imag_bytes))
    polygeist_v2_fatal("transform_thrust_polar_components_f32", "unsupported output overlap");
  if (polygeist_v2_overlaps(angle, angle_bytes, real, real_bytes))
    polygeist_v2_fatal("transform_thrust_polar_components_f32", "unsupported output overlap");
  if (polygeist_v2_overlaps(angle, angle_bytes, imag, imag_bytes))
    polygeist_v2_fatal("transform_thrust_polar_components_f32", "unsupported output overlap");
  if (polygeist_v2_overlaps(real, real_bytes, imag, imag_bytes))
    polygeist_v2_fatal("transform_thrust_polar_components_f32", "unsupported output overlap");
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "transform_thrust_polar_components_f32", "polygeist_thrust_polar_components_f32");
  int pv2_k_magnitude = polygeist_v2_map_add(&pv2_map, magnitude, magnitude_bytes, 0);
  int pv2_k_angle = polygeist_v2_map_add(&pv2_map, angle, angle_bytes, 0);
  int pv2_k_real = polygeist_v2_map_add(&pv2_map, real, real_bytes, 1);
  int pv2_k_imag = polygeist_v2_map_add(&pv2_map, imag, imag_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  polygeist_require_gpu_libraries_begin();
  polygeist_cublas_pipeline_begin();
  polygeist_thrust_polar_components_f32(n, (const float *)polygeist_v2_map_dev(&pv2_map, pv2_k_magnitude), (const float *)polygeist_v2_map_dev(&pv2_map, pv2_k_angle), (float *)polygeist_v2_map_dev(&pv2_map, pv2_k_real), (float *)polygeist_v2_map_dev(&pv2_map, pv2_k_imag));
  polygeist_cublas_pipeline_end();
  polygeist_require_gpu_libraries_end();
  polygeist_v2_map_to_host(&pv2_map);
}

/* END GENERATED WRAPPERS */
