/* Generic descriptor-based cuDNN emission for an imported pointwise program.
 * Input refs 0..7, scalar refs 8..15, topological tensor_graph_instruction
 * refs 16..39. This layer consumes a proved library body; it does not inspect
 * applications.
 */
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
extern int polygeist_thrust_memref_copy(int64_t, int64_t, const int64_t *,
                                        const int64_t *, const int64_t *,
                                        const void *, void *);
#define TENSOR_GRAPH_CUDNN(x)                                                  \
  do {                                                                         \
    cudnnStatus_t s = (x);                                                     \
    if (s != CUDNN_STATUS_SUCCESS) {                                           \
      fprintf(stderr, "tensor graph: %s: %s\n", #x, cudnnGetErrorString(s));   \
      abort();                                                                 \
    }                                                                          \
  } while (0)
#define TENSOR_GRAPH_CUDA(x)                                                   \
  do {                                                                         \
    cudaError_t s = (x);                                                       \
    if (s != cudaSuccess) {                                                    \
      fprintf(stderr, "tensor graph CUDA: %s\n", cudaGetErrorString(s));       \
      abort();                                                                 \
    }                                                                          \
  } while (0)
static void tensor_graph_invalid(void) {
  fprintf(stderr, "tensor_graph_invalid imported tensor graph descriptor\n");
  abort();
}
static cudnnBackendDescriptor_t
tensor_graph_create(cudnnBackendDescriptorType_t type) {
  cudnnBackendDescriptor_t d = NULL;
  TENSOR_GRAPH_CUDNN(cudnnBackendCreateDescriptor(type, &d));
  return d;
}
static void tensor_graph_attr(cudnnBackendDescriptor_t d,
                              cudnnBackendAttributeName_t a,
                              cudnnBackendAttributeType_t t, int64_t n,
                              const void *v) {
  TENSOR_GRAPH_CUDNN(cudnnBackendSetAttribute(d, a, t, n, v));
}
static cudnnBackendDescriptor_t
tensor_graph_tensor(int rank, const int64_t *dims, const int64_t *strides,
                    int64_t uid, bool virtual_, bool scalar) {
  cudnnBackendDescriptor_t d =
      tensor_graph_create(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
  cudnnDataType_t dtype = CUDNN_DATA_FLOAT;
  int64_t alignment = 4;
  tensor_graph_attr(d, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1,
                    &dtype);
  tensor_graph_attr(d, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, rank,
                    dims);
  tensor_graph_attr(d, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, rank,
                    strides);
  tensor_graph_attr(d, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &uid);
  tensor_graph_attr(d, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1,
                    &alignment);
  tensor_graph_attr(d, CUDNN_ATTR_TENSOR_IS_VIRTUAL, CUDNN_TYPE_BOOLEAN, 1,
                    &virtual_);
  tensor_graph_attr(d, CUDNN_ATTR_TENSOR_IS_BY_VALUE, CUDNN_TYPE_BOOLEAN, 1,
                    &scalar);
  TENSOR_GRAPH_CUDNN(cudnnBackendFinalize(d));
  return d;
}
struct tensor_graph_plan {
  bool occupied, used_inputs[8], used_scalars[8];
  PolygeistPointwiseTensorLayout key, layout;
  bool broadcast_batch[8];
  float *batch_inputs[8];
  cudnnHandle_t handle;
  cudnnBackendDescriptor_t leaves[16], values[24], pw[24], ops[24], graph, heur,
      config, execution_plan, pack;
  void *workspace, *pointers[9];
  float scalars[8];
};
static struct tensor_graph_plan tensor_graph_cache[32];
static pthread_mutex_t tensor_graph_lock = PTHREAD_MUTEX_INITIALIZER;
static uint32_t
tensor_graph_instruction(const PolygeistPointwiseTensorLayout *p, int n) {
  return (uint32_t)(p->words[n / 2] >> (32 * (n % 2)));
}
static cudnnBackendDescriptor_t tensor_graph_ref(struct tensor_graph_plan *p,
                                                 int n) {
  return n < 16 ? p->leaves[n] : p->values[n - 16];
}
static void tensor_graph_build(struct tensor_graph_plan *p) {
  p->layout = p->key;
  PolygeistPointwiseTensorLayout *l = &p->layout;
  /* The vendor's 3D fusion surface requires the batch extent on vector
   * operands. Materialize only this missing coordinate, never spatial axes. */
  for (int i = 0; i < l->inputs; ++i)
    if (l->input_dims[i][0] == 1 && l->dims[0] != 1) {
      p->broadcast_batch[i] = true;
      l->input_dims[i][0] = l->dims[0];
      int64_t stride = 1;
      for (int d = 2; d >= 0; --d) {
        l->input_strides[i][d] = stride;
        stride *= l->input_dims[i][d];
      }
      TENSOR_GRAPH_CUDA(cudaMalloc((void **)&p->batch_inputs[i],
                                   (size_t)stride * sizeof(float)));
    }
  TENSOR_GRAPH_CUDNN(cudnnCreate(&p->handle));
  TENSOR_GRAPH_CUDNN(
      cudnnSetStream(p->handle, (cudaStream_t)polygeist_cuda_graph_stream()));
  int rank = l->rank;
  int64_t packed[8], ones[8];
  int64_t stride = 1;
  for (int d = rank - 1; d >= 0; --d) {
    packed[d] = stride;
    stride *= l->dims[d];
    ones[d] = 1;
  }
  for (int i = 0; i < l->nodes; ++i) {
    uint32_t inst = tensor_graph_instruction(l, i);
    int mode = inst >> 24;
    if (mode < 1 || mode > 3)
      tensor_graph_invalid();
    int references[2] = {(inst >> 16) & 255, (inst >> 8) & 255};
    for (int j = 0; j < 2; ++j) {
      int r = references[j];
      if (r < 8) {
        if (r >= l->inputs)
          tensor_graph_invalid();
        p->used_inputs[r] = true;
      } else if (r < 16)
        p->used_scalars[r - 8] = true;
      else if (r >= 16 + i)
        tensor_graph_invalid();
    }
  }
  for (int i = 0; i < 8; ++i) {
    if (p->used_inputs[i])
      p->leaves[i] = tensor_graph_tensor(
          rank, l->input_dims[i], l->input_strides[i], 100 + i, false, false);
    if (p->used_scalars[i])
      p->leaves[8 + i] =
          tensor_graph_tensor(rank, ones, ones, 200 + i, false, true);
  }
  for (int i = 0; i < l->nodes; ++i) {
    uint32_t inst = tensor_graph_instruction(l, i);
    int op = inst >> 24;
    cudnnPointwiseMode_t mode = op == 1   ? CUDNN_POINTWISE_ADD
                                : op == 2 ? CUDNN_POINTWISE_MUL
                                          : CUDNN_POINTWISE_SUB;
    cudnnDataType_t precision = CUDNN_DATA_FLOAT;
    p->pw[i] = tensor_graph_create(CUDNN_BACKEND_POINTWISE_DESCRIPTOR);
    tensor_graph_attr(p->pw[i], CUDNN_ATTR_POINTWISE_MODE,
                      CUDNN_TYPE_POINTWISE_MODE, 1, &mode);
    tensor_graph_attr(p->pw[i], CUDNN_ATTR_POINTWISE_MATH_PREC,
                      CUDNN_TYPE_DATA_TYPE, 1, &precision);
    TENSOR_GRAPH_CUDNN(cudnnBackendFinalize(p->pw[i]));
    bool last = i == l->nodes - 1;
    p->values[i] = tensor_graph_tensor(rank, l->dims, l->output_strides,
                                       last ? 400 : 300 + i, !last, false);
    p->ops[i] =
        tensor_graph_create(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR);
    int ra = (inst >> 16) & 255, rb = (inst >> 8) & 255;
    bool broadcast_a = ra < 16, broadcast_b = rb < 16;
    if (ra < 8)
      broadcast_a =
          memcmp(l->input_dims[ra], l->dims, rank * sizeof(int64_t)) != 0;
    if (rb < 8)
      broadcast_b =
          memcmp(l->input_dims[rb], l->dims, rank * sizeof(int64_t)) != 0;
    /* cuDNN requires a full-sized first input. ADD/MUL commute without
     * reassociating the reference expression; SUB must retain operand order. */
    if ((op == 1 || op == 2) && broadcast_a && !broadcast_b) {
      int tmp = ra;
      ra = rb;
      rb = tmp;
    }
    cudnnBackendDescriptor_t a = tensor_graph_ref(p, ra),
                             b = tensor_graph_ref(p, rb);
    tensor_graph_attr(p->ops[i], CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR,
                      CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &p->pw[i]);
    tensor_graph_attr(p->ops[i], CUDNN_ATTR_OPERATION_POINTWISE_XDESC,
                      CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &a);
    tensor_graph_attr(p->ops[i], CUDNN_ATTR_OPERATION_POINTWISE_BDESC,
                      CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &b);
    tensor_graph_attr(p->ops[i], CUDNN_ATTR_OPERATION_POINTWISE_YDESC,
                      CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &p->values[i]);
    TENSOR_GRAPH_CUDNN(cudnnBackendFinalize(p->ops[i]));
  }
  p->graph = tensor_graph_create(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
  tensor_graph_attr(p->graph, CUDNN_ATTR_OPERATIONGRAPH_HANDLE,
                    CUDNN_TYPE_HANDLE, 1, &p->handle);
  tensor_graph_attr(p->graph, CUDNN_ATTR_OPERATIONGRAPH_OPS,
                    CUDNN_TYPE_BACKEND_DESCRIPTOR, l->nodes, p->ops);
  TENSOR_GRAPH_CUDNN(cudnnBackendFinalize(p->graph));
  const cudnnBackendHeurMode_t modes[] = {
      CUDNN_HEUR_MODE_INSTANT, CUDNN_HEUR_MODE_A, CUDNN_HEUR_MODE_FALLBACK};
  for (int m = 0; m < 3 && !p->execution_plan; ++m) {
    cudnnBackendDescriptor_t h = tensor_graph_create(
                                 CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR),
                             c = NULL, e = NULL;
    cudnnStatus_t s =
        cudnnBackendSetAttribute(h, CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                 CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &p->graph);
    if (s == CUDNN_STATUS_SUCCESS)
      s = cudnnBackendSetAttribute(h, CUDNN_ATTR_ENGINEHEUR_MODE,
                                   CUDNN_TYPE_HEUR_MODE, 1, &modes[m]);
    if (s == CUDNN_STATUS_SUCCESS)
      s = cudnnBackendFinalize(h);
    int64_t count = 0;
    if (s == CUDNN_STATUS_SUCCESS) {
      c = tensor_graph_create(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR);
      s = cudnnBackendGetAttribute(h, CUDNN_ATTR_ENGINEHEUR_RESULTS,
                                   CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &count,
                                   &c);
    }
    if (s == CUDNN_STATUS_SUCCESS && count) {
      e = tensor_graph_create(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
      s = cudnnBackendSetAttribute(e, CUDNN_ATTR_EXECUTION_PLAN_HANDLE,
                                   CUDNN_TYPE_HANDLE, 1, &p->handle);
      if (s == CUDNN_STATUS_SUCCESS)
        s = cudnnBackendSetAttribute(e, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                     CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &c);
      if (s == CUDNN_STATUS_SUCCESS)
        s = cudnnBackendFinalize(e);
    }
    if (s == CUDNN_STATUS_SUCCESS && e) {
      p->execution_plan = e;
      p->config = c;
      p->heur = h;
    } else {
      if (e)
        cudnnBackendDestroyDescriptor(e);
      if (c)
        cudnnBackendDestroyDescriptor(c);
      cudnnBackendDestroyDescriptor(h);
    }
  }
  if (!p->execution_plan) {
    fprintf(stderr, "cuDNN has no engine for imported tensor graph\n");
    abort();
  }
  int64_t count = 0, bytes = 0;
  TENSOR_GRAPH_CUDNN(cudnnBackendGetAttribute(
      p->execution_plan, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE,
      CUDNN_TYPE_INT64, 1, &count, &bytes));
  if (bytes)
    TENSOR_GRAPH_CUDA(cudaMalloc(&p->workspace, bytes));
}
void polygeist_cudnn_tensor_pointwise_f32(
    const PolygeistPointwiseTensorLayout *layout, const float *const *inputs,
    const float *scalars, float *output) {
  if (!layout || !inputs || layout->rank != 3 || layout->inputs < 1 ||
      layout->inputs > 8 || layout->nodes < 1 || layout->nodes > 24 || !output)
    tensor_graph_invalid();
  int64_t count = 1;
  for (int d = 0; d < layout->rank; ++d) {
    if (layout->dims[d] <= 0 || layout->dims[d] > INT32_MAX / count ||
        layout->output_strides[d] <= 0)
      tensor_graph_invalid();
    count *= layout->dims[d];
    for (int i = 0; i < layout->inputs; ++i)
      if (!inputs[i] ||
          (layout->input_dims[i][d] != 1 &&
           layout->input_dims[i][d] != layout->dims[d]) ||
          layout->input_strides[i][d] <= 0)
        tensor_graph_invalid();
  }
  /* Only contiguous row-major storage is currently emitted by the ABI. */
  int64_t os = 1, is[8] = {1, 1, 1, 1, 1, 1, 1, 1};
  for (int d = 2; d >= 0; --d) {
    if (layout->output_strides[d] != os)
      tensor_graph_invalid();
    os *= layout->dims[d];
    for (int i = 0; i < layout->inputs; ++i) {
      if (layout->input_strides[i][d] != is[i])
        tensor_graph_invalid();
      is[i] *= layout->input_dims[i][d];
    }
  }
  pthread_mutex_lock(&tensor_graph_lock);
  struct tensor_graph_plan *p = NULL;
  for (int i = 0; i < 32; ++i)
    if (tensor_graph_cache[i].occupied &&
        !memcmp(&tensor_graph_cache[i].key, layout, sizeof(*layout))) {
      p = &tensor_graph_cache[i];
      break;
    }
  if (!p)
    for (int i = 0; i < 32; ++i)
      if (!tensor_graph_cache[i].occupied) {
        p = &tensor_graph_cache[i];
        p->occupied = true;
        p->key = *layout;
        tensor_graph_build(p);
        break;
      }
  if (!p) {
    fprintf(stderr, "tensor graph tensor plan capacity exceeded\n");
    abort();
  }
  for (int i = 0; i < 8; ++i)
    if (p->used_scalars[i]) {
      if (!scalars)
        tensor_graph_invalid();
      p->scalars[i] = scalars[i];
    }
  void *pointers[9] = {0};
  for (int i = 0; i < layout->inputs; ++i) {
    pointers[i] = (void *)inputs[i];
    if (p->broadcast_batch[i]) {
      int64_t source[3] = {0, layout->input_strides[i][1],
                           layout->input_strides[i][2]};
      if (!polygeist_thrust_memref_copy(
              3, sizeof(float), p->layout.input_dims[i], source,
              p->layout.input_strides[i], inputs[i], p->batch_inputs[i]))
        tensor_graph_invalid();
      pointers[i] = p->batch_inputs[i];
    }
  }
  pointers[8] = output;
  if (!p->pack || memcmp(p->pointers, pointers, sizeof(pointers))) {
    if (p->pack)
      TENSOR_GRAPH_CUDNN(cudnnBackendDestroyDescriptor(p->pack));
    p->pack = tensor_graph_create(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR);
    int64_t ids[17];
    void *values[17];
    int n = 0;
    for (int i = 0; i < 8; ++i)
      if (p->used_inputs[i]) {
        ids[n] = 100 + i;
        values[n++] = pointers[i];
      }
    for (int i = 0; i < 8; ++i)
      if (p->used_scalars[i]) {
        ids[n] = 200 + i;
        values[n++] = &p->scalars[i];
      }
    ids[n] = 400;
    values[n++] = output;
    tensor_graph_attr(p->pack, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                      CUDNN_TYPE_VOID_PTR, n, values);
    tensor_graph_attr(p->pack, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,
                      CUDNN_TYPE_INT64, n, ids);
    tensor_graph_attr(p->pack, CUDNN_ATTR_VARIANT_PACK_WORKSPACE,
                      CUDNN_TYPE_VOID_PTR, 1, &p->workspace);
    TENSOR_GRAPH_CUDNN(cudnnBackendFinalize(p->pack));
    memcpy(p->pointers, pointers, sizeof(pointers));
  }
  TENSOR_GRAPH_CUDNN(
      cudnnBackendExecute(p->handle, p->execution_plan, p->pack));
  pthread_mutex_unlock(&tensor_graph_lock);
}

#undef TENSOR_GRAPH_CUDNN
#undef TENSOR_GRAPH_CUDA
