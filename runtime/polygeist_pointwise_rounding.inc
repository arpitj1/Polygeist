/* Preserve explicit FP32 multiplication results across possible FMA fusion.
 * Cache storage is never recycled: captured CUDA graphs retain these pointers. */
#include "polygeist_pointwise_partition.h"
#define POINTWISE_ROUNDING_CACHE_CAP 8
struct pointwise_rounding_plan {
  int in_use, nodes;
  int32_t N;
  uint64_t words[12];
  struct pg_pointwise_partition partition;
  void *storage;
  float *temporaries[24];
};
static struct pointwise_rounding_plan
    g_pointwise_rounding_cache[POINTWISE_ROUNDING_CACHE_CAP];

static int pointwise_rounding_arity(int opcode) {
  cudnnPointwiseMode_t mode;
  if (!pointwise_graph_mode(opcode, &mode)) return -1;
  return pointwise_graph_ternary_opcode(opcode) ? 3 :
         pointwise_graph_binary_opcode(opcode) ? 2 : 1;
}
static int pointwise_rounding_multiply(int opcode) { return opcode == 2; }
static int pointwise_rounding_add_subtract(int opcode) {
  return opcode == 1 || opcode == 3;
}
static int pointwise_rounding_separate_backend(int opcode) { return opcode == 4; }
static int pointwise_rounding_boolean(int opcode) { return pointwise_graph_boolean_opcode(opcode); }
static struct pointwise_rounding_plan *pointwise_rounding_find(
    int32_t N, const uint64_t words[12], int nodes, int capturing) {
  for (int i = 0; i < POINTWISE_ROUNDING_CACHE_CAP; ++i) {
    struct pointwise_rounding_plan *p = &g_pointwise_rounding_cache[i];
    if (p->in_use && p->N == N && p->nodes == nodes &&
        memcmp(p->words, words, sizeof(p->words)) == 0) return p;
  }
  struct pg_pointwise_partition partition;
  int status = pg_pointwise_partition_build_mixed(
      words, nodes, pointwise_rounding_arity, pointwise_rounding_multiply,
      pointwise_rounding_add_subtract, pointwise_rounding_separate_backend,
      pointwise_rounding_boolean, &partition);
  if (!status) return NULL;
  if (status < 0) {
    fprintf(stderr, "polygeist runtime: pointwise FP32 rounding partition "
                    "exceeds the supported typed graph interface\n");
    abort();
  }
  if (capturing) {
    fprintf(stderr, "polygeist runtime: pointwise rounding plan requires "
                    "warmup before CUDA graph capture\n");
    abort();
  }
  for (int i = 0; i < POINTWISE_ROUNDING_CACHE_CAP; ++i) {
    struct pointwise_rounding_plan *p = &g_pointwise_rounding_cache[i];
    if (p->in_use) continue;
    memset(p, 0, sizeof(*p));
    p->N = N;
    p->nodes = nodes;
    p->partition = partition;
    memcpy(p->words, words, sizeof(p->words));
    size_t bytes = (size_t)N * sizeof(float);
    if (bytes > SIZE_MAX - 15) abort();
    size_t pitch = (bytes + 15) & ~(size_t)15;
    size_t count = (size_t)partition.count - 1;
    if (count > SIZE_MAX / pitch) abort();
    if (count) CUDA_CHECK(cudaMalloc(&p->storage, count * pitch));
    for (int k = 0; k + 1 < partition.count; ++k)
      p->temporaries[partition.stages[k].output] =
          (float *)((char *)p->storage + (size_t)k * pitch);
    p->in_use = 1;
    return p;
  }
  fprintf(stderr, "polygeist runtime: pointwise rounding cache full\n");
  abort();
}

static int pointwise_rounding_execute(
    int32_t N, const uint64_t words[12], int nodes, const float scalars[8],
    const float *inputs[4], const int32_t strides[4], int32_t out_stride,
    float *output, int capturing) {
  /* Most graphs cannot form an FMA; retain their original execution path. */
  int has_multiply = 0, has_add = 0, has_separate_backend = 0;
  for (int i = 0; i < nodes; ++i) {
    int opcode = (int)(pg_pointwise_instruction(words, i) >> 24);
    has_multiply |= pointwise_rounding_multiply(opcode);
    has_add |= pointwise_rounding_add_subtract(opcode);
    has_separate_backend |= pointwise_rounding_separate_backend(opcode);
  }
  if (!has_separate_backend && (!has_multiply || !has_add)) return 0;
  struct pointwise_rounding_plan *p =
      pointwise_rounding_find(N, words, nodes, capturing);
  if (!p) return 0;

  /* Resolve the complete host alias group before launching any stage. A later
   * overlapping registration must not invalidate an earlier queued address. */
  void *host[5] = {(void *)inputs[0], (void *)inputs[1], (void *)inputs[2],
                   (void *)inputs[3], output};
  size_t bytes[5] = {0};
  for (int i = 0; i < 5; ++i) {
    if (i < 4 && !(p->partition.used_inputs & (1u << i))) continue;
    int32_t stride = i == 4 ? out_stride : strides[i];
    if (stride <= 0 || (size_t)(N - 1) >
        (SIZE_MAX / sizeof(float) - 1) / (size_t)stride) {
      fprintf(stderr, "polygeist runtime: invalid pointwise rounding layout\n");
      abort();
    }
    bytes[i] = ((size_t)(N - 1) * (size_t)stride + 1) * sizeof(float);
  }
  void *device[5];
  register_host_operands_safe(host, bytes, device, 5);
  /* Keep original pointers for nested calls: a mapped device alias is not a
   * new host allocation. All subsequent subset lookups hit this registration. */
  for (int i = 0; i < p->partition.count; ++i) {
    const struct pg_pointwise_stage *stage = &p->partition.stages[i];
    const float *in[4] = {NULL};
    int32_t inc[4] = {1, 1, 1, 1};
    for (int j = 0; j < stage->external_count; ++j) {
      int ref = stage->external[j];
      in[j] = ref < 4 ? inputs[ref] : p->temporaries[ref - 12];
      inc[j] = ref < 4 ? strides[ref] : 1;
    }
    int final = i + 1 == p->partition.count;
    float *out = final ? output : p->temporaries[stage->output];
    const uint64_t *w = stage->words;
    polygeist_cudnn_pointwise_graph_f32(
        N, (int64_t)w[0], (int64_t)w[1], (int64_t)w[2], (int64_t)w[3],
        (int64_t)w[4], (int64_t)w[5], (int64_t)w[6], (int64_t)w[7],
        (int64_t)w[8], (int64_t)w[9], (int64_t)w[10], (int64_t)w[11],
        stage->nodes, scalars[0], scalars[1], scalars[2], scalars[3],
        scalars[4], scalars[5], scalars[6], scalars[7], inc[0], inc[1],
        inc[2], inc[3], final ? out_stride : 1, in[0], in[1], in[2], in[3], out);
  }
  return 1;
}
