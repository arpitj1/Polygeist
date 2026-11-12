/* Descriptor-directed memory movement. No arithmetic or library recognition.
 * Included after matrix_layout.inc in CPU/CUDA runtime templates. */
typedef struct { int64_t low, bytes; } pv2_copy_range;
#ifdef POLYGEIST_V2_MATRIX_CUDA
#ifdef __cplusplus
extern "C"
#endif
int polygeist_thrust_memref_copy(int64_t, int64_t, const int64_t *,
                               const int64_t *, const int64_t *,
                               const void *, void *) __attribute__((weak));
#endif
static pv2_copy_range pv2_copy_bounds(int64_t rank, int64_t element,
                                    const int64_t *sizes, const int64_t *strides) {
  __int128 low = 0, high = 0;
  for (int64_t i = 0; i < rank; ++i) {
    if (sizes[i] < 0) pv2_matrix_fail("invalid copy extent");
    if (!sizes[i]) return (pv2_copy_range){0, 0};
    __int128 part = (__int128)(sizes[i] - 1) * strides[i];
    if (part < INT64_MIN / element || part > INT64_MAX / element)
      pv2_matrix_fail("copy footprint overflow");
    part *= element;
    if (part < 0) low += part; else high += part;
    if (low < INT64_MIN || high > INT64_MAX - element)
      pv2_matrix_fail("copy footprint overflow");
  }
  __int128 bytes = high - low + element;
  if (bytes > INT64_MAX) pv2_matrix_fail("copy footprint overflow");
  return (pv2_copy_range){(int64_t)low, (int64_t)bytes};
}

static uintptr_t pv2_copy_begin(const void *data, pv2_copy_range range) {
  __int128 begin = (__int128)(uintptr_t)data + range.low;
  if (!data || begin < 0 || begin + range.bytes > UINTPTR_MAX)
    pv2_matrix_fail("invalid copy pointer/footprint");
  return (uintptr_t)begin;
}

/* A permuted/reversed dense view can repeat source dimensions (stride zero)
 * without leaving holes in its physical footprint. Only such a footprint
 * may be transferred wholesale: arbitrary strided views can cross unmapped
 * gaps, or contain destination bytes that must remain untouched. */
static int pv2_copy_dense_footprint(int64_t rank, const int64_t *sizes,
                                  const int64_t *strides) {
  int64_t active = 0;
  for (int64_t i = 0; i < rank; ++i)
    if (sizes[i] > 1 && strides[i]) ++active;
  __int128 span = 1;
  for (int64_t visited = 0; visited < active; ++visited) {
    int64_t found = -1;
    for (int64_t i = 0; i < rank; ++i) {
      __int128 stride = strides[i];
      if (stride < 0) stride = -stride;
      if (sizes[i] > 1 && stride == span) {
        if (found >= 0) return 0;
        found = i;
      }
    }
    if (found < 0) return 0;
    span *= sizes[found];
    if (span > INT64_MAX) return 0;
  }
  return 1;
}

/* Registered host storage can still be the target of queued CUDA work.
 * Keep synchronization eligibility separate from the physical copy direction. */
static int pv2_copy_pointer_kind(const void *data) {
#ifdef POLYGEIST_V2_MATRIX_CUDA
  struct cudaPointerAttributes attributes;
  cudaError_t status = cudaPointerGetAttributes(&attributes, data);
  if (status == cudaSuccess) {
    if (attributes.type == cudaMemoryTypeHost) return 2;
    if (attributes.type == cudaMemoryTypeDevice ||
        attributes.type == cudaMemoryTypeManaged) return 1;
    return 0;
  }
  if (status != cudaErrorInvalidValue)
    pv2_matrix_fail(cudaGetErrorString(status));
  (void)cudaGetLastError();
#else
  (void)data;
#endif
  return 0;
}

static void pv2_copy_bytes(void *dst, const void *src, size_t bytes, int device) {
#ifdef POLYGEIST_V2_MATRIX_CUDA
  if (device) {
    cudaError_t status = cudaMemcpy(dst, src, bytes, cudaMemcpyDefault);
    if (status != cudaSuccess) pv2_matrix_fail(cudaGetErrorString(status));
    return;
  }
#else
  (void)device;
#endif
  memcpy(dst, src, bytes);
}

void polygeist_v2_memref_copy(int64_t rank, int64_t element,
                            const int64_t *sizes, const int64_t *src_strides,
                            const int64_t *dst_strides, const void *source,
                            void *destination) {
  if (rank < 0 || (uint64_t)rank > SIZE_MAX / sizeof(int64_t) || element <= 0 ||
      (rank && (!sizes || !src_strides || !dst_strides)))
    pv2_matrix_fail("invalid copy descriptor");
  pv2_copy_range src_range = pv2_copy_bounds(rank, element, sizes, src_strides);
  pv2_copy_range dst_range = pv2_copy_bounds(rank, element, sizes, dst_strides);
  if (!src_range.bytes || !dst_range.bytes) return;
  uintptr_t x = pv2_copy_begin(source, src_range);
  uintptr_t y = pv2_copy_begin(destination, dst_range);
  if (source == destination &&
      (!rank || !memcmp(src_strides, dst_strides, rank * sizeof(int64_t))))
    return;
  int source_kind = pv2_copy_pointer_kind(source);
  int destination_kind = pv2_copy_pointer_kind(destination);
  int source_device = source_kind == 1, destination_device = destination_kind == 1;
  int pending_gpu_work = source_kind || destination_kind;
  pv2_matrix_synchronize(pending_gpu_work);

  /* Collapse a common contiguous suffix, including arbitrary unit strides. */
  int64_t chunk_elements = 1, outer_rank = rank;
  while (outer_rank) {
    int64_t i = outer_rank - 1;
    if (sizes[i] != 1 && (src_strides[i] != chunk_elements ||
                          dst_strides[i] != chunk_elements)) break;
    chunk_elements = pv2_matrix_mul(chunk_elements, sizes[i]);
    --outer_rank;
  }
  int64_t chunk = pv2_matrix_mul(chunk_elements, element), chunks = 1;
  for (int64_t i = 0; i < outer_rank; ++i)
    chunks = pv2_matrix_mul(chunks, sizes[i]);
  int64_t total = pv2_matrix_mul(chunk, chunks);
  int overlap = x <= y ? y - x < (uint64_t)src_range.bytes
                       : x - y < (uint64_t)dst_range.bytes;
#ifdef POLYGEIST_V2_MATRIX_CUDA
  if (!overlap && chunks > 1 && source_device && destination_device &&
      polygeist_thrust_memref_copy &&
      polygeist_thrust_memref_copy(rank, element, sizes, src_strides, dst_strides,
                                 source, destination)) {
    pv2_matrix_synchronize(pending_gpu_work);
    return;
  }
  /* A host/device broadcast previously issued one synchronous CUDA transfer
   * per logical element. Stage its dense physical host footprint once, then
   * perform the layout copy on device. Preserve the existing serial fallback
   * for aliases, holes, and layouts the parallel copy cannot prove safe. */
  if (!overlap && chunks > 1 && source_device != destination_device &&
      polygeist_thrust_memref_copy && rank <= 16 &&
      pv2_copy_dense_footprint(rank, sizes,
                              source_device ? dst_strides : src_strides)) {
    pv2_copy_range host_range = source_device ? dst_range : src_range;
    void *staging = NULL;
    cudaError_t status = cudaMalloc(&staging, (size_t)host_range.bytes);
    if (status != cudaSuccess) pv2_matrix_fail(cudaGetErrorString(status));
    void *view = (unsigned char *)staging - host_range.low;
    int copied;
    if (source_device) {
      copied = polygeist_thrust_memref_copy(
          rank, element, sizes, src_strides, dst_strides, source, view);
      if (copied) {
        pv2_matrix_synchronize(1);
        pv2_copy_bytes((void *)y, staging, (size_t)host_range.bytes, 1);
      }
    } else {
      pv2_copy_bytes(staging, (const void *)x, (size_t)host_range.bytes, 1);
      copied = polygeist_thrust_memref_copy(
          rank, element, sizes, src_strides, dst_strides, view, destination);
      if (copied) pv2_matrix_synchronize(1);
    }
    status = cudaFree(staging);
    if (status != cudaSuccess) pv2_matrix_fail(cudaGetErrorString(status));
    if (copied) return;
  }
#endif
  unsigned char *snapshot = overlap ? (unsigned char *)malloc((size_t)total) : NULL;
  int64_t *indices = outer_rank ? (int64_t *)calloc(outer_rank, sizeof(int64_t)) : NULL;
  if ((overlap && !snapshot) || (outer_rank && !indices))
    pv2_matrix_fail("copy scratch allocation failed");

  /* Snapshot overlapping views before any write; holes are never copied. */
  for (int phase = 0; phase < (overlap ? 2 : 1); ++phase) {
    if (outer_rank) memset(indices, 0, outer_rank * sizeof(int64_t));
    for (int64_t c = 0; c < chunks; ++c) {
      int64_t src_offset = 0, dst_offset = 0;
      for (int64_t i = 0; i < outer_rank; ++i) {
        src_offset += indices[i] * src_strides[i];
        dst_offset += indices[i] * dst_strides[i];
      }
      const void *src = (const unsigned char *)source + src_offset * element;
      void *dst = (unsigned char *)destination + dst_offset * element;
      int device = source_device || destination_device;
      if (overlap) {
        if (!phase) { dst = snapshot + c * chunk; device = source_device; }
        else { src = snapshot + c * chunk; device = destination_device; }
      }
      pv2_copy_bytes(dst, src, (size_t)chunk, device);
      for (int64_t i = outer_rank; i; --i) {
        if (++indices[i - 1] < sizes[i - 1]) break;
        indices[i - 1] = 0;
      }
    }
  }
  pv2_matrix_synchronize(pending_gpu_work);
  free(indices);
  free(snapshot);
}
