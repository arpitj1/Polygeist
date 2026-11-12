/* Bounded opt-in reuse for compiler-generated device workspaces. No library
 * selection or arithmetic. Active allocations are exclusive leases; releasing
 * one retains cudaFree's device-completion semantics before it can be reused.
 * Context IDs prevent reuse after context destruction or cudaDeviceReset. */
#include <pthread.h>
#include <errno.h>

typedef CUresult (*Pv2WorkspaceGetContext)(CUcontext *);
typedef CUresult (*Pv2WorkspaceGetContextId)(CUcontext, unsigned long long *);
typedef struct {
  void *pointer;
  uint64_t bytes;
  int device, state; /* 0 unused/idle, 1 leased, 2 allocation/release in flight */
  CUcontext context;
  unsigned long long context_id;
} Pv2WorkspaceEntry;
#define PV2_WORKSPACE_SLOTS 64
static Pv2WorkspaceEntry pv2_workspace_entries[PV2_WORKSPACE_SLOTS];
static pthread_mutex_t pv2_workspace_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_once_t pv2_workspace_once = PTHREAD_ONCE_INIT;
static Pv2WorkspaceGetContext pv2_workspace_get_context;
static Pv2WorkspaceGetContextId pv2_workspace_get_context_id;
static int pv2_workspace_enabled, pv2_workspace_diagnostics;
static uint64_t pv2_workspace_retained;
static uint64_t pv2_workspace_limit = UINT64_C(256) * 1024 * 1024;

static void pv2_workspace_initialize(void) {
  pv2_workspace_enabled = environment_flag("POLYGEIST_GPU_WORKSPACE_CACHE", 0);
  pv2_workspace_diagnostics = environment_flag("POLYGEIST_GPU_WORKSPACE_DIAGNOSTICS", 0);
  const char *limit = getenv("POLYGEIST_GPU_WORKSPACE_CACHE_BYTES");
  if (limit && *limit) {
    char *end = NULL;
    errno = 0;
    unsigned long long value = strtoull(limit, &end, 10);
    if (errno || *end || *limit == '-') {
      fprintf(stderr, "polygeist workspace: invalid cache byte limit\n");
      abort();
    }
    pv2_workspace_limit = value;
  }
  if (!pv2_workspace_enabled || !pv2_workspace_limit) return;
  /* Existing executables link cudart rather than the driver directly. A driver
   * without stable context IDs safely keeps the ordinary allocation path. */
  void *driver = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
  if (!driver) return;
  *(void **)(&pv2_workspace_get_context) = dlsym(driver, "cuCtxGetCurrent");
  *(void **)(&pv2_workspace_get_context_id) = dlsym(driver, "cuCtxGetId");
  if (!pv2_workspace_get_context || !pv2_workspace_get_context_id) {
    pv2_workspace_get_context = NULL;
    pv2_workspace_get_context_id = NULL;
    dlclose(driver);
  }
  /* The driver handle lives as long as this process-local cache. */
}

static int pv2_workspace_context(int *device, CUcontext *context,
                                 unsigned long long *id) {
  pthread_once(&pv2_workspace_once, pv2_workspace_initialize);
  if (!pv2_workspace_enabled || !pv2_workspace_limit ||
      !pv2_workspace_get_context || !pv2_workspace_get_context_id) return 0;
  CUDA_CHECK(cudaGetDevice(device));
  if (pv2_workspace_get_context(context) != CUDA_SUCCESS) return 0;
  if (!*context) {
    CUDA_CHECK(cudaFree(NULL));
    if (pv2_workspace_get_context(context) != CUDA_SUCCESS || !*context) return 0;
  }
  return pv2_workspace_get_context_id(*context, id) == CUDA_SUCCESS;
}

static void pv2_workspace_trace(const char *event, uint64_t bytes,
                                 unsigned long long context_id) {
  if (pv2_workspace_diagnostics)
    fprintf(stderr, "POLYGEIST_GPU_WORKSPACE event=%s bytes=%llu retained_bytes=%llu context_id=%llu\n",
            event, (unsigned long long)bytes,
            (unsigned long long)pv2_workspace_retained, context_id);
}

static void *pv2_workspace_allocate(uint64_t bytes, uint8_t host_shared) {
  void *pointer = NULL;
  if (!bytes) return NULL;
  if (host_shared) {
    CUDA_CHECK(cudaHostAlloc(&pointer, (size_t)bytes, cudaHostAllocMapped));
    return pointer;
  }
  int device;
  CUcontext context;
  unsigned long long context_id;
  if (!pv2_workspace_context(&device, &context, &context_id)) {
    if (pv2_workspace_enabled) {
      pthread_mutex_lock(&pv2_workspace_mutex);
      pv2_workspace_trace("context_unavailable", bytes, 0);
      pthread_mutex_unlock(&pv2_workspace_mutex);
    }
    CUDA_CHECK(cudaMalloc(&pointer, (size_t)bytes));
    return pointer;
  }
  pthread_mutex_lock(&pv2_workspace_mutex);
  int best = -1, unused = -1;
  for (int i = 0; i < PV2_WORKSPACE_SLOTS; ++i) {
    Pv2WorkspaceEntry *entry = &pv2_workspace_entries[i];
    if (!entry->pointer && !entry->state) unused = i;
    if (entry->pointer && !entry->state && entry->device == device &&
        entry->context == context && entry->context_id == context_id &&
        entry->bytes >= bytes && (best < 0 || entry->bytes < pv2_workspace_entries[best].bytes))
      best = i;
  }
  if (best >= 0) {
    Pv2WorkspaceEntry *entry = &pv2_workspace_entries[best];
    entry->state = 1;
    pointer = entry->pointer;
    pv2_workspace_trace("hit", bytes, context_id);
    pthread_mutex_unlock(&pv2_workspace_mutex);
    return pointer;
  }
  if (unused >= 0 && bytes <= pv2_workspace_limit - pv2_workspace_retained) {
    Pv2WorkspaceEntry *entry = &pv2_workspace_entries[unused];
    *entry = (Pv2WorkspaceEntry){NULL, bytes, device, 2, context, context_id};
    pv2_workspace_retained += bytes;
    pthread_mutex_unlock(&pv2_workspace_mutex);
    CUDA_CHECK(cudaMalloc(&pointer, (size_t)bytes));
    pthread_mutex_lock(&pv2_workspace_mutex);
    entry->pointer = pointer;
    entry->state = 1;
    pv2_workspace_trace("miss", bytes, context_id);
  } else {
    /* No eviction of active leases or buffers owned by another CUDA context.
     * Full caches fall back without increasing retained storage. */
    pv2_workspace_trace("uncached", bytes, context_id);
  }
  pthread_mutex_unlock(&pv2_workspace_mutex);
  if (!pointer) CUDA_CHECK(cudaMalloc(&pointer, (size_t)bytes));
  return pointer;
}

static void pv2_workspace_release(void *pointer) {
  if (!pointer) return;
  pthread_once(&pv2_workspace_once, pv2_workspace_initialize);
  int device;
  CUcontext context;
  unsigned long long context_id;
  int have_context = pv2_workspace_context(&device, &context, &context_id);
  pthread_mutex_lock(&pv2_workspace_mutex);
  Pv2WorkspaceEntry *entry = NULL;
  for (int i = 0; have_context && i < PV2_WORKSPACE_SLOTS; ++i)
    if (pv2_workspace_entries[i].pointer == pointer &&
        pv2_workspace_entries[i].device == device &&
        pv2_workspace_entries[i].context == context &&
        pv2_workspace_entries[i].context_id == context_id) {
      entry = &pv2_workspace_entries[i];
      break;
    }
  if (entry) {
    if (entry->state != 1) {
      fprintf(stderr, "polygeist workspace: invalid lease at release\n");
      abort();
    }
    entry->state = 2;
    pthread_mutex_unlock(&pv2_workspace_mutex);
    CUDA_CHECK(cudaDeviceSynchronize());
    pthread_mutex_lock(&pv2_workspace_mutex);
    entry->state = 0;
    pv2_workspace_trace("release", entry->bytes, context_id);
    pthread_mutex_unlock(&pv2_workspace_mutex);
    return;
  }
  pthread_mutex_unlock(&pv2_workspace_mutex);
  struct cudaPointerAttributes attributes;
  CUDA_CHECK(cudaPointerGetAttributes(&attributes, pointer));
  if (attributes.type == cudaMemoryTypeHost) {
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFreeHost(pointer));
  }
  else CUDA_CHECK(cudaFree(pointer));
}
