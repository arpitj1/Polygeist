# Bounded device-workspace reuse

The compiler-generated `mgpuMemAlloc`/`mgpuMemFree` runtime hooks can retain up to
256 MiB in 64 device-buffer slots when `POLYGEIST_GPU_WORKSPACE_CACHE=1`.
`POLYGEIST_GPU_WORKSPACE_CACHE_BYTES` changes the retention bound. The default
is disabled. Active buffers are exclusive leases protected by a mutex. Reuse
requires matching CUDA device, context handle, and stable context ID. Release
retains device-completion semantics. A full pool or unavailable context identity
uses ordinary CUDA allocation; host-shared allocations use cudaHostAlloc and
cudaFreeHost. The process-local cache never evicts an active lease.

The real-CUDA unit passes with caching on and off. It checks zero allocations,
best-fit reuse, distinct simultaneous leases, a queued producer on a nonblocking
stream, two host threads, oversized uncached allocation, mapped-host deallocation,
and cudaDeviceReset. The reset test first takes the safe uncached path when
context identity is unavailable, then records a new context ID. Retention peaks
at 12 KiB within the configured 16 KiB unit-test bound.

Controlled relinks preserve the optimized descriptor-copy object, application,
ABI, C oracle, native ATen, and five-pair harness. Only the main runtime object
changes. All three recorded paper cases pass strict correctness with caching
both on and off. Untimed diagnostics show misses then hits. Five warmups and
five synchronized alternating pairs, with diagnostic output disabled, give:

* Adaptive average pooling: 12.623639107 ms, prior 15.282596111 ms; native
  0.501543999 ms. Still 18.24x the paper runtime.
* Pixel shuffle CPU backend: 22.535774946 ms, prior 25.622803926 ms; native
  0.658506870 ms. Still 70.30x the paper runtime.
* Batch normalization: 25.637203932 ms, prior 49.887729168 ms; native
  0.324677944 ms. Still 97.63x the paper runtime.

Caching therefore improves this cohort but does not explain the full gap.
No emitted library calls or mathematical matching rules change. The earlier
uncached binaries/timings remain in `../descriptor-copy-optimized/`.
`run_workspace_cache.py` runs the unit before benchmarks, preserves all link
inputs, verifies strict outputs before timing, records cache-disabled correctness
separately, and verifies dynamic-library hashes before/after each case.
