# First two runtime validations

Both fresh whole-operation matches pass strict output, guard, and unchanged-input
checks at the recorded paper dimensions. The original ATen C fixtures are
unchanged. Both C references also passed host ASan/UBSan. All timings below are
minimums of five synchronized alternating raised/native pairs after five warmups.

## Nested all

* Shape: B=724, N=5793; 724 exact int32 outputs in each host/resident mode.
* One imported `cub_ragged_reduce_and_i32` call, using the actual CUB
  `DeviceSegmentedReduce` backend. No residual loops or Linalg operations.
* Normal, edge, zero, and recorded paper input sets all pass. Non-paper length
  metadata includes empty, full, and mixed valid prefixes within [0,N].
* Raised: 0.049697161 ms; actual native ATen: 0.421030998 ms; current ratio 8.4719x.
* Paper raised: 1.672672012 ms; current raised runtime is 33.66x faster.

The native adapter follows the historical `native_exact_region_bench.cpp` API
sequence: preallocated mask/nonzero/where/all_out intermediates plus the int32
output copy, all inside the timed operation. The paper-native input recipe is
different: it used patterned boolean values and lengths N-(b%31), whereas the
recovered publication harness initializes all three integer arrays to zero.
This run preserves the recovered recipe and gives both current implementations
identical values. Its current native/raised ratio must not be presented as an
exact reproduction of the paper speedup. The separate non-paper tests exercise
the nonempty operation.

The CUDA unit test links the exact CUB object used in the benchmark and passes
1,350 exact results. It covers ragged and equal-width AND/OR reductions, zero
width, empty/mixed/full prefixes, nonboolean int32 values, both logical seeds,
repeated calls, allocation guards, and input/length preservation.

## Conv1d

* Shape: B=32, IC=64, OC=128, W=4096, K=3.
* One imported cuDNN convolution-with-bias call. No residual loops or Linalg.
* Raised: 5.805764198 ms; actual `at::convolution_out`: 5.367772102 ms.
* Current native/raised ratio: 0.92456x. Paper raised/native: 4.714207993 / 5.172288 ms.
* Raised runtime is 23.15% slower than paper; native is 3.78% slower.

`POLYGEIST_CUDNN_PLAN_CACHE=1` is recorded for the campaign, but this particular
legacy conv1d wrapper creates its descriptors, requests its algorithm, and
destroys descriptors on every invocation; it does not use the generic cached
convolution helper. No cache improvement or explanation of the entire performance
gap is claimed from this observation alone.

## Evidence

* `gpu-nested-all/aten_nested_all_cpu/{build,gpu-result,paper-comparison}.json`
* `gpu-conv1d/aten_conv1d/{build,gpu-result,paper-comparison}.json`
* `segmented-logical-unit/gpu-test.json` and `gpu-test.log`
* `host-nested-all/` and `host-conv1d/` for sanitizer checks
* `frozen-runtime-inputs/sha256.json` and corresponding copied inputs

Dynamic-library hashes before/after each campaign agree. Executables, linked
object snapshots, hardware/preflight captures, five timing pairs, output checks,
and source/proof hashes remain with each campaign. `run_first_two.py` reproduces
the serial unit and benchmark sequence from fresh frozen builds and refuses to
overwrite the saved result files.

The consolidated ledger is
`../aten_paper29_followup_20261112/runtime-validation-ledger.json`:
106 complete matches, 102 strict passes, 4 preserved strict failures, 9
unmatched. There are 101 strict exact-shape timings, excluding the separately
documented corrected conv2d-columns shape; 46 raised runtimes are within 5% of
paper or faster, 16 are slower by 5-20%, and 39 are slower by more than 20%.
These are raised-runtime comparisons, not a claim that all native baselines and
input distributions match the paper. Previous ledger versions are preserved in
`../aten_paper29_followup_20261112/runtime-ledger-before-paper13/`.

## Conv1d cached runtime follow-up

The existing cuDNN convolution plan cache is now used by the conv1d ABI adapter.
A controlled relink retains the previously validated compiler, specification ABI,
native ATen, strict C reference, and five-pair harness objects; only the runtime
object changes. This does not substitute for a fresh compiler/proof build.
The original full build and 5.805764198 ms measurement remain above.

The cached and cache-disabled binaries both pass all strict paper correctness,
guard, and unchanged-input checks. Untimed diagnostics record one plan miss
followed by one hit with caching enabled, and no cache events when disabled.
No diagnostic tracing is enabled during timing. Five warmups and five alternating
synchronized timing pairs give a cached raised minimum of **5.793699026 ms**,
native ATen **5.505470037 ms**, and current native/raised **0.9502513x**.
The raised result is only 0.21% faster than the earlier uncached measurement and
remains **22.90% slower** than the published 4.714207993 ms. Descriptor/plan reuse
works, but it does not resolve this performance difference.

Evidence: `gpu-conv1d-cache-relink/` contains frozen runtime source and hashes,
unchanged-object hashes, before/after dynamic-library hashes, cache diagnostics,
strict outputs, board state, and all five timing pairs.
`prepare_conv1d_cache_relink.py` and `run_conv1d_cache.py` reproduce this isolated
runtime experiment; the first requires the recorded frozen source directory and
refuses to overwrite a prepared result. `gpu-conv1d-cache/` preserves the earlier
fresh-build rejection caused by stale proof provenance; it is not passing
compiler evidence.

## Generic descriptor-copy and workspace follow-up

The descriptor-copy path originally materialized its runtime stride arrays in a
992-byte per-thread stack. Compile-time indexing removes that stack and all
spills while preserving the runtime descriptors. The earlier optimizations
coalesce compatible axes, retain exact division for coordinate decoding, and
copy integer payload words so signed zeros and NaN payloads are unchanged.
A bounded, opt-in workspace cache separately reuses completed device allocations.

The latest isolated relink (`descriptor-copy-unrolled/`) changes only the Thrust
object relative to `workspace-cache/`; compiler, ABI, C/native reference, other
runtime objects and five-pair harness remain frozen. The byte-exact regression
checks 297,402,368 bytes, including misalignment, offsets, padding, broadcasts,
negative source strides, overlap snapshots, signaling NaNs and signed zeros.
All three paper cases pass strict output, guards and unchanged-input checks.

* Adaptive average pool: 1.087919950 ms, versus 12.623639107 ms immediately
  before unrolling and 0.692063965 ms in the paper.
* Pixel shuffle: 1.714843035 ms, versus 22.535774946 ms immediately before
  unrolling and 0.320575989 ms in the paper.
* Batch normalization: 2.923820972 ms, versus 25.637203932 ms immediately before
  unrolling and 0.262591797 ms in the paper.

These are substantial improvements over the recovered implementation, but all
three remain slower than the paper. Pool and batch normalization also retain a
paper best-of-twenty versus current best-of-five statistic difference. The ledger
records that difference; none of these five-pair results is relabeled as twenty.

Emitted ABI inspection identifies remaining overhead: pool materializes a full
window gather before CUB reduction; pixel shuffle packs both the input and old
output, then performs Scopy and an output layout copy; batch normalization
broadcasts four channel vectors and calls four separate pointwise operations.
The old batch-normalization adapter squared a signed inverse standard deviation
and therefore does not preserve the submitted C fixture. It must not be restored
solely to improve timing.

## Generic tiled-permutation follow-up

When source and destination have different contiguous axes, the descriptor copy
now exchanges a tile through shared memory so both directions use contiguous
transactions. Runtime extents, strides and offsets still determine every byte;
this does not recognize arithmetic or a benchmark. Small layouts, unaligned
payloads and unsupported element widths keep the existing path. Unique output
addresses and non-overlapping source/output storage remain required for either
parallel copy path. Overlapping copies retain snapshot semantics.

The expanded byte-exact GPU regression passes **722,206,720 bytes**, including
both partial tile edges, reversed source axes, outer broadcasts, all previous
alignment/padding/overlap tests, signaling NaNs and signed zeros. All five tiled
payload kernels have zero per-thread stack and zero spill loads/stores.

Only pixel shuffle's frozen runtime was relinked, as its source layout exercises
this path. Strict paper correctness, guards and input preservation pass. With
five warmups and five alternating synchronized pairs, raised minimum is
**1.580666065 ms**, native **0.660202980 ms**. This is a **7.82%** improvement
over the unrolled-copy result, but still **4.93x** the paper's raised time.
Compiler/application, ABI, original C/native references, all other runtime
objects, and the five-pair harness remain unchanged. Shared-library hashes before
and after the run agree. Evidence and frozen source are in
`descriptor-copy-tiled/`; `run_descriptor_copy_tiled.py` reproduces the unit
regression and controlled runtime relink without overwriting saved results.
