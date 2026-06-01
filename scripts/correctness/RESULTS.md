# PolyBench end-to-end correctness — current status

Last run: 2026-05-14. Pipeline = `cgeist` → `polygeist-opt --remove-iter-args --affine-parallelize --raise-affine-to-linalg-pipeline --lower-polygeist-submap [--linalg-debufferize]` → `mlir-opt` (standard MLIR lowering, with `--expand-strided-metadata`, `--lower-affine`, `--empty-tensor-to-alloc-tensor` on the debuf path) → `mlir-translate` → `clang` → run + diff against pure-`clang` reference. Dataset: `MINI_DATASET`.

## Lowering smoke test (lower-polygeist-submap → mlir-opt to LLVM dialect)

**26 / 30 kernels lower clean.** Up from 17 / 30 before broadcast support.

Remaining 4:
- `adi` (10 ops): stencil shape rejected by Compose's iter-dim-coverage check (all operands drop the reduction dim).
- `seidel-2d` (9 ops): same.
- `durbin` (2 ops): reverse-index access `-d0 + s0 - 1`. Needs negative-stride subview support.
- `ludcmp` (1 op): similar to durbin.

## Raise-only e2e (25 / 26 PASS)

| Kernel | Result |
|---|---|
| gemm, syr2k, syrk, gesummv, gemver, symm, trmm | PASS |
| bicg, atax, mvt, 2mm, 3mm, doitgen | PASS |
| cholesky, gramschmidt, lu, trisolv | PASS |
| heat-3d, jacobi-1d, jacobi-2d, fdtd-2d | PASS |
| floyd-warshall, deriche, nussinov, covariance | PASS |
| **correlation** | **FAIL_DIFF** — raise-side bug (diagonal accumulation; the kernel sets `corr[i][i]=1.0` only once but our lowered linalg.generic accumulates the dot product over the diagonal too, producing `corr[i][i]=2.0`). Independent of the lowering pass — needs a fix in the raise pass to mask the diagonal. |

## Raise + debufferize e2e (24 / 26 PASS)

Same 24 pass through debuferize as well.

Two fail:
- `correlation` — same diagonal bug as raise-only.
- `covariance` — new debuf-path failure: `LinalgDebufferize` produces a `linalg.generic` with mixed tensor/memref operands. Probably interaction with the new broadcast lowering. Needs separate investigation.

## What changed today

1. **Broadcast-shape lowering in `ComposeSubmapIntoLinalgGeneric`.** Extended the
   per-base-dim decomposition to handle pure `SymbolExpr` and pure `ConstantExpr`
   results — these become rank-reducing offsets in the emitted `memref.subview`.
   The consumer linalg.generic's indexing_map for that operand drops the
   corresponding view-dim(s). Unlocks covariance, durbin, cholesky, gramschmidt,
   lu, ludcmp, trisolv, symm, doitgen, trmm in the smoke test.

2. **Subview-for-offsets instead of compose-into-linalg.** When ANY operand
   of a linalg has a non-zero offset (shifted stencil access, fixed-index
   capture), emit a `memref.subview` for that operand AND for all other
   operands so iter-dim bounds stay consistent. Composes only the
   permutation part of the original submap map into the linalg's
   indexing_map. Fixes heat-3d numerical bug.

3. **`--expand-strided-metadata`** before standard lowering. Required to
   handle the strided memref results from `memref.subview` in the
   final-to-llvm stage.

4. **`--lower-affine` + `--empty-tensor-to-alloc-tensor`** before
   `--one-shot-bufferize` on the debuf path. Lifts `affine.for` with
   tensor iter_args to `scf.for` (which one-shot-bufferize handles) and
   converts `tensor.empty` from privatization to `bufferization.alloc_tensor`.

## Running

- Single kernel: `scripts/correctness/run_kernel_e2e.sh <kernel_dir> <short_name> [--debuf]`
- All 26: `scripts/correctness/run_all_e2e.sh [--debuf]`
- Smoke-only: `scripts/correctness/lower_smoke_test.sh`

## Jetson warmed raised runtime vs PolyBenchGPU CUDA

Run date: 2026-05-28. Device: Jetson Orin. Datatype: double. Dimensions:
`N/NI/NJ/NK/NL/NM=512`.

Method: 50 in-process iterations, discard first 10 warmups, then report a 10%
trimmed mean over the remaining 40 samples. Raised path uses
`POLYGEIST_RT_TIMING=1` runtime-shim device timings summed per benchmark
iteration. PolyBenchGPU path uses CUDA events around the handwritten kernel
sequence. This avoids counting cuBLAS first-use cold-start as steady-state
runtime.

| Kernel | Raised rt-gpu ms | PolyBenchGPU CUDA ms | Result |
|---|---:|---:|---|
| gemm | 3.809 | 7.697 | raised 2.02x faster |
| 2mm | 7.640 | 11.200 | raised 1.47x faster |
| 3mm | 11.451 | 10.501 | PolyBenchGPU 1.09x faster |
| gesummv | 0.069 | 0.341 | raised 4.93x faster |
| gemver | 0.188 | 0.313 | raised 1.66x faster |

Previous cold outer-harness comparison, kept for context only:

| Kernel | Raised outer s | Raised rt-gpu s | PolyBenchGPU CUDA s |
|---|---:|---:|---:|
| gemm | 0.103025 | 0.033008 | 0.008401 |
| 2mm | 0.112321 | 0.036679 | 0.034213 |
| 3mm | 0.117875 | 0.040612 | 0.038889 |
| gesummv | 0.097759 | 0.032294 | 0.019568 |
| gemver | 0.100270 | 0.032451 | 0.031399 |

## Darknet im2col + GEMM fused path

Run date: 2026-05-29. Device: Jetson Orin. Fixture:
`third_party/cnn-extracted/darknet_im2col_gemm.c`, `MINI_DATASET`
(`IC=3`, `OC=4`, `H=W=8`, `K=3`, `stride=1`, `pad=1`).

Progress saved:
- Raise pipeline lifts the guarded im2col workspace fill and the following
  `i,k,j` GEMM.
- Kernel matcher recognizes the 3-step composition
  `zero(output) + guarded im2col(workspace) + SGEMM(output)` and emits one
  `kernel.launch @cudnnConvolutionFwd_im2col_gemm`.
- ABI lowering maps that launch to
  `polygeist_cudnn_conv2d_im2col_gemm_f32`, avoiding materialized im2col.
- Host CPU shim matches the original C reference exactly.
- Jetson run exits 0. Output compare: 256 printed values, max absolute diff
  `0.0001`, no values above `1.1e-3`.
- First-call Jetson timing from the fused path:
  `POLYGEIST_RT_TIMING op=cudnnConv2d_im2col_gemm m=4 n=64 k=27 host_ms=26.356336 device_ms=15.357408`.

## llama2.c RMSNorm and softmax lowering

Run date: 2026-05-29. Device: Jetson Orin. Fixtures:
`third_party/cnn-extracted/llama2_rmsnorm.c` and
`third_party/cnn-extracted/llama2_softmax.c`, `N=128`.

Progress saved:
- Matcher emits `kernel.launch @rmsnorm_f32(%x, %weight, %out)` for the
  two-stage llama2 RMSNorm pattern.
- Matcher emits `kernel.launch @cudnnSoftmaxForward(%x)` for the three-stage
  max / exp+sum / divide softmax pattern.
- ABI lowering maps RMSNorm to `polygeist_rmsnorm_f32` and softmax to
  `polygeist_cudnn_softmax_forward_f32`.
- Host CPU-stub correctness is byte-exact for both fixtures versus plain
  `gcc -O2` reference output.
- Jetson RMSNorm exits 0 through cuDNN backend graph
  `CUDNN_RMS_NORM` / `CUDNN_NORM_FWD_INFERENCE` and is byte-exact versus the
  aarch64 reference. Timing:
  `POLYGEIST_RT_TIMING op=cudnnRmsNormForward m=1 n=128 k=0 host_ms=180.841512 device_ms=8.238944`.
- Jetson softmax exits 0 using `cudnnSoftmaxForward`. Output compare:
  128 values, max absolute diff `1.0e-8`, no values above `1.0e-6`.
  Timing: `POLYGEIST_RT_TIMING op=cudnnSoftmaxForward m=1 n=128 k=0 host_ms=121.393178 device_ms=120.336578`.
- Caveat: the installed target has cuDNN's C backend graph API rather than the
  C++ `cudnn_frontend` wrapper headers, so the runtime builds the graph with
  `cudnnBackend*` descriptors directly. The graph path currently uses real
  CUDA device allocations/copies; mapped host pointers hit
  `CUDNN_STATUS_BAD_PARAM_MISALIGNED_POINTER` at execution time.

## llama2 tiny forward tensor path

Run date: 2026-05-30. Fixture:
`third_party/cnn-extracted/llama2_tiny_forward.c`, `N=16`, `H=16`.

Progress saved:
- Debufferized tensor path now matches RMSNorm as
  `kernel.launch @rmsnorm_f32_tensor`, zero-init as `@memset_zero_1D_f32`,
  and GEMV as `@cublasSgemv`.
- ABI lowering emits three runtime calls:
  `polygeist_rmsnorm_f32`, `polygeist_cublas_memset_zero_1d_f32`, and
  `polygeist_cublas_sgemv`.
- Host CPU-stub output is byte-exact versus the native C reference.
- Jetson output matches native within `2.0e-08` max absolute difference.
  Runtime timing confirmed RMSNorm + SGEMV dispatch:
  `POLYGEIST_RT_TIMING op=host_rmsnorm_f32 ...` and
  `POLYGEIST_RT_TIMING op=cublasSgemv m=16 n=16 ...`.
- Caveat: the whole-forward softmax tail remains residual tensor code in this
  fixture because the max phase is still an `affine.for` + `scf.if`, not the
  clean 3-step softmax linalg pattern.

## llama2 larger forward tensor path

Run date: 2026-05-31. Fixture:
`third_party/cnn-extracted/llama2_forward_bench.c`, default `N=1024`, `H=4096`;
Jetson run used `REPEAT=5` in one process.

Progress saved:
- The default tensor path matches all four intended launches:
  `@rmsnorm_f32_tensor`, `@memset_zero_1D_f32`, `@cublasSgemv`, and
  `@cudnnSoftmaxForward_tensor`.
- Host CPU-stub output is byte-exact versus native C for the printed sample
  and checksum.
- Jetson output matches native with max absolute diff `2.56e-06` over the
  printed 32 values plus softmax checksum.
- Unlike the tiny `N=16` fixture, RMSNorm uses the cuDNN backend graph at
  `N=1024` instead of falling back to the host path.
- Warm Jetson device timings after first-use setup:
  `cudnnRmsNormForward` ~`0.09-0.10 ms`, `cublasSgemv` ~`0.53-0.55 ms`,
  `cudnnSoftmaxForward` ~`0.028-0.030 ms`.

## llama.cpp suffix comparison

Run date: 2026-05-31. Device: Jetson Orin. Goal: apples-to-apples comparison
against the part of llama.cpp/ggml that corresponds to the C suffix we can
raise today.

Workload compared:
`RMSNorm + scale + output projection GEMV -> logits`
with `N=2048`, `H=32000`, 5 warmup iterations, 30 measured iterations.
This is not a full `llama-bench` comparison. `llama-bench` measures whole
`llama_decode` to logits, while our C fixture only covers the final suffix.
Sampling softmax is also outside the `llama_decode` path, so the clean
comparison stops at logits rather than probabilities.

Artifacts:
- ggml helper: `scripts/correctness/llama_suffix_ggml_bench.cpp`.
- ggml Jetson log:
  `/tmp/llama_suffix_ggml_logits_n2048_h32000.log`.
- raised C Jetson log:
  `/tmp/llama2_forward_bench_raised_n2048_h32000.log`.

Measured warm numbers:
- ggml/llama.cpp CUDA logits suffix: median `1.494 ms`, trimmed mean
  `1.494 ms`.
- Raised pipeline logits suffix, device-only: median `2.135 ms`, trimmed mean
  `2.134 ms`.
- Raised pipeline logits suffix, host-visible: median `186.1 ms`, trimmed mean
  `186.1 ms`.
- Device-only ratio: raised pipeline is about `1.43x` slower than ggml for
  this suffix.

Correctness sanity:
- ggml logits sample:
  `0.06607100, 0.33554888, -0.36427033, 0.09345388`.
- Native C logits for the same initialization match to expected FP32
  tolerance.
- Full raised softmax checksum for the fixture is approximately `1.000001`.

Slowness diagnosis:
- Host-visible time is dominated by RMSNorm setup. `cudnnRmsNormForward`
  warm host median is `184.0 ms`, while its device median is only `0.093 ms`.
  The runtime currently rebuilds cuDNN backend descriptors, engine config,
  execution plan, variant pack, device allocations, input copies, output copy,
  and descriptor cleanup on every call.
- Device time is mostly the output projection. Raised `cublasSgemv` warm
  device median is `2.038 ms`, which is already slower than ggml's entire
  RMSNorm+projection logits suffix at `1.494 ms`.
- ggml benefits from graph scheduling/CUDA graph reuse and a matvec-oriented
  layout/kernel path. Our lowering emits separate runtime calls
  (`RMSNorm`, zero-fill, SGEMV) and synchronizes each shim for timing/current
  ABI behavior.

Next runtime fixes, in priority order:
1. Cache cuDNN RMSNorm descriptors/plans/buffers, or replace RMSNorm with a
   simple custom fused CUDA kernel for the Llama vector case.
2. Replace decode-style output `cublasSgemv` with a row-major custom matvec
   kernel or a cuBLASLt matmul path tuned for `H x N` by `N`.
3. Drop explicit logits zero-fill when GEMV uses `beta=0`.
4. Avoid per-shim synchronization; run the suffix asynchronously on one stream
   or capture it as a graph.

RMSNorm cache update, 2026-06-01:
- Runtime change: `polygeist_rmsnorm_f32` now caches cuDNN backend descriptors,
  execution plan, variant pack, workspace, and device buffers by `N` instead of
  rebuilding them on every call.
- Rebuilt and reran the same `N=2048`, `H=32000`, `REPEAT=35` Jetson fixture.
  Cached log: `/tmp/llama2_forward_bench_cached_rms_n2048_h32000.log`.
- First call still pays cuDNN plan creation (`cudnnRmsNormForward` host
  `214.7 ms`), but warm calls reuse the plan.
- Warm RMSNorm host median dropped from `184.0 ms` to `0.052 ms`.
- Warm raised logits suffix host median dropped from `186.1 ms` to `1.652 ms`.
- Warm raised logits suffix device median in this rerun was `1.614 ms`.
- With the cached path, the remaining gap to ggml's `1.494 ms` logits suffix
  is primarily the output projection path (`cublasSgemv` median `1.588 ms` in
  this rerun) plus separate shim overhead, not cuDNN RMSNorm plan setup.

Standalone Llama op sweep, 2026-06-01:
- Fixture source: `third_party/cnn-extracted/llama_forward_ops.c`.
- Timing harness: `third_party/cnn-extracted/llama_forward_ops_harness.c`.
- Build path: `scripts/correctness/polygeist_build.sh --target=jetson`
  with one raised function per binary.
- Run setup: Jetson Orin, `REPEAT=50`, discard first 5 iterations, report warm
  median/mean. Shapes are `MODEL_DIM=64`, `FFN_DIM=128`, `SEQ_LEN=32`,
  `VOCAB=256`.
- All 17 matched standalone ops ran successfully. The interleaved RoPE and
  branchy mask variants still do not raise; the split/branchless variants do.

```
op                       launch host_med_ms host_mean_ms dev_med_ms dev_mean_ms
token_embedding               1      0.0319       0.0322     0.0243      0.0245
attention_rmsnorm             1      0.0652       0.0657     0.0471      0.0461
qkv_projection                6      0.0687       0.0686     0.0446      0.0445
rope_split                    4      0.1486       0.1494     0.0969      0.0973
kv_cache_rw                   4      0.1244       0.1252     0.0908      0.0925
attention_scores              2      0.0215       0.0221     0.0135      0.0141
attention_mask_select         1      0.0422       0.0422     0.0275      0.0275
attention_softmax             2      0.0552       0.0534     0.0384      0.0363
attention_output              2      0.0208       0.0210     0.0128      0.0131
output_projection             2      0.0252       0.0257     0.0157      0.0164
residual_add                  1      0.0440       0.0393     0.0361      0.0308
ffn_rmsnorm                   1      0.0652       0.0644     0.0465      0.0445
gate_up_projection            4      0.0445       0.0451     0.0286      0.0286
swiglu                        1      0.0376       0.0376     0.0248      0.0248
down_projection               2      0.0252       0.0259     0.0156      0.0161
final_rmsnorm                 1      0.0662       0.0654     0.0475      0.0455
lm_head_projection            2      0.0246       0.0251     0.0156      0.0163
```

- Approximate standalone-composed one-layer total: host median `0.8322 ms`,
  device median `0.5750 ms`.
- Approximate `token_embedding + one layer + final_rmsnorm + lm_head` total:
  host median `0.9548 ms`, device median `0.6623 ms`.

## Known remaining bugs / next investigations

1. *correlation FAIL_DIFF*: raise pass accumulates dot product over the
   diagonal (which the C source sets to 1.0 explicitly and skips in its
   off-diagonal computation). Needs a mask in the produced linalg.generic.
   *Diagonal = 2.0 instead of 1.0.*

2. *covariance debuf-path FAIL*: debuferize produces a linalg.generic with
   mixed tensor and memref operands.

3. *adi / seidel-2d lowering*: Compose's iter-dim-coverage check
   correctly rejects (all operands drop the reduction dim). Real fix
   needs raise to encode the iter-dim bound explicitly (or a different
   representation).

4. *durbin / ludcmp lowering*: reverse-indexed access (`-d0 + s0 - 1`).
   Needs negative-stride subview support in the lowering.
