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
