# PolyBench end-to-end correctness — current status

Last run: 2026-05-14. Pipeline = `cgeist` → `polygeist-opt --remove-iter-args --affine-parallelize --raise-affine-to-linalg-pipeline --lower-polygeist-submap [--linalg-debufferize]` → `mlir-opt` (standard MLIR lowering) → `mlir-translate` → `clang` → run + diff against pure-`clang` reference. Dataset: `MINI_DATASET`.

## Raise-only path (15 / 17 PASS)

| Kernel | Result | Notes |
|---|---|---|
| gemm | PASS | bit-exact |
| syr2k | PASS | |
| syrk | PASS | |
| gesummv | PASS | |
| gemver | PASS | |
| bicg | PASS | |
| atax | PASS | |
| mvt | PASS | |
| 2mm | PASS | |
| 3mm | PASS | |
| jacobi-1d | PASS | |
| jacobi-2d | PASS | |
| floyd-warshall | PASS | |
| deriche | PASS | requires `--convert-math-to-llvm` |
| nussinov | PASS | |
| heat-3d | **FAIL_DIFF** | numerical bug — stencil compose loses something |
| correlation | **FAIL_DIFF** | numerical bug — likely similar shape issue |

## Raise + debuferize path (12 / 17 PASS)

Same kernels as above, plus `--linalg-debufferize` in the polygeist-opt pipeline.

| Kernel | Result | Notes |
|---|---|---|
| gemm, syr2k, syrk, gesummv, gemver, bicg, atax, mvt, 2mm, 3mm, floyd-warshall, nussinov | PASS | |
| jacobi-1d | bufferize-back FAIL | `affine.for` with tensor iter_args isn't handled by `one-shot-bufferize` |
| jacobi-2d | bufferize-back FAIL | same |
| heat-3d | bufferize-back FAIL | same |
| deriche | bufferize-back FAIL | same / related |
| correlation | bufferize-back FAIL | same / related |

## Running

- Single kernel: `scripts/correctness/run_kernel_e2e.sh <kernel_dir> <short_name> [--debuf]`
- All 17: `scripts/correctness/run_all_e2e.sh [--debuf]`
- Smoke-only (no run, just lower-to-LLVM-dialect): `scripts/correctness/lower_smoke_test.sh`

The per-kernel wrapper is generated automatically from the C source by
`scripts/correctness/gen_wrapper.py`.

## Known issues / next investigations

1. *heat-3d FAIL_DIFF (numerical)*: the stencil composition produces an
   IR that compiles and runs but gives different values from the C
   reference. The C reference happens to preserve initial values for
   the linear-in-(i+j+k) field (Laplacian = 0), while our lowered
   version produces non-trivial values. The bug is likely in either
   the raise pass's handling of shifted stencil submaps, or in my
   `ComposeSubmapIntoLinalgGeneric` composing `d+const` shifts in a way
   that doesn't agree with what `convert-linalg-to-loops` expects.

2. *correlation FAIL_DIFF (numerical)*: similar — has shifted/sliced
   submaps that lower but produce wrong numerics. Needs the same
   investigation.

3. *5 kernels fail debuferize-path bufferize-back*: `affine.for` with
   tensor iter_args (produced by the debuferize pass) isn't lowered
   correctly by `one-shot-bufferize`. Either need to convert these
   `affine.for` to `scf.for` (which one-shot-bufferize handles) before
   bufferize, or extend the bufferize-back step.

4. *13 / 30 PolyBench kernels still don't lower at all* (broadcasts,
   stencil rejections, chained submaps — see
   `notes/polygeist_raise_to_linalg/` and `raise_correctness_testing.md`
   memory). Each adds another set of e2e candidates once handled.
