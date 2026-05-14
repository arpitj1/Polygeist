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
