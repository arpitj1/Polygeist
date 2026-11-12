# Public BSR reference and remaining proof gaps

Recorded 2026-11-12. This is reference-import evidence, not a completed ATen
match, GPU correctness result, or performance measurement.

The imported C implementation comes from public NVIDIA AMGX/CUSP
`bsr_matvec<int,float>` and its `gemv` helper. The manifest pins both upstream
files by SHA-256. The import selects square blocks, provides a zero-filled output
for alpha=1/beta=0, inlines the public helper, and gives the contiguous blocks and
vectors explicit VLA dimensions. The block dimension remains a runtime parameter;
there is no special case for the paper's block dimension 4. The original ATen C
fixture is unchanged.

`test_cusp_bsr_reference.py` compiles the real upstream CUSP header and the imported
C independently, then compares them bit for bit at block dimensions 1, 2, 3, 5,
and 7. It includes empty rows, nonzero initial output values, an untouched spare
output block, and input-preservation checks. All pass.

The shared import reports affine and memref-Linalg numerical validation EXACT.
The tensor definition route is UNSOUND (maximum absolute difference 0.8961), so
that route must not supply proof evidence. Its memref output submap aliases the
reduction axis onto the same output element; turning that aliasing view into an
independent tensor loses the intended accumulation. The imported affine stage is
usable instead.

The initial `diagnostic.json` captures two blockers before the generic scalar
carry improvement:

- A nested scalar loop carry creates a scratch seed read of the enclosing carry
  that the equation builder rejects as an intermediate memory value.
- The public reference has the analogous case: an inner scalar carry seeds from
  an output element and copies its result back to that element.

The common equation builder now coalesces those seed/copy cells only after
checking their storage lifetimes. It keeps every update's original order,
expression, guards, and iteration domain. It rejects independent uses of the old
outer value. Five focused tests verify matching against direct updates, wrong
seeds and updates, captured-old-value rejection, empty inner loops, and rejection
of floating-point reassociation under the strict arithmetic model.

The common coordinate-view normalizer now retains an existing dominating
contiguous output view, after checking that every access factors exactly and
stays inside that view. For a readonly flat input, it can create a contiguous
view only when every accessed block covers the full inner residue interval and
the outer coordinate is independent of that residue. Valid original accesses
then imply that every accessed block fits in floor(physical_capacity / width)
rows, including when the physical buffer has an unused tail. It neither assumes
an index range for the sparse columns nor changes the original input contract.
Partial-block accesses, extra scalar accesses, residue-dependent indices, writes
to the operand, unsupported layouts, and tensor-writeback cases remain excluded.

The application memref-Linalg stage now proves equal to the validated affine
reference under the ordinary equation binding search. The diagnostic records
block_rows=16 and block_dim=4, inferred from application iteration domains. The
same mechanism is tested at widths 2, 3, 5, and 7; it contains no BSR recognizer.
The emitted kernel proof, ABI lowering, and GPU validation remain separate gates.
The original `diagnostic.json` preserves the initial failure, and
`after-coordinate-views/diagnostic.json` records the successful equation comparison.

Reproduce the source parity test:

```
/usr/bin/python3 -m unittest scripts.correctness.tests.test_cusp_bsr_reference -v
```

Regenerate current raising/equation diagnostics into a separate output directory:

```
export POLYGEIST_ROOT=$PWD
source envsetup.sh
/usr/bin/python3 scripts/correctness/diagnose_bsr_reference.py --output /tmp/bsr-reference-current
```
