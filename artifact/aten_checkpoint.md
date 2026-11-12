# ATen checkpoint — 2026-11-12

The paper cohort contains 115 non-pointwise cases. This checkpoint has 102
complete library matches and 98 strict GPU correctness passes. Thirteen cases
still lack a complete match; four matched cases retain failed comparisons.
Matching coverage, correctness, and performance are tracked separately.

## Evidence

- [Current status](../issues/aten_paper29_followup_20261112/runtime-validation-status.json)
- [Per-case results and timing samples](../issues/aten_paper29_followup_20261112/runtime-validation-ledger.json)
- [Failure diagnoses](../issues/aten_paper29_followup_20261112/runtime-failure-diagnoses.json)
- [Historical paper data](../issues/aten_c_kernels/paper_analysis/aten_gpu_nonpointwise_115.csv)
- [Pinned dimensions](../issues/aten_paper86_correctness_20261112/shapes.json)
- [COO baseline attribution](../issues/aten_paper29_followup_20261112/coo-native-attribution/paper-comparisons.json)

The retained records include correctness output, timing samples, build metadata,
proof receipts, and machine snapshots. Absolute paths record the original run
location; resolve the portion after `Polygeist/` against this checkout. Local
compiler workspaces, binaries, and intermediate object files are not included.
These records are evidence, not inputs substituting for a fresh build.

## Scope and remaining work

There are 97 validated timings at the recorded paper dimensions: 44 are within
5% of the reported raised runtime or faster, 16 are 5–20% slower, and 37 are
more than 20% slower. Only 89 also use the same recorded statistic. These are
runtime comparisons, not a claim that every native-baseline speedup reproduces.
The ledger retains baseline-scope and input-distribution caveats. The separate
COO follow-up isolates the historical native API/preparation scope.

The four failed comparisons are sparse norm, outer product, transposed 3D
convolution weight gradient, and slow 3D convolution weight gradient. The first,
third, and fourth agree with native CUDA but differ from the C reference under
the strict rounding gate. Outer product agrees with C but differs from native
CUDA in signed zero. They remain failed in the strict pass count.

New timing runs use five warmups and five alternating synchronized pairs.
Historical twenty-pair records remain labeled with their original protocol.

## Reproduction drivers

The source fixtures in `issues/aten_c_kernels` remain unchanged. The drivers
materialize separate copies with the recorded paper dimensions, raise both
application and library references through the established pipeline, prove
matches against `lib_v3`, and emit the corresponding library calls.

- `scripts/correctness/aten_paper115_campaign.py`: fresh raising, matching,
  proof verification, and ABI emission; accepts `--shape-specs` and
  `--case-options` for the saved dimension and option records.
- `scripts/correctness/aten_fixture_batch.py`: fresh build and strict C/native
  CUDA comparisons; use `--correctness-kinds paper --timing-samples 5` with
  the same shape specification and a successful matching campaign.
- `scripts/correctness/aten_fixture_host_sanitize.py`: host memory checks.
- `scripts/correctness/aten_frozen_runtime_campaign.py`: separate timing
  experiments on saved build inputs; not the fresh-source reproduction driver.
- `scripts/correctness/aten_coo_native_attribution.py`: isolated follow-up on
  the COO native baseline preparation and materialization scope.

The shared environment and target-library requirements are documented in the
[artifact README](README.md). Consult each driver's `--help` for build/run
stages and transport configuration. Authentication belongs in local transport
configuration, not in the repository.

## Checkpoint validation

All 304 Python correctness tests passed. The four compiler regression fixtures
for flat fills, large injectivity proofs, permuted initializers, and private
state normalization passed their RUN commands and FileCheck assertions. The
GPU pass count above comes from the recorded campaign, not this host test suite.
