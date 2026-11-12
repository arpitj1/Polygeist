# ATen validation checkpoint

The 2026-11-12 campaign adds 73 GPU-validated cases to the earlier 13-case
cohort, giving 86 accumulated successes out of the 115 Figure 5 fixtures.
The remaining 29 cases do not yet have a successful execution record.
The original benchmark C files and their fixed dimensions are unchanged.

`VALIDATED_RESULTS.json` records all 73 additions, source/proof/binary hashes,
normal/edge/zero-input correctness results, native baseline descriptions, and
raw timing samples. These are small-fixture measurements, not reproduction
of the paper-sized Figure 5 performance comparison. Some baselines are
explicit compositions of actual ATen CUDA operations.

Every additional case passed three comparisons: generated output against C,
generated output against native ATen, and native ATen against C. Checks also
cover input preservation and allocation guards. Sixty earlier cases were
reaudited by rebuilding the strengthened harness and linking it to the
original, hashed compiler/runtime/native objects. This preserves attribution
to the recorded snapshots; it is not a fresh build of the entire cohort with
the latest compiler. The earlier 13-case cohort was not reaudited here.

Fresh builds use `scripts/correctness/aten_paper115_campaign.py` for raising
and matching and `scripts/correctness/aten_fixture_batch.py` for GPU builds,
correctness, and timings. Both provide `--help`. The fixture runner requires a
complete specification match and an independent native baseline. Site SSH
configuration is supplied separately with `--transport`.

`audit_native_reference.py` documents the local recheck procedure. It requires
the original experiment objects and is not a clean-checkout reproduction
driver. The local build-record paths in the result file identify retained
evidence; executables, build directories, and private transport configuration
are not committed.

The final precommit checks pass 283 Python tests, 48 compiler tests, and
all 22 prior application matches. Regression checks also cover Llama. The latest Llama run preserves
26 calls and all 32,000 logits from the previous binary, passes the strict
expert-output comparison, and takes 11.431936264 ms versus 11.377632141 ms
before these changes (minimum of five samples).
