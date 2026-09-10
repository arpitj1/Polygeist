# MFEM Egglog whole-build results

- Scope: 31 source-selectable fixtures, five fresh
  sequential builds per fixture, 155 attempts total.
- Mode: Egglog/equality saturation enabled; handwritten semantic fallback
  disabled. There is no equality-saturation-off comparison in this campaign.
- Boundary: C source through linked Jetson AArch64 executable; executables were
  not run. Timing and RSS medians below include successful builds only.
- Successful attempts: 45/155.
- Fully successful fixtures: 9/31.
- Partial fixtures: 0; consistently
  failing fixtures: 22.
- Successful-build wall median: 21.550 s
  [Q1 12.800, Q3 44.120].
- Minimum successful whole-build time: 10.460 s.
- Successful-build peak-RSS median: 154.1 MiB.
- Successful-build matcher median: 2758.3 ms.
- Whole-build timeouts: 0.

The raw attempt ledger is `runs.csv`; per-fixture medians and failure reasons
are in `per_kernel.csv`; exact sources, hashes, compiler binaries, parameters,
and host information are in `manifest.json` and `metadata.json`.
