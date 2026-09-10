# ATen Egglog whole-build results

- Scope: 598 source-selectable fixtures, five fresh
  sequential builds per fixture, 2990 attempts total.
- Mode: Egglog/equality saturation enabled; handwritten semantic fallback
  disabled. There is no equality-saturation-off comparison in this campaign.
- Boundary: C source through linked Jetson AArch64 executable; executables were
  not run. Timing and RSS medians below include successful builds only.
- Successful attempts: 2463/2990.
- Fully successful fixtures: 491/598.
- Partial fixtures: 3; consistently
  failing fixtures: 104.
- Successful-build wall median: 7.800 s
  [Q1 7.630, Q3 8.010].
- Minimum successful whole-build time: 7.160 s.
- Successful-build peak-RSS median: 154.1 MiB.
- Successful-build matcher median: 332.2 ms.
- Whole-build timeouts: 0.

The raw attempt ledger is `runs.csv`; per-fixture medians and failure reasons
are in `per_kernel.csv`; exact sources, hashes, compiler binaries, parameters,
and host information are in `manifest.json` and `metadata.json`.
