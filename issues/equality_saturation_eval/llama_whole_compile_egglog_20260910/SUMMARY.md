# Llama Egglog whole-build results

- Scope: 23 source-selectable fixtures, five fresh
  sequential builds per fixture, 115 attempts total.
- Mode: Egglog/equality saturation enabled; handwritten semantic fallback
  disabled. There is no equality-saturation-off comparison in this campaign.
- Boundary: C source through linked Jetson AArch64 executable; executables were
  not run. Timing and RSS medians below include successful builds only.
- Successful attempts: 85/115.
- Fully successful fixtures: 17/23.
- Partial fixtures: 0; consistently
  failing fixtures: 6.
- Successful-build wall median: 7.710 s
  [Q1 7.540, Q3 7.870].
- Minimum successful whole-build time: 7.320 s.
- Successful-build peak-RSS median: 154.1 MiB.
- Successful-build matcher median: 338.3 ms.
- Whole-build timeouts: 0.

The raw attempt ledger is `runs.csv`; per-fixture medians and failure reasons
are in `per_kernel.csv`; exact sources, hashes, compiler binaries, parameters,
and host information are in `manifest.json` and `metadata.json`.
