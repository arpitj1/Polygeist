# Paper evaluations

The LoopLibrarian benchmark sources, paper cohorts, timing drivers, and retained
results are maintained in the [LoopLibrarian-Benchmarks repository](https://github.com/PRONTOLab/LoopLibrarian-Benchmarks).
Its `third_party/Polygeist` submodule pins this `artifact-submission` branch.

This branch contains the compiler, runtime adapters, reference specifications,
and compiler regression tests. Configure `POLYGEIST_DEPENDENCY_ROOT` to use
external public source checkouts and SDK headers when importing specifications.
Source hashes and validation requirements remain enforced.

The legacy robustness harness can access the external evaluation fixtures with
`POLYGEIST_BENCHMARK_ROOT=/path/to/LoopLibrarian-Benchmarks`.
