# LoopLibrarian artifact

This directory contains fresh-source reproduction commands for eight completed
evaluation items. Run commands from the repository root. Each run writes its
commands, compiler and input hashes, intermediate IR, full output checks, and
raw samples to a separate output directory. Saved historical executables and
timings are not inputs to these drivers.

## What is included

- `polybench_cpu`: Figure 3, all 17 FP64 LARGE programs, native CPU versus
  LoopLibrarian/OpenBLAS. Five warmups, five measurements, minimum elapsed time.
- `polybench_gpu`: Figure 4, all 12 FP64 LARGE kernels, native CUDA versus
  resident LoopLibrarian. Five warmups, five measurements, minimum CUDA-event
  compute interval. Separate full-output and actual vendor-call audits.
- `llama`: all 19 operation fixtures and the complete FP32, one-token,
  one-layer inference fixture, including the real ggml CUDA implementation.
- `mfem`: the 18 supported contraction fixtures and integrated PCG updates,
  with full-output correctness checks.
- `eqsat`: all 30 PolyBench inputs with full/disabled equality saturation,
  plus SpEQ on the original inputs and its linearized GEMM.
- `graph_sizes`: the paired GEMM experiment, retaining the actual serialized
  LoopLibrarian graphs and SpEQ DOT graphs.
- `ginsbach`: the author's IDL compiler on all 30 PolyBench programs, then
  execution and full-output checking of its detected replacements.
- `kernelfarer`: Table 9, fresh native, real KernelFaRer, and whole-program
  LoopLibrarian builds for GEMM, SYR2K, 2mm, and 3mm.

ATen, PVA, compilation-cost measurements, and the 257-specification inventory
are outside this package's eight evaluation commands. The ongoing ATen campaign
has a separate [checkpoint and evidence index](aten_checkpoint.md); it is not
yet a completed 115-case reproduction.

## Machines and dependencies

The GPU evaluations require a Jetson AGX Orin (SM87). Figure 3 also uses its
CPU, pinned to core 0 with one OpenBLAS thread. An x86 host with a working
Polygeist build cross-compiles the executables. KernelFaRer and IDL run on the
x86 host. Record the host model and load when interpreting their timings;
these are not the Orin Figure 3 baseline. EqSat and graph measurements need
only the host and Docker.

Install the repository's pinned LLVM/MLIR and Polygeist build as described in
the root README. Required executables are `build/bin/{cgeist,polygeist-opt}`
and the LLVM tools under `llvm-project/build/bin`. Do not substitute a system
MLIR release. Also install:

- Python 3.10 or later, `egglog==11.4.0`, and NumPy (tested with 1.21.5);
- CMake, Ninja, Git, GCC/G++, Clang/LLVM 14, OpenBLAS development files;
- `gcc-aarch64-linux-gnu` and `g++-aarch64-linux-gnu`;
- CUDA 12.6 with the AArch64 SBSA target, nvcc, cuBLAS/cuSOLVER/cuDNN,
  and AArch64 cuTENSOR headers and libraries;
- an AArch64 OpenBLAS sysroot for Figure 3;
- Docker for the real SpEQ LLVM fork and its older egglog dependency.

Llama and the runtime need CUDA/CCCL headers, including Thrust. The source
reference importer also verifies the pinned public library sources. NVIDIA
libraries are obtained separately under their distribution terms; this
directory does not redistribute those libraries or any PVA SDK source.

Create your local configuration:

```
cp artifact/config.example.json artifact/config.json
export POLYGEIST_ROOT="$PWD"
source envsetup.sh
```

Edit the paths in `artifact/config.json`. It is gitignored. `ssh_host` is an
ordinary alias from your SSH configuration; authentication and proxy routing
stay there. `remote_library_path` must contain the target's CUDA, cuDNN,
cuTENSOR, and other required shared libraries. The literal SSH host `local`
is supported when execution is already on the target. A site with an existing
transport can optionally set `transport_module` to a Python module exporting
`run(command, timeout)` and `upload(local_path, remote_path)`.

Every configuration key can be overridden by an uppercase `LL_` environment
variable, for example `LL_SSH_HOST=orin`. The `environment` mapping supplies
compiler variables such as `CUDA`, `POLYGEIST_CUTENSOR_ROOT`, and
`POLYGEIST_GGML_BUILD`. `POLYGEIST_NVCC` can override the runtime compiler's
nvcc path. Keep these consistent with `nvcc` and `cuda_target`.

Prepare public dependencies as needed:

```
/usr/bin/python3 artifact/setup.py sources
/usr/bin/python3 artifact/setup.py verify-sources
/usr/bin/python3 artifact/setup.py specs
/usr/bin/python3 artifact/setup.py kernelfarer
/usr/bin/python3 artifact/setup.py ginsbach
/usr/bin/python3 artifact/setup.py speq
/usr/bin/python3 artifact/setup.py ggml
```

`sources` fetches the revisions in `dependency_sources/repos.json` and CCCL
3.2.0, then checks every upstream file against the specification manifests.
Existing checkouts with other revisions are rejected rather than reset.
`specs` raises and numerically validates the V3 references with your compiler;
run it after building or changing the compiler. The implementation lives in
the historically named `scripts/correctness/lib_v2` directory, but these
commands explicitly select **lib_v3**. Reference bodies, not handwritten
operation recognizers, supply the matching specifications.

The KernelFaRer and IDL setup commands apply the disclosed patches beside
their drivers. KernelFaRer's plugin is then compiled by its evaluation command.
SpEQ setup downloads the [published artifact](https://zenodo.org/records/10802555)
(6.3 GB; checksum recorded in `setup.py`) and builds its LLVM fork at
`00bd6254b3832d94558b7c38a394ea03d01a2763`. Allow additional space and time for
the LLVM build. The Dockerfile omits the unrelated comparison toolchains in
SpEQ's original image; its recognizer, reference library, LLVM passes, and
5/1/3 saturation schedule are retained.

Configure the Orin's clocks/power for the publication protocol before timing.
The preflight records clocks, load, temperature, power mode, software, and fan
state; it rejects busy/hot devices and unlocked CPU clocks. The scripts do
not silently change system settings. Automatic fan control differs from the
paper's fixed-fan setting and must be disclosed in a result report.

## Run the evaluations

Use a **new output directory** for every `all` or `build` invocation:

```
/usr/bin/python3 artifact/run.py polybench_cpu --output artifact/results/fig3
/usr/bin/python3 artifact/run.py polybench_gpu --output artifact/results/fig4
/usr/bin/python3 artifact/run.py llama --output artifact/results/llama
/usr/bin/python3 artifact/run.py mfem --output artifact/results/mfem
/usr/bin/python3 artifact/run.py eqsat --output artifact/results/eqsat
/usr/bin/python3 artifact/run.py graph_sizes --output artifact/results/graphs
/usr/bin/python3 artifact/run.py ginsbach --output artifact/results/idl
/usr/bin/python3 artifact/run.py kernelfarer --output artifact/results/table9
```

Run timing suites sequentially. Do not compile other programs or share the
device during performance measurements. `jobs` controls compilation and
analysis concurrency, not concurrent timing. Host uptime is recorded.

`--stage plan` prints the command without executing it. `--stage build` creates
fresh binaries; `--stage run --output ...` executes those exact binaries and
checks their recorded hashes. For EqSat and graph sizes, the analysis happens
in `build`, and `run` rechecks/reports those analysis outputs. Most suites also
support `--stage summarize`; CPU writes its report as part of `run`.

`--cases gemm` selects a subset for CPU/GPU PolyBench or KernelFaRer.
MFEM and EqSat also accept explicit case names. Subsets must not be reported
as full-cohort reproduction. For a host-only CPU harness smoke test:

```
/usr/bin/python3 artifact/run.py polybench_cpu --cases gemm --target host --size MINI --output artifact/results/cpu-smoke
```

That smoke test checks the executable path, not Figure 3 performance.
The drivers can also be called directly with `PYTHONPATH=artifact` and
`LL_ARTIFACT_OUTPUT` set, but the central launcher records more provenance.

## Acceptance and interpretation

Correctness failure is an error exit and excludes a timing result. A speedup
difference is reported, not hidden or converted to a pass through a loose
tolerance. Timing values depend on the recorded machine and clock state.

- **Figure 3:** all 17 full outputs must pass `atol=rtol=1e-8`.
  `results.json` records raw samples and the paper speedup comparison;
  `RESULTS.csv` is the compact timing table. C source computation is unchanged;
  only output precision is increased. Whole-program compilation preserves
  initialization and helpers.
- **Figure 4:** `results.csv` reports compute time and speedup after complete
  FP64 output checks at `atol=rtol=1e-8`. Device compute excludes initial/final
  transfers and allocation; synchronized end-to-end timing is retained
  separately. These are separate-kernel ABI harnesses. Ten native computation
  bodies come from pinned PolyBenchGPU; correlation/covariance use the paper's
  explicitly labeled local CUDA adaptations. Vendor tracing is a separate
  run of the same timed executable.
- **Llama:** `summary.json` reports 14/19 recognition, ABI acceptance separately,
  five raw runtime samples, and all 32,000 logits compared with real ggml.
  The expert-output gate is `atol=0.001, rtol=0.0001`. The native CPU comparison
  is still recorded separately; it is not silently claimed to pass. Reported
  LL runtime is 12.826 ms; the later observed 11.38 ms is not a hardcoded target.
  Source accommodations, including split RoPE and branchless masking, remain
  visible in the committed input fixture.
- **MFEM:** 18 supported fixtures use `NE=1024`; integrated PCG uses `NE=2`.
  The extraction harness checks all output buffers and prints explicit
  correctness evidence. Current cuBLAS call counts and the paper's 128
  cuTensorNet contraction sites are different counting units. No new timing
  claim is made here.
- **EqSat:** the submitted cohort is 17 with EqSat versus 8 without. An
  additional recognition, currently `ludcmp`, is listed separately, along
  with its ABI-lowering result. SpEQ's one match is the linearized GEMM;
  its count on unmodified PolyBench is recorded separately. All 30 inputs
  and both arms are retained; cases are not selected by their outcome.
- **Graph sizes:** the paired result is 370 SpEQ nodes versus 21 LL nodes.
  LL counts the largest accepted scalar proof, including three primitive
  leaves (18 function nodes). SpEQ counts a whole LLVM loop graph with
  primitive values inline in labels. These are different graph scopes and
  conventions; the raw graphs are included in the output.
- **IDL:** four detected idiom instances occur in correlation (one), covariance
  (two), and trmm (one). All three programs execute and fail numerical checking.
  Reproducing those failures is the expected negative result, not successful
  acceleration. This experiment uses the original PolyBench allocation helpers
  because the inlined helpers in compiler tests change the author's detection.
- **KernelFaRer:** all four complete programs must match the native output.
  Inputs are FP32 LARGE, with `atol=0.011, rtol=0.0005` for the historical
  two-decimal comparison convention. New native/KF/LL measurements all use
  five warmups and five samples, minimum time. The paper LL speedups are
  8.6/366/115/93; KF speedups are 8.6/346/113/112. The historical native runs
  used a different repetition protocol (one warmup, ten fresh processes);
  this difference remains explicit. See `summary.csv` and `timing.json`.

## Source adaptations and validation

`inputs/` contains the same C extractions and harnesses used by these
experiments, with paths made relative to this package. It does not contain
historical binaries or measured output dumps. MFEM and Llama inputs are
extracted/transliterated fixtures, not a claim to compile their entire
unmodified upstream applications.

`VALIDATION.md` records what was actually tested while packaging this branch.
It distinguishes harness smoke tests from full-cohort experiments. A clean
rebuild on an independent machine remains the final portability check.
