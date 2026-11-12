#!/usr/bin/env python3
"""Preserve the exact source inputs and tools supporting this campaign."""
import datetime
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from common import OUT as D, ROOT as R
S = D/'source-snapshot-final'
files = list((R/'scripts/correctness').rglob('*.py'))
files += list(Path(__file__).parent.glob('*.py')) + [Path(__file__).with_name('iteration_driver.c'), Path(__file__).with_name('paper_targets.csv')]
files += [R/name for name in (
    'scripts/correctness/polygeist_build.sh', 'scripts/correctness/build_jetson.sh',
    'runtime/polygeist_v2_rt_cuda.c', 'runtime/polygeist_cublas_rt_cuda.c',
    'runtime/polygeist_v2_rt.h', 'runtime/polygeist_v2_rt_cpu.c',
    'runtime/polygeist_cublas_rt.h',
    'generic_solver/kernel_library_v3.mlir', 'generic_solver/lib_v3/import_manifest.json',
    'lib/polygeist/Passes/PlanGpuDataResidency.cpp',
    'lib/polygeist/Passes/PrepareGpuResidualPipeline.cpp',
    'lib/polygeist/Passes/InstrumentGpuRegionTiming.cpp',
    'include/polygeist/Passes/Passes.td')]
files += list((R/'generic_solver/lib_v3/manifest').glob('*.json'))
files += list((R/'generic_solver/lib_v3/ref_c').glob('*.c'))
for case in json.loads((D/'cases.json').read_text()):
    proof = json.loads((D/case/'export/match.json').read_text())
    files += [R/p for p in proof.get('semantic_inputs', {})]
hashes = {}
for path in files:
    relative = path.relative_to(R)
    target = S/relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target)
    hashes[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
for name in ('build/bin/cgeist','build/bin/polygeist-opt','llvm-project/build/bin/mlir-opt',
             'llvm-project/build/bin/mlir-translate','llvm-project/build/bin/clang'):
    hashes[name] = hashlib.sha256((R/name).read_bytes()).hexdigest()
record = dict(timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),
    host_uptime=subprocess.check_output(['uptime'],text=True).strip(), hashes=hashes,
    upstream_polybench_gpu_commit=subprocess.check_output(
        ['git','-C',str(R/'third_party/polybenchGpu'),'rev-parse','HEAD'],text=True).strip(),
    protocol=dict(warmups=5,samples=5,statistic='minimum',dataset='LARGE',datatype='FP64',
        rtol=1e-8,atol=1e-8,timing_scope='CUDA-event compute interval',
        synchronized_host_scope='whole kernel including allocation and transfers',
        fan='automatic; fixed-fan protocol not satisfied'))
(D/'final-provenance.json').write_text(json.dumps(record,indent=2)+'\n')
print('Saved',len(hashes),'source/tool hashes')
