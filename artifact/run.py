#!/usr/bin/env python3
"""Run one paper evaluation, recording commands and provenance in a fresh directory."""
import argparse
import datetime
import os
from pathlib import Path
import subprocess
import sys

from common import ARTIFACT, ROOT, environment, provenance, save, setting

SUITES = ['polybench_cpu', 'polybench_gpu', 'llama', 'mfem', 'eqsat',
          'graph_sizes', 'ginsbach', 'kernelfarer']

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('suite', choices=SUITES)
    p.add_argument('--stage', choices=['plan', 'build', 'run', 'summarize', 'all'], default='all')
    p.add_argument('--output', type=Path)
    p.add_argument('--cases', nargs='+', help='Explicit subset; omit for the full paper cohort')
    a, extra = p.parse_known_args()
    if a.stage in ('run', 'summarize') and a.output is None:
        p.error('--output is required when resuming a built experiment')
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = (a.output or ARTIFACT/'results'/f'{a.suite}-{stamp}').resolve()
    script = ARTIFACT/'benchmarks'/a.suite/'reproduce.py'
    stage = 'all' if a.stage == 'plan' else a.stage
    cmd = [sys.executable, str(script), '--stage', stage]
    if a.suite == 'polybench_cpu':
        if stage == 'summarize':
            p.error('PolyBench CPU writes RESULTS.csv during --stage run')
        cmd += ['--output', str(out), '--remote-helper', str(ARTIFACT/'remote.py'),
                '--cpu', str(setting('cpu', 0))]
        if setting('openblas_root'):
            cmd += ['--openblas-root', str(setting('openblas_root'))]
        if a.cases:
            cmd += ['--kernels', ','.join(a.cases)]
    elif a.cases:
        cmd += ['--cases', *a.cases]
    cmd += extra
    if a.stage == 'plan':
        print(' '.join(cmd))
        print('Output:', out)
        return 0
    if stage in ('all', 'build') and out.exists() and any(out.iterdir()):
        p.error('Use a fresh output directory for build/all; run/summarize reuse existing outputs')
    out.mkdir(parents=True, exist_ok=True)
    env = environment(LL_ARTIFACT_OUTPUT=out, LL_ARTIFACT_SUITE=a.suite)
    os.environ.update(env)
    provenance(out if stage in ('build', 'all') else out/('stage-'+stage))
    save(out/('command-'+stage+'.json'), cmd)
    # Keep timing campaigns serial. The drivers may parallelize compilation only.
    return subprocess.run(cmd, cwd=ROOT, env=env).returncode

if __name__ == '__main__':
    sys.exit(main())
