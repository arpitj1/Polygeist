#!/usr/bin/env python3
"""Record optional workspace placement after verified specification ABI lowering.

These passes change storage, never select library computations. The caller
must verify the matching report and perform mechanical ABI lowering first.
The selected entry is invoked serially; host-visible retained workspaces are
not suitable for concurrent calls to that same entry.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import re
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def calls(text):
    return collections.Counter(re.findall(r'\b(?:func\.)?call\s+@([\w.$-]+)\(', text))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--match-proof', type=Path, required=True)
    p.add_argument('--function', required=True)
    p.add_argument('--opt', type=Path, required=True)
    a = p.parse_args()
    text = a.input.read_text()
    if not re.fullmatch(r'[A-Za-z_$][\w.$-]*', a.function):
        p.error('invalid function symbol')
    if 'kernel.launch' in text:
        p.error('workspace planning requires already lowered specification calls')
    proof = json.loads(a.match_proof.read_text())
    if proof.get('backend') != 'cuda':
        p.error('device workspace requires a CUDA matching report')
    options = ['--plan-persistent-gpu-workspace=function='+a.function,
               '--plan-gpu-data-residency=function='+a.function+' promote-function-arguments=false']
    run = subprocess.run([str(a.opt), *options], input=text, text=True,
                         capture_output=True, timeout=120)
    if run.returncode:
        raise RuntimeError(run.stderr[-4000:])
    if calls(text) != calls(run.stdout) or 'kernel.launch' in run.stdout:
        raise RuntimeError('workspace placement changed the library-call inventory')
    a.output.write_text(run.stdout)
    a.report.write_text(json.dumps(dict(
        schema=1, function=a.function, passes=options, compiler_sha256=digest(a.opt),
        input_sha256=digest(a.input), output_sha256=digest(a.output),
        verified_matching_report_sha256=digest(a.match_proof),
        calls=dict(calls(text)), gpu_allocations=run.stdout.count('gpu.alloc '),
        invocation_contract='serial calls; host-visible persistent scratch is not reentrant'),
        indent=2)+'\n')


if __name__ == '__main__':
    main()
