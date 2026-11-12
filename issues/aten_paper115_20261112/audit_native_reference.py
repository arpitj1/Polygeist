"""Recheck frozen ATen executables with the independent native/C counter.

Only the harness is rebuilt. Every reused object is hashed; this is explicitly
an audit of earlier compiler/runtime snapshots, not a fresh compiler campaign.
"""
import argparse
import concurrent.futures
import json
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

D = Path(__file__).resolve().parent
R = D.parents[1]
sys.path.insert(0, str(R / 'scripts/correctness'))
import aten_fixture_batch as B


def rebuild(name, old_dir, output):
    old = json.loads((old_dir / 'build.json').read_text())
    assert old['rc'] == 0
    dest = output / name
    dest.mkdir(parents=True, exist_ok=True)
    src = dest / 'harness.c'
    src.write_text(B.harness(name, old['configuration']))
    compile_cmd = ['aarch64-linux-gnu-gcc', '-O2', '-fno-fast-math',
                   '-ffp-contract=off', '-I' + str(B.CUDA / 'include'),
                   '-c', str(src), '-o', str(dest / 'harness.o')]
    link = list(old['commands'][-1])
    reused = {}
    harness_count = 0
    for i, token in enumerate(link):
        if token.endswith('/harness.o'):
            link[i] = str(dest / 'harness.o')
            harness_count += 1
        elif token.endswith('.o'):
            reused[token] = B.sha(Path(token))
    assert harness_count == 1, 'Expected exactly one original harness object'
    link[link.index('-o') + 1] = str(dest / 'raised-program')
    with (dest / 'build.log').open('w') as log:
        for cmd in (compile_cmd, link):
            log.write(shlex.join(cmd) + '\n')
            log.flush()
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=120)
    for path, digest in reused.items():
        assert B.sha(Path(path)) == digest, 'Reused object changed during audit'
    row = dict(old, commands=[compile_cmd, link],
               binary_sha256=B.sha(dest / 'raised-program'),
               audit_scope='Frozen original compiler/runtime/native objects; strengthened harness only',
               original_build=str(old_dir / 'build.json'),
               original_binary_sha256=old['binary_sha256'],
               reused_object_sha256=reused, harness_sha256=B.sha(src))
    B.save(dest / 'build.json', row)
    print(name, 'audit build PASS', flush=True)
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['build', 'run'], required=True)
    p.add_argument('--output', type=Path, default=D / 'gpu-native-reference-audit')
    p.add_argument('--transport')
    p.add_argument('--jobs', type=int, default=3)
    args = p.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    additions = json.loads((D / 'validated-additions.json').read_text())['new_cases']
    names = [row[0] for row in json.loads((D / 'native-reference-counter-audit.json').read_text())]
    # All sixty source runs used the same default (uncached) cuDNN policy.
    for name in names:
        rows = json.loads((D / additions[name] / 'gpu-results.json').read_text())
        prior = next(row for row in rows if row['kernel'] == name)
        assert not prior.get('cudnn_plan_cache', False)
    if args.stage == 'build':
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            rows = list(pool.map(lambda name: rebuild(name, D / additions[name] / name,
                                                     args.output), names))
        B.save(args.output / 'build-results.json', rows)
    else:
        run_args = SimpleNamespace(
            output=args.output, transport=args.transport, cudnn_plan_cache=False,
            time_validated_normal_inputs=False,
            remote_library_path='/home/nvidia/polygeist_aten_native_24_12:/home/nvidia/polygeist_cuda_libs:/home/nvidia/jetson-cuda-libs:/home/nvidia/cuda-12.6/lib64:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu')
        B.run_gpu(names, run_args)
        rows = json.loads((args.output / 'gpu-results.json').read_text())
        assert len(rows) == len(names)
        B.save(args.output / 'audit-summary.json', {
            'scope': 'Earlier frozen objects, strengthened harness; original small fixtures',
            'total': len(rows),
            'three_way_passed': sum(r['fully_validated'] and r['native_matches_reference'] is True for r in rows),
            'failed': [r['kernel'] for r in rows if not r['fully_validated'] or r['native_matches_reference'] is not True]})


if __name__ == '__main__':
    main()
