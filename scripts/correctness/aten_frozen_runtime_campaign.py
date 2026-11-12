#!/usr/bin/env python3
"""Measure saved ATen executables without rebuilding their compiler/runtime code.

Prepare copies all link inputs and changes only TIMING_SAMPLES in the saved
harness. Run requires the original paper-input correctness result; a failed gate
stays failed and receives no validated timing. New runs collect five alternating
pairs. Reports can also read historical longer runs without relabeling their
statistics, with current and historical native denominators.
"""
import argparse
import concurrent.futures
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time


def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    temporary = p.with_suffix(p.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(p)


def check_sample_count(record, requested, label):
    count = record.get('timing_samples', 5)
    if count != requested:
        raise ValueError(f'{label}: harness has {count} samples, requested {requested}; '
                         'prepare a new five-sample snapshot before execution')
    return count


def check_harness_count(directory, requested):
    definitions = re.findall(r'^#define TIMING_SAMPLES (\d+)$',
                             (directory / 'harness.c').read_text(), re.M)
    if definitions != [str(requested)]:
        raise ValueError(f'{directory}: saved harness does not define exactly '
                         f'{requested} timing samples')


def measured_sample_count(timing):
    counts = [sum(s['path'] == path for s in timing.get('samples', []))
              for path in ('raised', 'native_aten_cuda')]
    if not counts[0] or counts[0] != counts[1]:
        raise ValueError('Timing result does not contain paired samples')
    return counts[0]


def prepare_one(source, output, timing_samples=5):
    old = json.loads((source / 'build.json').read_text())
    assert old['rc'] == 0 and digest(source / 'raised-program') == old['binary_sha256']
    dest = output / old['kernel']
    dest.mkdir(parents=True, exist_ok=False)
    shutil.copy2(source / 'build.json', dest / 'original-build.json')
    shutil.copy2(source / 'raised-program', dest / 'original-program')
    command = old['commands'][-1]
    objects = [Path(x) for x in command if x.endswith('.o')]
    harness_objects = [obj for obj in objects if obj.name in ('harness.o', 'timing-harness.o')]
    if len(harness_objects) != 1:
        raise ValueError('Expected one saved harness link input: ' + str(source))
    replacements, hashes = {}, {}
    for obj in objects:
        target = dest / ('work' if obj.parent.name == 'work' else 'objects') / obj.name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(obj, target)
        assert digest(target) == digest(obj)
        replacements[str(obj)] = str(target)
        hashes[str(target.relative_to(dest))] = digest(target)
    harness = (source / 'harness.c').read_text()
    count = old['configuration'].get('timing_samples', 5)
    assert harness.count('#define TIMING_SAMPLES ' + str(count) + '\n') == 1
    (dest / 'original-harness.c').write_text(harness)
    (dest / 'harness.c').write_text(harness.replace(
        '#define TIMING_SAMPLES ' + str(count) + '\n',
        '#define TIMING_SAMPLES ' + str(timing_samples) + '\n'))
    commands = []
    if count == timing_samples:
        shutil.copy2(source / 'raised-program', dest / 'timing-program')
    else:
        # Match polygeist_build.sh's dedicated harness compilation flags.
        includes = [x for x in old['commands'][2] if x.startswith('-I')]
        compile_command = ['aarch64-linux-gnu-gcc', *includes, '-O3', '-fno-inline',
            '-fno-inline-functions', '-fsemantic-interposition', '-fno-ipa-cp',
            '-fno-ipa-cp-clone', '-fno-ipa-sra', '-c', str(dest / 'harness.c'),
            '-o', str(dest / 'timing-harness.o')]
        link_command = [replacements.get(x, x) for x in command]
        saved_harness_object = replacements[str(harness_objects[0])]
        link_command = [str(dest / 'timing-harness.o') if x == saved_harness_object
                        else x for x in link_command]
        link_command[link_command.index('-o') + 1] = str(dest / 'timing-program')
        commands = [compile_command, link_command]
        with (dest / 'prepare.log').open('w') as log:
            for cmd in commands:
                log.write(shlex.join(cmd) + '\n'); log.flush()
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                               check=True, timeout=120)
    row = dict(kernel=old['kernel'], original_binary_sha256=old['binary_sha256'],
        timing_binary_sha256=digest(dest / 'timing-program'),
        original_harness_sha256=digest(dest / 'original-harness.c'),
        timing_harness_sha256=digest(dest / 'harness.c'),
        frozen_object_sha256=hashes, original_configuration=old['configuration'],
        original_proof_sha256=old['proof_sha256'], timing_samples=timing_samples,
        adaptation=f'Only the saved harness TIMING_SAMPLES changes to {timing_samples}; '
                   'all application, compiler runtime, C reference and native ATen objects are frozen.',
        commands=commands)
    save(dest / 'prepared.json', row)
    return row


def transport(path):
    spec = importlib.util.spec_from_file_location('runtime_board', str(path))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def remote_libraries(board, program, environment, output, label):
    # Resolve and hash on the board: transport output may redact home paths.
    script = """import hashlib,json,re,subprocess
program = PROGRAM
result = subprocess.run(['ldd', program], text=True, capture_output=True)
log = result.stdout + result.stderr
if result.returncode or 'not found' in log:
    raise RuntimeError('Dynamic library resolution failed: ' + log)
paths = sorted(set(re.findall(r'(?:=>\\s+|^\\s*)(/\\S+)\\s+\\(', log, re.M)))
if not paths:
    raise RuntimeError('No dynamic library paths found')
hashes = []
for path in paths:
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    hashes.append(h.hexdigest() + '  ' + path)
print(json.dumps(dict(ldd=log, hashes=hashes)))
""".replace('PROGRAM', repr(program))
    rc, text = board.run(environment + "python3 - <<'PY'\n" + script + '\nPY', timeout=180)
    if rc:
        (output / ('libraries-' + label + '-error.log')).write_text(text)
        raise RuntimeError('Dynamic library hashing failed')
    result = json.loads(text)
    (output / ('libraries-' + label + '.ldd')).write_text(result['ldd'])
    hashes = '\n'.join(result['hashes']) + '\n'
    (output / ('libraries-' + label + '.sha256')).write_text(hashes)
    return hashes


def parse_timing(rc, log, timing_samples=5):
    samples = [dict(path=a, sample=int(b), ms=float(c)) for a, b, c in
               re.findall(r'SAMPLE path=(\w+) sample=(\d+) ms=([\deE.+-]+)', log)]
    paths = ('raised', 'native_aten_cuda')
    valid = rc == 0 and 'PASS timing' in log and len(samples) == 2 * timing_samples and all(
        sorted(s['sample'] for s in samples if s['path'] == p) == list(range(timing_samples))
        for p in paths)
    result = dict(rc=rc, passed=valid, samples=samples, timing_samples=timing_samples,
                  statistic=f'{timing_samples} synchronized alternating samples, '
                            'five warmups per path; resets outside timing')
    if valid:
        result['minimum_ms'] = {p: min(s['ms'] for s in samples if s['path'] == p) for p in paths}
        result['minimum_first_five_ms'] = {p: min(s['ms'] for s in samples
            if s['path'] == p and s['sample'] < 5) for p in paths}
    return result


def report(args):
    paper = {r['kernel']: r for r in csv.DictReader(args.paper.open())}
    rows = []
    records = {p.parent.name: p for p in args.output.glob('*/runtime-result.json')}
    superseded = {}
    for directory in getattr(args, 'override', []):
        for replacement in directory.glob('*/runtime-result.json'):
            name = replacement.parent.name
            if name not in records:
                raise ValueError('Override is outside original cohort: ' + name)
            superseded[name] = str(records[name]); records[name] = replacement
    for name, p in sorted(records.items()):
        r = json.loads(p.read_text()); old = paper[r['kernel']]
        r['runtime_result_source'] = str(p)
        if name in superseded:
            r['supersedes_result'] = superseded[name]
        r['paper'] = old
        r['paper_shape_exact'] = not (r.get('shape_record') or {}).get('correction', False)
        timing = r.get('timing', {})
        if timing.get('passed'):
            count = measured_sample_count(timing)
            paper_count = 20 if '20' in old['timing_protocol'] else 5
            statistic = 'minimum_ms' if count == paper_count else 'minimum_first_five_ms'
            selected_count = count if statistic == 'minimum_ms' else min(count, 5)
            current = timing[statistic]
            raised_us = 1000 * current['raised']; native_us = 1000 * current['native_aten_cuda']
            r['comparison'] = dict(selected_statistic=statistic,
                measured_samples_per_path=count, selected_samples_per_path=selected_count,
                paper_samples_per_path=paper_count,
                timing_statistic_matches_paper=selected_count == paper_count,
                raised_us=raised_us, native_us=native_us,
                current_native_over_raised=native_us / raised_us,
                paper_native_over_raised=float(old['native_us']) / float(old['raised_us']),
                raised_runtime_over_paper=raised_us / float(old['raised_us']),
                paper_native_over_current_raised=float(old['native_us']) / raised_us,
                baseline_note='Current native is freshly measured real ATen CUDA; '
                    'paper-native ratio uses the recorded historical baseline, not a new measurement.')
        rows.append(r)
    save(args.output / 'runtime-results.json', rows)
    summary = dict(cases=len(rows), correctness_passed=sum(r['correctness_passed'] for r in rows),
        timings_passed=sum(r.get('timing', {}).get('passed', False) for r in rows),
        validated_timings=sum(r['correctness_passed'] and r.get('timing', {}).get('passed', False) for r in rows),
        diagnostic_timings=sum(not r['correctness_passed'] and r.get('timing', {}).get('passed', False) for r in rows),
        failures=[dict(kernel=r['kernel'], status=r['status']) for r in rows
                  if not r.get('timing', {}).get('passed', False)])
    valid = [r for r in rows if r['correctness_passed'] and r.get('timing', {}).get('passed') and r['paper_shape_exact']]
    summary['validated_exact_shape_timings'] = len(valid)
    summary['correctness_failures'] = [r['kernel'] for r in rows if not r['correctness_passed']]
    summary['shape_corrections'] = [dict(kernel=r['kernel'],
        correction=r['shape_record']['correction'], recorded_dims=r['shape_record'].get('recorded_dims'),
        measured_dims=r['shape_record']['dims']) for r in rows if (r.get('shape_record') or {}).get('correction')]
    summary['measured_sample_counts'] = sorted({r['comparison']['measured_samples_per_path']
        for r in rows if 'comparison' in r})
    summary['validated_exact_shape_and_statistic_timings'] = sum(
        r['comparison']['timing_statistic_matches_paper'] for r in valid)
    summary['scope'] = ('Paper input recipe and both host/resident pointer correctness; '
        'alternating resident samples after five warmups. Each comparison records its '
        'actual sample count and whether its minimum statistic matches the paper. '
        'Correctness failures are excluded from validated performance counts.')
    summary['paper_runtime_comparison'] = dict(
        at_least_as_fast=sum(r['comparison']['raised_runtime_over_paper'] <= 1 for r in valid),
        slower_within_five_percent=sum(1 < r['comparison']['raised_runtime_over_paper'] <= 1.05 for r in valid),
        slower_five_to_twenty_percent=sum(1.05 < r['comparison']['raised_runtime_over_paper'] <= 1.2 for r in valid),
        slower_over_twenty_percent=sum(r['comparison']['raised_runtime_over_paper'] > 1.2 for r in valid))
    summary['largest_runtime_regressions'] = [dict(kernel=r['kernel'],
        ratio=r['comparison']['raised_runtime_over_paper']) for r in
        sorted(valid, key=lambda r: r['comparison']['raised_runtime_over_paper'], reverse=True)[:10]]
    save(args.output / 'runtime-status.json', summary)
    fields = ['kernel', 'status', 'correctness_passed', 'paper_shape_exact', 'paper_raised_us', 'paper_native_us',
              'paper_protocol', 'raised_us', 'native_us', 'current_native_over_raised',
              'paper_native_over_raised', 'raised_runtime_over_paper',
              'paper_native_over_current_raised', 'selected_statistic',
              'measured_samples_per_path', 'selected_samples_per_path',
              'paper_samples_per_path', 'timing_statistic_matches_paper']
    with (args.output / 'runtime-comparison.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for r in rows:
            row = {k: r[k] for k in ('kernel', 'status', 'correctness_passed', 'paper_shape_exact')}
            row.update(paper_raised_us=r['paper']['raised_us'], paper_native_us=r['paper']['native_us'],
                       paper_protocol=r['paper']['timing_protocol'])
            row.update({k: v for k, v in r.get('comparison', {}).items() if k in fields})
            writer.writerow(row)
    return summary


def run(args):
    environment = 'LD_LIBRARY_PATH=' + shlex.quote(args.remote_library_path) + ' '
    if args.wait_pid:
        while Path('/proc/' + str(args.wait_pid) + '/cmdline').exists():
            try:
                if args.wait_marker.encode() not in Path('/proc/' + str(args.wait_pid) + '/cmdline').read_bytes():
                    break
            except OSError:
                break
            time.sleep(10)
    names = [p.parent.name for p in sorted(args.output.glob('*/prepared.json'))]
    for name in names:
        dest = args.output / name
        check_sample_count(json.loads((dest / 'prepared.json').read_text()),
                           args.timing_samples, name)
        check_harness_count(dest, args.timing_samples)
    board = transport(args.transport)
    (args.output / 'host-uptime.txt').write_text(subprocess.check_output(['uptime'], text=True))
    probe = args.output / 'board-preflight.py'
    initial_hashes = None; first_remote = None
    for name in names:
        dest = args.output / name
        if (dest / 'runtime-result.json').exists():
            continue
        prepared = json.loads((dest / 'prepared.json').read_text())
        correctness_path = args.builds / name / 'gpu-result.json'
        correctness = json.loads(correctness_path.read_text()) if correctness_path.exists() else {}
        passed = correctness.get('fully_validated', False)
        if correctness:
            assert correctness.get('binary_sha256') == prepared['original_binary_sha256'], name
        row = dict(kernel=name, correctness_passed=passed, correctness=correctness,
            original_binary_sha256=prepared['original_binary_sha256'],
            timing_binary_sha256=prepared['timing_binary_sha256'],
            status=('correctness_missing' if not correctness else 'correctness_failed') if not passed else 'timing_pending',
            correctness_basis='Original saved binary paper-input check; all computation objects preserved. '
                'The timing harness differs only in sample count.',
            correctness_evidence=str(correctness_path),
            shape_record=json.loads((dest / 'original-build.json').read_text()).get('shape_record'))
        diagnostic = bool(correctness) and not passed and name in args.diagnostic_failed
        row['diagnostic_only'] = diagnostic
        row['baseline_scope'] = 'Real ATen CUDA calls for the extracted fixture; preallocated outputs.'
        if name == 'aten_xor_sum_cpu':
            row['baseline_caveat'] = 'Installed ATen lacks the newer xor_sum CUDA entry point; baseline is the saved composition of real ATen operations, not that kernel.'
        if passed or diagnostic:
            # Reuse measurements only for an identical binary and requested count.
            if (prepared['timing_binary_sha256'] == prepared['original_binary_sha256'] and
                    correctness.get('timing', {}).get('passed') and
                    measured_sample_count(correctness['timing']) == args.timing_samples):
                row['timing'] = correctness['timing']; row['timing_reused_from'] = str(correctness_path)
            else:
                rc, preflight = board.run("python3 - <<'PY'\n" + probe.read_text() + '\nPY', timeout=40)
                (dest / 'board-before.json').write_text(preflight)
                if rc:
                    raise RuntimeError('Board not idle before ' + name)
                binary = dest / 'timing-program'
                assert digest(binary) == prepared['timing_binary_sha256']
                remote = '/tmp/ll-aten-runtime-' + name + '-' + digest(binary)[:12]
                if board.upload(binary, remote):
                    raise RuntimeError('Upload failed: ' + name)
                rc, sha = board.run('chmod +x ' + shlex.quote(remote) + ' && sha256sum ' + shlex.quote(remote), timeout=20)
                (dest / 'remote-hash.log').write_text(sha)
                if rc or digest(binary) not in sha:
                    raise RuntimeError('Remote executable differs: ' + name)
                if initial_hashes is None:
                    rc, hardware = board.run("python3 - <<'PY'\nimport json,platform,subprocess\nfrom pathlib import Path\np=Path('/proc/device-tree/model')\nprint(json.dumps(dict(model=p.read_text().strip(chr(0)) if p.exists() else 'unavailable',uname=platform.uname()._asdict(),uptime=subprocess.check_output(['uptime'],text=True))))\nPY", timeout=20)
                    (args.output / 'board-hardware.json').write_text(hardware)
                    if rc: raise RuntimeError('Hardware capture failed')
                    initial_hashes = remote_libraries(board, remote, environment, args.output, 'before')
                    first_remote = remote
                rc, log = board.run(environment + 'timeout ' + str(args.timeout) + ' ' + shlex.quote(remote) + ' timing', timeout=args.timeout + 5)
                (dest / 'timing-gpu.log').write_text(log)
                row['timing'] = parse_timing(rc, log, args.timing_samples)
            row['status'] = ('diagnostic_timed' if diagnostic else 'timed') if row['timing']['passed'] else 'timing_failed'
        save(dest / 'runtime-result.json', row)
        status = report(args)
        print(name, row['status'], 'progress', status['cases'], '/', len(names), flush=True)
    if initial_hashes is not None:
        final_hashes = remote_libraries(board, first_remote, environment, args.output, 'after')
        save(args.output / 'dynamic-library-stability.json', dict(unchanged=initial_hashes == final_hashes))
        if initial_hashes != final_hashes:
            raise RuntimeError('Dynamic libraries changed during campaign')
    print(json.dumps(report(args)), flush=True)


def replace_runtime(args):
    records = sorted(args.frozen.glob('*/prepared.json'))
    for record in records:
        check_sample_count(json.loads(record.read_text()), args.timing_samples,
                           str(record.parent))
        check_harness_count(record.parent, args.timing_samples)
        if (args.output / record.parent.name).exists():
            raise FileExistsError('Refusing to overwrite saved snapshot: ' +
                                  str(args.output / record.parent.name))
    metadata = json.loads(args.runtime_manifest.read_text())
    expected_source = metadata['runtime_source_sha256']
    replacement = args.output / 'shared/spec_runtime.o'
    replacement.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.runtime_object, replacement)
    save(args.output / 'runtime-replacement.json', dict(metadata=metadata,
        object_sha256=digest(replacement), source_manifest=str(args.runtime_manifest)))
    rows = []
    for old_record in records:
        old_dir = old_record.parent; name = old_dir.name; dest = args.output / name
        original_source = args.builds / name / 'work/spec_runtime/polygeist_v2_rt_cuda.c'
        if digest(original_source) != expected_source:
            raise RuntimeError('Generated runtime source mismatch: ' + name)
        shutil.copytree(old_dir, dest)
        for stale in ['runtime-result.json', 'gpu-result.json', 'timing-gpu.log']:
            (dest / stale).unlink(missing_ok=True)
        original = json.loads((dest / 'original-build.json').read_text())
        old = json.loads(old_record.read_text())
        command = original['commands'][-1]
        updated = []
        for item in command:
            path = Path(item)
            if item.endswith('.o'):
                if path.name == 'spec_runtime.o':
                    item = str(replacement)
                elif path.name == 'harness.o' and (dest / 'timing-harness.o').exists():
                    item = str(dest / 'timing-harness.o')
                else:
                    item = str(dest / ('work' if path.parent.name == 'work' else 'objects') / path.name)
            updated.append(item)
        program = dest / 'timing-program'
        updated[updated.index('-o') + 1] = str(program)
        with (dest / 'relink.log').open('w') as log:
            subprocess.run(updated, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
        new_hash = digest(program)
        shutil.copy2(program, dest / 'raised-program')
        built = dict(original, binary_sha256=new_hash, prior_binary_sha256=original['binary_sha256'])
        built['configuration'] = dict(original['configuration'], timing_samples=args.timing_samples)
        built['commands'] = original['commands'][:-1] + [updated]
        built['runtime_replacement'] = dict(object_sha256=digest(replacement),
            generated_source_sha256=expected_source, copy_helper_sha256=metadata['copy_helper_sha256'])
        save(dest / 'build.json', built)
        prepared = dict(old, original_binary_sha256=new_hash, timing_binary_sha256=new_hash,
            prior_original_binary_sha256=old['original_binary_sha256'],
            prior_timing_binary_sha256=old['timing_binary_sha256'],
            original_configuration=built['configuration'], commands=[updated],
            adaptation='Only the generated spec_runtime.o changes to the bit-tested generic copy runtime; '
                       'all application, native reference and vendor objects remain frozen; '
                       f'{args.timing_samples}-sample harness retained.',
            runtime_replacement=built['runtime_replacement'])
        save(dest / 'prepared.json', prepared); rows.append(prepared)
    shutil.copy2(args.frozen / 'board-preflight.py', args.output / 'board-preflight.py')
    save(args.output / 'prepared.json', rows)
    print('Relinked', len(rows), 'cases with one verified shared runtime replacement', flush=True)


def validate(args):
    # Use the established unchanged correctness gates on the replacement binaries.
    from types import SimpleNamespace
    import aten_fixture_batch as fixtures
    names = [p.parent.name for p in sorted(args.output.glob('*/build.json'))]
    for name in names:
        dest = args.output / name
        check_sample_count(json.loads((dest / 'build.json').read_text())['configuration'],
                           args.timing_samples, name)
        check_harness_count(dest, args.timing_samples)
    board = transport(args.transport)
    first = args.output / names[0] / 'raised-program'
    remote = '/tmp/ll-aten-library-snapshot-' + digest(first)[:12]
    if board.upload(first, remote):
        raise RuntimeError('Library snapshot executable upload failed')
    rc, _ = board.run('chmod +x ' + shlex.quote(remote), timeout=15)
    if rc:
        raise RuntimeError('Library snapshot executable setup failed')
    environment = 'LD_LIBRARY_PATH=' + shlex.quote(args.remote_library_path) + ' '
    before = remote_libraries(board, remote, environment, args.output, 'validation-before')
    options = SimpleNamespace(output=args.output, transport=str(args.transport),
        remote_library_path=args.remote_library_path, cudnn_plan_cache=False,
        run_timeout=args.timeout, correctness_kinds=['paper'], skip_timing=False,
        time_validated_normal_inputs=False)
    fixtures.run_gpu(names, options)
    after = remote_libraries(board, remote, environment, args.output, 'validation-after')
    save(args.output / 'validation-library-stability.json', dict(unchanged=before == after))
    if before != after:
        raise RuntimeError('Dynamic libraries changed during validation and timings')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'replace-runtime', 'validate', 'run', 'report'])
    p.add_argument('--builds', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--paper', type=Path, required=True)
    p.add_argument('--transport', type=Path)
    p.add_argument('--frozen', type=Path)
    p.add_argument('--override', type=Path, action='append', default=[],
        help='For reports, replace a cohort result with a separately preserved repair run.')
    p.add_argument('--runtime-object', type=Path)
    p.add_argument('--runtime-manifest', type=Path)
    p.add_argument('--remote-library-path')
    p.add_argument('--timeout', type=int, default=600)
    p.add_argument('--timing-samples', type=int, choices=[5], default=5,
        help='Future campaigns use five pairs; report also reads historical longer runs.')
    p.add_argument('--diagnostic-failed', nargs='*', default=[],
        help='Named cases may be timed despite an existing failed correctness gate; never counted as validated.')
    p.add_argument('--wait-pid', type=int)
    p.add_argument('--wait-marker', default='/tmp/aten86-run-final.py')
    args = p.parse_args()
    args.builds = args.builds.resolve(); args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.action == 'prepare':
        source = Path(__file__).resolve().parent / 'fig3_board_preflight.py'
        shutil.copy2(source, args.output / 'board-preflight.py')
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            rows = list(pool.map(lambda b: prepare_one(b.parent, args.output, args.timing_samples),
                                 sorted(args.builds.glob('*/build.json'))))
        save(args.output / 'prepared.json', rows)
        print('Prepared', len(rows), 'immutable executables', flush=True)
    elif args.action == 'replace-runtime':
        if not args.frozen or not args.runtime_object or not args.runtime_manifest:
            p.error('replace-runtime requires frozen, runtime-object and runtime-manifest')
        args.frozen = args.frozen.resolve()
        replace_runtime(args)
    elif args.action == 'validate':
        if not args.transport or not args.remote_library_path:
            p.error('validate requires transport and remote-library-path')
        validate(args)
    elif args.action == 'run':
        if not args.transport or not args.remote_library_path:
            p.error('run requires transport and remote-library-path')
        run(args)
    else:
        print(json.dumps(report(args)))


if __name__ == '__main__':
    main()
