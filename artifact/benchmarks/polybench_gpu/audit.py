#!/usr/bin/env python3
"""Compare complete FP64 output dumps and separately report timing replication."""
import csv
import hashlib
import json
import math
from pathlib import Path
import re

from common import OUT as HERE
EXPECTED = {
    '2mm': {'D': 960000}, '3mm': {'G': 880000},
    'atax': {'y': 2100}, 'bicg': {'s': 1900, 'q': 2100},
    'correlation': {'corr': 1440000}, 'covariance': {'cov': 1440000},
    'gemm': {'C': 1100000}, 'gemver': {'w': 2000},
    'gesummv': {'y': 1300}, 'mvt': {'x1': 2000, 'x2': 2000},
    'syr2k': {'C': 1440000}, 'syrk': {'C': 1440000},
}
ATOL = RTOL = 1e-8

def arrays(path, expected):
    text = path.read_text()
    assert text.count('==BEGIN DUMP_ARRAYS==') == 1, 'missing/duplicate dump start'
    assert text.count('==END   DUMP_ARRAYS==') == 1, 'missing/duplicate dump end'
    text = text.split('==BEGIN DUMP_ARRAYS==')[1].split('==END   DUMP_ARRAYS==')[0]
    matches = re.findall(r'begin dump: (\w+)\s*(.*?)end\s+dump: \1', text, re.S)
    assert len(matches) == len(expected), 'array count'
    result = {}
    for name, body in matches:
        assert name in expected and name not in result, 'unknown/duplicate array'
        values = [float(x) for x in body.split()]
        assert len(values) == expected[name], (name, len(values), expected[name])
        assert all(math.isfinite(x) for x in values), 'nonfinite output'
        result[name] = values
    assert result.keys() == expected.keys(), 'missing array'
    return result

def compare(reference, actual):
    result = {}
    for name, refs in reference.items():
        max_abs = max_rel = 0.0
        failures = 0
        first = None
        for i, (r, a) in enumerate(zip(refs, actual[name])):
            err = abs(r-a)
            max_abs = max(max_abs, err)
            max_rel = max(max_rel, err/max(abs(r), ATOL))
            if err > ATOL + RTOL*abs(r):
                failures += 1
                if first is None:
                    first = dict(index=i, reference=r, actual=a)
        result[name] = dict(count=len(refs), failures=failures,
                            max_abs=max_abs, max_rel=max_rel, first_failure=first)
    return dict(passed=all(x['failures']==0 for x in result.values()), arrays=result)

def main():
    paper = HERE / 'paper_targets.csv'
    targets = {r['label']: r for r in csv.DictReader(paper.open())}
    frozen = json.loads((HERE/'final-provenance.json').read_text())['hashes']
    rows = []
    selected = json.loads((HERE/'cases.json').read_text())
    for kernel in selected:
        expected = EXPECTED[kernel]
        d = HERE/kernel
        row = dict(kernel=kernel, atol=ATOL, rtol=RTOL, variants={})
        try:
            ref = arrays(d/'reference.log', expected)
            ref_run = json.loads((d/'reference-run.json').read_text())
            assert ref_run['rc'] == 0, 'reference exit'
            assert ref_run['sha256'] == hashlib.sha256((d/'reference').read_bytes()).hexdigest(), 'reference binary changed'
        except (OSError, AssertionError, ValueError) as e:
            row['reference_error'] = str(e)
            rows.append(row)
            continue
        variants = ['raised', 'paper-baseline' if kernel in ('correlation', 'covariance') else 'native']
        for variant in variants:
            result = {}
            row['variants'][variant] = result
            try:
                run = json.loads((d/(variant+'-run.json')).read_text())
                result['run'] = run
                result['binary_matches_run'] = run['sha256'] == hashlib.sha256((d/variant).read_bytes()).hexdigest()
                result['correctness'] = compare(ref, arrays(d/(variant+'.log'), expected))
                result['valid_run'] = (run['rc']==0 and run['warmups']==5 and
                    run['measured']==5 and run['dump_count']==1 and run['thermal_ok'] and
                    result['binary_matches_run'] and len(run.get('compute_device_ms',{}).get('samples_ms',[]))==5)
                result['accepted'] = result['valid_run'] and result['correctness']['passed']
                current = run['compute_device_ms']['min_ms']
                target = float(targets[kernel][('raised' if variant == 'raised' else 'native')+'_min_ms'])
                result['paper_ms'] = target
                result['time_change_percent'] = 100*(current/target-1)
                if variant == 'native':
                    result['provenance'] = json.loads((d/'native-build.json').read_text())
                elif variant == 'paper-baseline':
                    result['provenance'] = json.loads((d/'paper-baseline-build.json').read_text())
            except (OSError, AssertionError, ValueError, KeyError) as e:
                result['error'] = str(e)
                result['accepted'] = False
        baseline = 'paper-baseline' if kernel in ('correlation', 'covariance') else 'native'
        row['paper_comparison_baseline'] = baseline
        try:
            trace = json.loads((d/'raised-trace-run.json').read_text())
            proof = json.loads((d/'export/match.json').read_text())
            build = json.loads((d/'build.json').read_text())
            row['frozen_proof_valid'] = (
                build['compiler_sha256'] == frozen['build/bin/polygeist-opt'] and
                build['proof_sha256'] == hashlib.sha256((d/'export/match.json').read_bytes()).hexdigest() and
                proof['library_sha256'] == frozen['generic_solver/kernel_library_v3.mlir'] and
                all(frozen.get(path) == value for path, value in proof['semantic_inputs'].items()))
            lines = trace['vendor_trace']
            row['vendor_trace'] = lines
            row['trace_correctness'] = compare(ref, arrays(d/'raised-trace.log', expected))
            staged = sum(int(value) for line in lines for value in
                         re.findall(r'\b(?:h2d|d2h)_bytes=(\d+)', line))
            calls = sum(int(re.search(r'\bcount=(\d+)', line)[1]) for line in lines)
            row['vendor_valid'] = bool(trace['rc'] == 0 and lines and staged == 0 and
                trace['sha256'] == hashlib.sha256((d/'raised').read_bytes()).hexdigest() and
                row['trace_correctness']['passed'] and
                proof['library'] == 'generic_solver/kernel_library_v3.mlir' and
                calls >= 10 * len(proof['emitted']))
            ir = (d/'gpu/gpu_resident.mlir').read_text().splitlines()
            start = next(i for i, line in enumerate(ir) if 'func.func @kernel_'+kernel+'(' in line)
            body = []
            depth = 0
            for line in ir[start:]:
                body.append(line)
                depth += line.count('{')-line.count('}')
                if depth == 0:
                    break
            text = '\n'.join(body)
            row['host_memory_operations'] = {op: text.count(op) for op in
                ('memref.load', 'memref.store', 'memref.alloc(', 'memref.alloca', 'gpu.host_register')}
            row['resident_entry_valid'] = not any(row['host_memory_operations'].values())
            row['vendor_valid'] &= row['resident_entry_valid'] and row['frozen_proof_valid']
        except (OSError, AssertionError, ValueError, KeyError, TypeError) as e:
            row['vendor_valid'] = False
            row['vendor_error'] = str(e)
        if all(row['variants'][v].get('accepted') for v in ('raised', baseline)):
            row['device_speedup'] = row['variants'][baseline]['run']['compute_device_ms']['min_ms']/row['variants']['raised']['run']['compute_device_ms']['min_ms']
            row['paper_speedup'] = float(targets[kernel]['speedup_native_over_raised'])
        rows.append(row)
    (HERE/'audit.json').write_text(json.dumps(rows, indent=2)+'\n')
    for row in rows:
        print(row['kernel'], ' '.join(v+':'+('PASS' if r.get('accepted') else 'PENDING/FAIL') for v,r in row['variants'].items()),row.get('reference_error',''), flush=True)
    raise SystemExit(not all(not r.get('reference_error') and r['variants'] and r.get('vendor_valid') and
                             all(v.get('accepted') for v in r['variants'].values())
                             for r in rows))

if __name__ == '__main__':
    main()
