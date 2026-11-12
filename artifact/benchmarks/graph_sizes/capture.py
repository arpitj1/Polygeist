#!/usr/bin/env python3
"""Collect current production V3 proof graphs without changing compiler sources."""
import argparse
import concurrent.futures
import csv
import dataclasses
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

from common import ROOT, OUT, INPUTS, required
SPEQ = Path(required('speq_root'))


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str) + '\n')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def manifest():
    rows = []
    for row in csv.DictReader((INPUTS / 'polybench_manifest.csv').open()):
        src = ROOT / row['source']
        rows.append(dict(name=row['kernel'], source=str(src),
                         function='kernel_' + row['kernel'].replace('-', '_'),
                         flags=['-I' + str(ROOT / 'tools/cgeist/Test/polybench/utilities'),
                                '-I' + str(src.parent), '-Dstatic=__attribute__((noipa))',
                                '-DLARGE_DATASET', '-DDATA_TYPE_IS_DOUBLE',
                                '-DPOLYBENCH_USE_C99_PROTO', '-DPOLYBENCH_DUMP_ARRAYS'],
                         scope='polybench'))
    rows.append(dict(name='gemm_ref_paired', source=str(SPEQ / 'lleq-artifact/benchmarks/gemm_ref.c'),
                     function='gemm_ref', flags=[], scope='paired_gemm'))
    return rows


def worker(name):
    row = next(r for r in manifest() if r['name'] == name)
    dest = OUT / 'll' / name
    dest.mkdir(parents=True, exist_ok=True)
    os.environ.update(POLYGEIST_SPEC_SUITE='lib_v3', POLYGEIST_NUMERIC_MODEL='relaxed',
                      POLYGEIST_KERNEL_LIBRARY=str(ROOT / 'generic_solver/kernel_library_v3.mlir'),
                      EQSAT_ABLATION='full', REGION_MATCH_NO_CACHE='1',
                      REGION_MATCH_CACHE_DIR=str(dest / 'cache'))
    sys.path[:0] = [str(ROOT / 'scripts/correctness'), str(ROOT / 'scripts/correctness/lib_v2')]
    import recipe
    import spec_match as S
    import kernel_match as K
    import region_match.matcher as RM
    import region_match.egg as RE

    proofs = []
    active = []
    context = dict(spec=None, phase='application')
    configure = K.configure_matcher
    def config(*args, **kwargs):
        kwargs['collect_egraph_sizes'] = True
        return configure(*args, **kwargs)
    K.configure_matcher = config

    serialize = K.EGraph._serialize
    def capture(self, *args, **kwargs):
        result = serialize(self, *args, **kwargs)
        if active:
            raw = result.to_json()
            data = json.loads(raw)
            proof = active[-1]
            path = dest / 'graphs' / f"proof-{proof['id']:04d}-{len(proof['graphs'])}.json"
            path.parent.mkdir(exist_ok=True)
            path.write_text(raw + '\n')
            proof['graphs'].append(dict(path=str(path.relative_to(OUT)),
                                        nodes=len(data.get('nodes', {})),
                                        classes=len(data.get('class_data', {})),
                                        function_nodes=sum(n for _, n in self.all_function_sizes()),
                                        serialized_function_nodes=sum(k.startswith('function-') for k in data.get('nodes', {})),
                                        primitive_nodes=sum(k.startswith('primitive-') for k in data.get('nodes', {})),
                                        serialize_options=kwargs))
        return result
    K.EGraph._serialize = capture

    def begin(level, a, b):
        p = dict(id=len(proofs), level=level, **context, left=repr(a), right=repr(b), graphs=[])
        proofs.append(p)
        active.append(p)
        return p

    equivalent = K.equivalent
    def scalar(a, b, include_distributivity=True):
        p = begin('linalg', a, b)
        before = dict(K._MATCHER_TELEMETRY)
        t0 = time.monotonic()
        try:
            result = equivalent(a, b, include_distributivity)
            p.update(ok=result, distributivity=include_distributivity,
                     counters={k: v-before.get(k, 0) for k, v in K._MATCHER_TELEMETRY.items()
                               if isinstance(v, (int, float))})
            return result
        finally:
            p['wall_seconds_with_capture'] = time.monotonic() - t0
            active.pop()
    K.equivalent = scalar

    check_equal = RM.check_equal
    def region(a, b, *args, **kwargs):
        p = begin('region', a, b)
        try:
            result = check_equal(a, b, *args, **kwargs)
            p.update(ok=result.ok, nodes=result.nodes, classes=result.classes,
                     proof_ms=result.ms, iterations=result.iters)
            return result
        finally:
            active.pop()
    RM.check_equal = region
    RE.check_equal = region

    find = S.MATCH.Matcher.find
    def find_traced(self, spec):
        previous = context['spec']
        context['spec'] = spec.name
        try:
            yield from find(self, spec)
        finally:
            context['spec'] = previous
    S.MATCH.Matcher.find = find_traced
    load = RM.load_specs
    def load_traced(*args, **kwargs):
        previous = context['phase']
        context['phase'] = 'library_validation'
        try:
            return load(*args, **kwargs)
        finally:
            context['phase'] = previous
    RM.load_specs = load_traced

    result = dict(**row, source_sha256=sha(Path(row['source'])),
                  configuration=dict(spec_suite='lib_v3', numeric_model='relaxed',
                                     backend='cuda', eqsat='full', region_budget_seconds=60,
                                     region_disk_cache=False, recipe='production'),
                  error=None)
    started = time.monotonic()
    try:
        raised = recipe.raise_c(recipe.PRODUCTION, row['source'], row['function'], str(dest/'source'), row['flags'], 180)
        save(dest/'raise.json', {k: dataclasses.asdict(v) if dataclasses.is_dataclass(v) else v for k,v in raised.items()})
        if not all(raised.get(k) and raised[k].ok for k in ['cgeist','raise','debuf']):
            raise RuntimeError('Raising failed; see raise.json')
        text = Path(raised['paths']['debuf']).read_text()
        result['input_sha256'] = hashlib.sha256(text.encode()).hexdigest()
        matched, proof = S.match(text, backend='cuda')
        (dest/'matched.mlir').write_text(matched)
        save(dest/'match.json', proof)
        result.update(emitted=proof['emitted'], diagnostics=proof['diagnostics'])
    except Exception:
        result['error'] = traceback.format_exc()
    result.update(proofs=proofs, elapsed_seconds=time.monotonic()-started)
    save(dest/'result.json', result)
    print(name, 'error' if result['error'] else 'ok', len(proofs), result.get('emitted', []), flush=True)
    return int(result['error'] is not None)


if __name__ == '__main__':
    sys.exit(worker(sys.argv[1]))
