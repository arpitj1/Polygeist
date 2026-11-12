#!/usr/bin/env python3
"""Checkpoint the paper's ATen fixtures through the specification-only pipeline.

This is a diagnostic, not a GPU correctness or performance claim. Legacy
recognition is recorded only as a hint for investigating a missing proof.
Each case runs in an isolated process and retains its C/MLIR/proof/ABI evidence.
"""
import argparse
import concurrent.futures
import csv
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def counts(text):
    return dict(linalg=len(re.findall(r'\blinalg\.(?:generic|reduce|map|matmul|batch_matmul)\b', text)),
                loops=len(re.findall(r'\b(?:affine|scf)\.(?:for|parallel|while)\b', text)),
                copies=len(re.findall(r'\bmemref\.copy\b', text)),
                launches=re.findall(r'\bkernel\.launch\s+@([\w.$-]+)', text))


def worker(request):
    req = json.loads(request.read_text())
    dest = request.parent
    sys.path[:0] = [str(HERE), str(HERE / 'lib_v2')]
    os.environ.update(POLYGEIST_SPEC_SUITE='lib_v3',
                     POLYGEIST_NUMERIC_MODEL=req['numeric_model'],
                     OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                     REGION_MATCH_CACHE_DIR=str(dest.parent / 'region-cache'))
    import recipe
    import spec_match as S
    import lower_v2
    result = dict(kernel=req['kernel'], family=req['paper_family'],
                  source=req['source'], source_sha256=sha(ROOT / req['source']),
                  legacy_match=req.get('legacy_match'), variants=[],
                  correctness='not_run', performance='not_run')
    started = time.monotonic()
    try:
        raised = recipe.raise_c(recipe.PRODUCTION, str(ROOT / req['source']),
                                req['kernel'], str(dest / 'source'), [], 120)
        save(dest / 'raise.json', {k: dataclasses.asdict(v) if dataclasses.is_dataclass(v)
                                 else v for k, v in raised.items()})
        result['stages'] = {k: dict(ok=raised[k].ok, recovery=raised[k].recovery)
                            for k in ('cgeist', 'raise', 'debuf') if k in raised}
        matching_stage = req.get('matching_stage', 'debuf')
        required = ('cgeist', 'raise', 'debuf') if matching_stage == 'debuf' else ('cgeist', 'raise')
        if not all(raised.get(k) and raised[k].ok for k in required):
            result['status'] = 'frontend_or_raising_failure'
        else:
            source = Path(raised['paths'][matching_stage]).read_text()
            result['matching_stage'] = matching_stage
            result['raised_counts'] = counts(source)
            for rank in req['ranks']:
                options = dict(backend='cuda', only=req.get('specifications'), region=req['region'],
                               region_budget=req['region_budget'],
                               split_reduction_inputs=req.get('split_reduction_inputs',False),
                               split_pointwise_inputs=req.get('split_pointwise_inputs',False))
                if rank:
                    options.update(max_operand_rank=rank, fuse_initializers=True)
                out = dest / ('rank' + str(rank)); out.mkdir(exist_ok=True)
                try:
                    matched, proof = S.match(source, **options)
                    (out / 'matched.mlir').write_text(matched)
                    save(out / 'proof.json', proof)
                    S.verify(matched, proof, backend='cuda')
                    record = dict(rank=rank, matching_stage=matching_stage,
                                  options=options, emitted=proof['emitted'],
                                  counts=counts(matched), elapsed_ms=proof['elapsed_ms'],
                                  rejected_specifications=proof['rejected_specifications'], abi_ok=False)
                    result['variants'].append(record)
                    if proof['emitted']:
                        report = {}
                        try:
                            lowered = lower_v2.lower(matched, backend='cuda', report=report)
                            (out / 'abi.mlir').write_text(lowered)
                            save(out / 'abi.json', report)
                            record['abi_ok'] = True
                        except Exception:
                            record['abi_error'] = traceback.format_exc()
                except Exception:
                    result['variants'].append(dict(rank=rank, error=traceback.format_exc()))
                save(dest / 'result.json', result)
            result['status'] = ('recognized' if any(v.get('emitted') for v in result['variants'])
                                else 'no_proven_match')
    except Exception:
        result.update(status='error', error=traceback.format_exc())
    result['seconds'] = round(time.monotonic() - started, 3)
    save(dest / 'result.json', result)


def summarize(output):
    rows = [json.loads(p.read_text()) for p in sorted(output.glob('aten_*/result.json'))]
    save(output / 'results.json', rows)
    fields = ['kernel', 'family', 'status', 'raised_linalg', 'raised_loops', 'recovered_stages',
              'best_rank', 'emitted', 'remaining_linalg', 'remaining_loops', 'abi_ok',
              'legacy_match', 'correctness', 'performance']
    with (output / 'summary.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for r in rows:
            variants = [v for v in r['variants'] if 'counts' in v]
            best = min(variants, key=lambda v: (v['counts']['linalg'],
                       v['counts']['loops'], len(v['emitted']))) if variants else {}
            writer.writerow(dict(kernel=r['kernel'], family=r['family'], status=r.get('status', 'in_progress'),
                raised_linalg=r.get('raised_counts', {}).get('linalg'),
                raised_loops=r.get('raised_counts', {}).get('loops'),
                recovered_stages=','.join(k for k,v in r.get('stages', {}).items() if v.get('recovery')),
                best_rank=best.get('rank'), emitted=','.join(best.get('emitted', [])),
                remaining_linalg=best.get('counts', {}).get('linalg'),
                remaining_loops=best.get('counts', {}).get('loops'), abi_ok=best.get('abi_ok'),
                legacy_match=r.get('legacy_match'), correctness=r['correctness'], performance=r['performance']))
    summary = dict(cases=len(rows), recognized=sum(r.get('status') == 'recognized' for r in rows),
        abi_ok=sum(any(v.get('abi_ok') for v in r['variants']) for r in rows),
        no_proven_match=sum(r.get('status') == 'no_proven_match' for r in rows),
        raising_recoveries=sum(any(v.get('recovery') for v in r.get('stages', {}).values()) for r in rows),
        error=sum(r.get('status') in ('error', 'timeout', 'frontend_or_raising_failure') for r in rows),
        GPU_correctness='not established by this diagnostic', performance='not measured')
    save(output / 'summary.json', summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--checklist', type=Path, default=ROOT / 'issues/aten_remaining_v3_20261111/paper115-reproduction-backlog.csv')
    p.add_argument('--jobs', type=int, default=3)
    p.add_argument('--ranks', type=int, nargs='+', default=[0])
    p.add_argument('--specifications', nargs='+', help='Optional library catalog subset; all proofs and ABI checks remain required')
    p.add_argument('--region', action='store_true')
    p.add_argument('--region-budget', type=int, default=30)
    p.add_argument('--matching-stage', choices=['debuf', 'linalg'], default='debuf',
                   help='Application representation supplied to the same specification matchers')
    p.add_argument('--split-reduction-inputs',action='store_true')
    p.add_argument('--split-pointwise-inputs',action='store_true')
    p.add_argument('--numeric-model', choices=['strict', 'relaxed'], default='strict')
    p.add_argument('--timeout', type=int, default=600)
    p.add_argument('--cases', nargs='+')
    p.add_argument('--case-options', type=Path,
                   help='Cohort JSON with previously validated per-case matching options')
    p.add_argument('--include-validated', action='store_true', help='Also allow earlier validated cases from the checklist')
    p.add_argument('--shape-specs', type=Path, help='Pinned benchmark dimensions for fresh source copies')
    p.add_argument('--summarize', action='store_true')
    args = p.parse_args()
    if args.worker:
        worker(args.worker); return
    if not args.output:
        p.error('--output is required')
    args.output = args.output.resolve(); args.output.mkdir(parents=True, exist_ok=True)
    if args.summarize:
        print(json.dumps(summarize(args.output))); return
    if (args.output / 'provenance.json').exists():
        p.error('use a fresh output directory to preserve previous evidence')
    old = {r['kernel']: r for r in csv.DictReader((ROOT / 'issues/aten_c_kernels/cuda_library_audit.csv').open())}
    rows = [r for r in csv.DictReader(args.checklist.open())
            if args.include_validated or r['status'] == 'not_in_current_validated_aggregate']
    if args.cases:
        wanted = {x if x.startswith('aten_') else 'aten_'+x for x in args.cases}
        rows = [r for r in rows if r['kernel'] in wanted]
        if wanted - {r['kernel'] for r in rows}:
            p.error('unknown or already validated case in --cases')
    inputs = [ROOT/'build/bin/cgeist', ROOT/'build/bin/polygeist-opt',
              ROOT/'generic_solver/lib_v3/import_manifest.json',
              ROOT/'generic_solver/kernel_library_v3.mlir', args.checklist]
    shapes = None
    if args.shape_specs:
        from aten_paper_shapes import load_shapes, materialize
        shapes = load_shapes(args.shape_specs)
        inputs.append(args.shape_specs)
        if any(r['kernel'] not in shapes for r in rows):
            p.error('Shape manifest lacks a selected kernel')
    case_options = {}
    if args.case_options:
        case_options = {r['kernel']: r.get('match_options') or {}
                        for r in json.loads(args.case_options.read_text())}
        inputs.append(args.case_options)
        if any(r['kernel'] not in case_options for r in rows):
            p.error('Case options lack a selected kernel')
    save(args.output / 'provenance.json', dict(arguments=vars(args),
        git=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        uptime=subprocess.check_output(['uptime'], text=True).strip(),
        hashes={str(x): sha(x) for x in inputs}, selected=[r['kernel'] for r in rows]))
    def one(row):
        dest = args.output / row['kernel']; dest.mkdir()
        if shapes is not None:
            original = ROOT / row['source']
            source = dest / (row['kernel']+'.c')
            materialize(original, source, shapes[row['kernel']])
            row.update(original_source=row['source'], original_source_sha256=sha(original),
                       source=str(source), shape_record=shapes[row['kernel']])
        row.update(ranks=args.ranks, region=args.region, region_budget=args.region_budget, specifications=args.specifications,
                   split_reduction_inputs=args.split_reduction_inputs,
                   split_pointwise_inputs=args.split_pointwise_inputs,
                   numeric_model=args.numeric_model, matching_stage=args.matching_stage,
                   legacy_match=old.get(row['kernel'], {}).get('current_match'))
        row.update({k: v for k, v in case_options.get(row['kernel'], {}).items()
                    if k in ('ranks', 'region', 'region_budget', 'numeric_model',
                             'split_reduction_inputs', 'split_pointwise_inputs', 'matching_stage')})
        request = dest / 'request.json'; save(request, row)
        with (dest / 'worker.log').open('w') as log:
            try:
                subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', str(request)],
                               stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout, check=True)
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
                path = dest / 'result.json'
                result = json.loads(path.read_text()) if path.exists() else dict(kernel=row['kernel'],
                    family=row['paper_family'], variants=[], correctness='not_run', performance='not_run')
                result.update(status='timeout' if isinstance(exc, subprocess.TimeoutExpired) else 'error', error=str(exc))
                save(path, result)
        result = json.loads((dest / 'result.json').read_text())
        return row['kernel'], result['status']
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for future in concurrent.futures.as_completed([pool.submit(one, row) for row in rows]):
            print(*future.result(), flush=True)
    print(json.dumps(summarize(args.output)))


if __name__ == '__main__':
    main()
