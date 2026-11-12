#!/usr/bin/env python3
"""Compare recorded paper routes with proven specifications and lowered calls.

The paper's raised_call column records matcher/wrapper names, not binary traces.
This report deliberately does not equate those names with current spec names or
infer correctness/performance from either. It never supplies matcher decisions.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper-csv', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--gpu-results', type=Path)
    parser.add_argument('--cohort', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    selected = None
    if args.cohort:
        selected = {row['kernel'] for row in read_json(args.cohort)}
    gpu = {r['kernel']: r for r in read_json(args.gpu_results, [])} if args.gpu_results else {}
    rows = []
    for paper in csv.DictReader(args.paper_csv.open()):
        name = paper['kernel']
        if selected is not None and name not in selected:
            continue
        directory = args.campaign / name
        result = read_json(directory / 'result.json', {})
        variants = [v for v in result.get('variants', []) if v.get('abi_ok')]
        best = min(variants, key=lambda v: (v['counts']['linalg'], v['counts']['loops'],
                                           len(v.get('emitted', [])))) if variants else None
        abi = read_json(directory / ('rank' + str(best['rank'])) / 'abi.json', {}) if best else {}
        calls = abi.get('launches', [])
        run = gpu.get(name, {})
        rows.append(dict(
            kernel=name,
            paper_recorded_calls=paper['raised_call'],
            paper_library_label=paper['raised_library'],
            paper_raised_us=paper['raised_us'],
            paper_native_us=paper['native_us'],
            paper_timing_protocol=paper['timing_protocol'],
            match_status=result.get('status', 'not_run'),
            current_specifications=','.join(best.get('emitted', [])) if best else '',
            current_runtime_calls=','.join(c['call'] for c in calls),
            current_gpu_apis=','.join(c.get('gpu_entry') or '' for c in calls),
            remaining_linalg=best['counts']['linalg'] if best else '',
            remaining_loops=best['counts']['loops'] if best else '',
            paper_input_correctness=run.get('correctness', {}).get('paper', {}).get('passed', 'not_run'),
            raised_matches_native=run.get('raised_matches_native', 'not_run'),
            evidence=str(directory / 'result.json') if result else '',
        ))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix('.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.output.with_suffix('.json').write_text(json.dumps(dict(
        paper_source=str(args.paper_csv),
        paper_sha256=hashlib.sha256(args.paper_csv.read_bytes()).hexdigest(),
        campaign=str(args.campaign),
        scope='Recorded paper wrappers versus current proof/ABI records; no inferred equivalence.',
        rows=rows), indent=2) + '\n')
    print(f'{len(rows)} paper routes compared; {sum(bool(r["current_runtime_calls"]) for r in rows)} have current ABI records')


if __name__ == '__main__':
    main()
