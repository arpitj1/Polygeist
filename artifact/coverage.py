"""Fresh C -> production Linalg -> imported V3 match -> checked ABI evidence.

Each worker is a separate process so ablation settings and egglog state cannot
leak between cases. No benchmark-specific matcher is installed here.
"""
import dataclasses
import json
import os
from pathlib import Path
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from common import ARTIFACT, ROOT, PYTHON, command, environment, save, setting, sha

def worker(request):
    row = json.loads(Path(request).read_text())
    dest = Path(row['output']); dest.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = [str(ROOT/'scripts/correctness'), str(ROOT/'scripts/correctness/lib_v2')]
    os.environ.update(environment(EQSAT_ABLATION=row.get('ablation', 'full'),
        REGION_MATCH_NO_CACHE=1, REGION_MATCH_CACHE_DIR=dest/'cache'))
    import recipe
    import spec_match as S
    import lower_v2
    result = dict(name=row['name'], ablation=row.get('ablation', 'full'),
                  source_sha256=sha(row['source']), raised=False, recognized=False,
                  abi_ok=False, emitted=[], error=None)
    try:
        raised = recipe.raise_c(recipe.PRODUCTION, row['source'], row['function'],
                               str(dest/'source'), row.get('flags', []), 300)
        save(dest/'raise.json', {k: dataclasses.asdict(v) if dataclasses.is_dataclass(v) else v
                                for k, v in raised.items()})
        if not all(raised.get(k) and raised[k].ok for k in ('cgeist', 'raise', 'debuf')):
            raise RuntimeError('Raising failed; inspect raise.json')
        result['raised'] = True
        text = Path(raised['paths']['debuf']).read_text()
        result['raised_sha256'] = sha(raised['paths']['debuf'])
        budget = int(setting('region_budget', 300))
        result['region_budget_seconds'] = budget
        matched, proof = S.match(text, backend='cuda', region_budget=budget)
        (dest/'matched.mlir').write_text(matched); save(dest/'match.json', proof)
        S.verify(matched, proof, backend='cuda')
        result.update(emitted=proof['emitted'], recognized=bool(proof['emitted']))
        if result['recognized']:
            report = {}
            lowered = lower_v2.lower(matched, report=report, backend='cuda')
            (dest/'abi.mlir').write_text(lowered); save(dest/'abi.json', report)
            result['abi_ok'] = True
    except Exception:
        result['error'] = traceback.format_exc()
    save(dest/'result.json', result)
    # Unsupported applications are evaluation outcomes, never silently successes.
    return result

def campaign(rows, output, ablations=('full',)):
    requests = []
    for row in rows:
        for ablation in ablations:
            dest = output/row['name']/ablation; dest.mkdir(parents=True, exist_ok=True)
            req = dict(row, output=str(dest), ablation=ablation)
            save(dest/'request.json', req); requests.append(dest)
    def one(dest):
        command([PYTHON, ARTIFACT/'coverage.py', dest/'request.json'], dest/'worker.log',
                timeout=int(setting('match_timeout', 1800)))
        return json.loads((dest/'result.json').read_text())
    with ThreadPoolExecutor(max_workers=int(setting('jobs', 3))) as pool:
        return list(pool.map(one, requests))

if __name__ == '__main__':
    worker(sys.argv[1])
