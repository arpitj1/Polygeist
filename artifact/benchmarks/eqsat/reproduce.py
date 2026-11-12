"""Rebuttal coverage: all 30 inputs, both EqSat arms, and real SpEQ."""
import argparse
import csv
import json
from pathlib import Path
from common import ROOT, OUT, INPUTS, save
from coverage import campaign
import speq

PUBLISHED = set('2mm 3mm atax bicg cholesky correlation covariance doitgen gemm gemver gesummv mvt symm syr2k syrk trisolv trmm'.split())

def inputs():
    rows = []
    for row in csv.DictReader((INPUTS/'polybench_manifest.csv').open()):
        src = ROOT/row['source']
        rows.append(dict(name=row['kernel'], source=str(src),
            function='kernel_'+row['kernel'].replace('-', '_'),
            flags=['-I'+str(ROOT/'tools/cgeist/Test/polybench/utilities'),
                   '-DMINI_DATASET', '-DDATA_TYPE_IS_DOUBLE']))
    return rows

def summarize(rows):
    results = [json.loads((OUT/'ll'/r['name']/arm/'result.json').read_text())
               for r in rows for arm in ['full', 'none']]
    totals = {}
    for arm in ['full', 'none']:
        group = [r for r in results if r['ablation']==arm]
        totals[arm] = dict(recognized=sum(r['recognized'] for r in group),
            abi_lowered=sum(r['abi_ok'] for r in group),
            published_cohort_recognized=sum(r['recognized'] and r['name'] in PUBLISHED for r in group),
            additional_recognitions=[r['name'] for r in group if r['recognized'] and r['name'] not in PUBLISHED],
            errors=[dict(name=r['name'], error=r['error']) for r in group if r['error']])
    native = json.loads((OUT/'speq-native/results.json').read_text())
    linear = json.loads((OUT/'speq-linearized/results.json').read_text())
    totals['speq'] = dict(unmodified_cases=len(native), matched_unmodified=sum(r.get('n_match',0)>0 for r in native),
                         linearized_gemm_matches=sum(r.get('n_match',0)>0 for r in linear))
    full = len(rows)==30
    reproduced = (totals['full']['published_cohort_recognized']==17 and
                  totals['none']['recognized']==8 and totals['speq']['linearized_gemm_matches']==1)
    save(OUT/'summary.json', dict(totals=totals, results=results, full_cohort=full,
         submitted_cohort_reproduced=full and reproduced,
         scope='Recognition counts; ABI lowering is reported separately. SpEQ one is a linearized GEMM, not unmodified PolyBench.'))
    print(json.dumps(totals, indent=2))
    if full and not reproduced: raise RuntimeError('Submitted cohort differs; see summary.json')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['build','run','summarize','all'], default='all')
    p.add_argument('--cases', nargs='+'); a=p.parse_args()
    rows=[r for r in inputs() if not a.cases or r['name'] in a.cases]
    if not rows or (a.cases and set(a.cases)-{r['name'] for r in rows}): p.error('unknown PolyBench case')
    if a.stage in ('build','all'):
        save(OUT/'cases.json', rows)
        campaign(rows, OUT/'ll', ablations=('full','none'))
        base=ROOT/'tools/cgeist/Test/polybench'
        speq.run(OUT/'speq-native', ['/pb/'+str(Path(r['source']).relative_to(base)) for r in rows],
                 func='AUTO', cflags='-I/pb/utilities -DLARGE_DATASET -DDATA_TYPE_IS_DOUBLE')
        speq.run(OUT/'speq-linearized', ['/lleq/benchmarks/gemm_ref.c'], func='gemm_ref')
    if a.stage in ('run','summarize'):
        rows=json.loads((OUT/'cases.json').read_text())
    summarize(rows)

if __name__=='__main__': main()
