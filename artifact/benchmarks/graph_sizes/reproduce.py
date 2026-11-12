"""Capture both actual graphs for the rebuttal's 370 versus 21 GEMM pair."""
import argparse
import json
from pathlib import Path
from common import OUT, PYTHON, command, save
import speq

HERE=Path(__file__).resolve().parent

def summarize():
    result=json.loads((OUT/'ll/gemm_ref_paired/result.json').read_text())
    if result['error']: raise RuntimeError(result['error'])
    proofs=[p for p in result['proofs'] if p['level']=='linalg' and p.get('ok')
            and p['phase']=='application' and p['spec'] in result['emitted']]
    graphs=[g for p in proofs for g in p['graphs']]
    if not graphs: raise RuntimeError('No accepted scalar graph captured')
    largest=max(graphs,key=lambda g:g['nodes'])
    sp=json.loads((OUT/'speq/results.json').read_text())[0]
    nests=[n for n in sp['nests'] if 'gemm' in n.get('matched', [])]
    if not nests: raise RuntimeError('SpEQ did not match GEMM')
    sn=max(n['egraph_nodes'] for n in nests)
    summary=dict(looplibrarian=largest, speq_nodes=sn, reported=dict(looplibrarian=21,speq=370),
        matches_reported=(largest['nodes']==21 and sn==370),
        convention='LL: largest accepted scalar proof including primitive leaves; SpEQ: LLVM loop graph with primitive constants inline in node labels. These are different scopes and counting conventions.')
    save(OUT/'summary.json',summary);print(json.dumps(summary,indent=2))
    if not summary['matches_reported']:raise RuntimeError('Graph sizes differ; inspect captured graphs')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['build','run','summarize','all'],default='all')
    a=p.parse_args()
    if a.stage in ('build','all'):
        command([PYTHON,HERE/'capture.py','gemm_ref_paired'],OUT/'capture.log',timeout=1800)
        speq.run(OUT/'speq',['/lleq/benchmarks/gemm_ref.c'],graph=True,func='gemm_ref')
    summarize()

if __name__=='__main__':main()
