"""Build/run all 18 supported MFEM extractions and the integrated PCG fixture."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import shlex
from common import ROOT, OUT, INPUTS, command, environment, save, setting, sha
import remote

BASE=INPUTS/'mfem'
CASES=json.loads((BASE/'manifest.json').read_text())
CASES.append(dict(kernel='pcg_application',function='mfem_app_ex9p_mass_convection_2d',
                  source='application_extractions/ex9p_mass_convection_2d.c',macro='MFEM_APP_EX9P'))

def build(row):
    name=row['kernel'];d=OUT/name;d.mkdir(parents=True,exist_ok=True)
    pcg=name=='pcg_application'
    harness=BASE/('application_extractions/mfem_application_reference_harness.c' if pcg else 'benchmarks/mfem_full_output_validation.c')
    env=environment(POLYGEIST_BUILD_WORK_DIR=d/'work',POLYGEIST_ABI_BACKEND='cuda',
                    POLYGEIST_KEEP_DEPENDENCIES=1,POLYGEIST_MATCH_MAX_PARALLEL_DIMS=2,POLYGEIST_MATCH_MAX_OPERAND_RANK=2)
    args=[ROOT/'scripts/correctness/polygeist_build.sh','--target=jetson','--function='+row['function'],
          '--harness='+str(harness),BASE/row['source'],'-D'+row['macro'],
          '-DMFEM_BENCH_NE='+('2' if pcg else '1024'),'-o',d/'program']
    if pcg:args+=['-DBENCH_ITERS=1','-DMFEM_REPLAY_WARMUPS=0','-DMFEM_PUBLICATION_WARMUPS=0',
                  '-DMFEM_PUBLICATION_SAMPLES=1','-DMFEM_DEBUG_OUTPUT_SUMMARY=1']
    command(args,d/'build.log',env)
    proof=json.loads((d/'work/match.json').read_text())
    save(d/'build.json',dict(case=name,source_sha256=sha(BASE/row['source']),harness_sha256=sha(harness),
         executable_sha256=sha(d/'program'),emitted=proof['emitted']))

def run(row):
    name=row['kernel'];d=OUT/name;b=json.loads((d/'build.json').read_text())
    assert sha(d/'program')==b['executable_sha256']
    target='/tmp/looplibrarian-artifact-mfem/'+name
    assert remote.upload(d/'program',target)==0
    cmd='ulimit -s unlimited; chmod +x '+shlex.quote(target)+'; '+remote.runtime_env()+' POLYGEIST_V2_TRACE=1 timeout 300 '+shlex.quote(target)
    rc,log=remote.run(cmd,timeout=320);(d/'run.log').write_text(log)
    valid=rc==0 and ('correctness=PASS' in log or ('ALL CHECKS PASSED' in log))
    calls=re.findall(r'POLYGEIST_V2_CALL backend=cuda call=(\S+)',log)
    valid=valid and bool(calls)
    result=dict(case=name,run_rc=rc,correctness=valid,static_calls=len(b['emitted']),vendor_invocations=len(calls),command=cmd)
    save(d/'result.json',result)
    if not valid:raise RuntimeError(f'MFEM {name} failed; see {d}/run.log')
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['build','run','summarize','all'],default='all')
    p.add_argument('--cases',nargs='+');a=p.parse_args();rows=[r for r in CASES if not a.cases or r['kernel'] in a.cases]
    if not rows or (a.cases and set(a.cases)-{r['kernel'] for r in rows}):p.error('unknown MFEM case')
    if a.stage in ['build','all']:
        with ThreadPoolExecutor(max_workers=int(setting('jobs',2))) as pool:list(pool.map(build,rows))
    if a.stage in ['run','all']:
        for row in rows:run(row)
    if a.stage in ['run','summarize','all']:
        results=[json.loads((OUT/r['kernel']/'result.json').read_text()) for r in rows]
        assert all(r['correctness'] for r in results)
        save(OUT/'summary.json',dict(results=results,passed=len(results),
             contraction_calls=sum(r['static_calls'] for r in results if r['case']!='pcg_application'),
             historical_sites=128,qualification='Current cuBLAS call count differs from historical cuTensorNet site count; no MFEM timing claim.'))
if __name__=='__main__':main()
