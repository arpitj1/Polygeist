"""Llama: 19-operation coverage and the complete FP32 layer versus real ggml."""
import argparse
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import sys
from common import ROOT, OUT, INPUTS, command, environment, save, setting, sha
from coverage import campaign
import remote

INPUT=INPUTS/'llama'
BINARIES=['llama_native','llama_resident','llama_ggml']

def build():
    rows=[dict(r, source=str(INPUT/'llama_forward_ops.c'), flags=[])
          for r in json.loads((INPUT/'operations.json').read_text())]
    coverage=campaign(rows,OUT/'coverage')
    save(OUT/'coverage.json',coverage)
    env=environment(POLYGEIST_LLAMA_BENCH=INPUT,
        POLYGEIST_LLAMA_SOURCE=INPUT/'llama2_extended_forward_bench.c')
    command([ROOT/'scripts/correctness/build_llama_reproduction.sh', OUT],OUT/'build.log',env,timeout=10800)
    save(OUT/'build.json',dict(binaries={n:sha(OUT/n) for n in BINARIES},
         inputs={p.name:sha(p) for p in INPUT.glob('*.c')},
         expert_source_sha256=sha(ROOT/'scripts/correctness/llama_extended_ggml_bench.cpp')))

def run():
    build=json.loads((OUT/'build.json').read_text())
    target='/tmp/looplibrarian-artifact-llama-'+sha(OUT/'build.json')[:12]
    ggml=Path(os.environ['POLYGEIST_GGML_BUILD'])
    deps={}
    for lib in (ggml/'ggml/src').rglob('libggml*.so.0'):
        if remote.upload(lib.resolve(),target+'/'+lib.name):raise RuntimeError('ggml library deployment failed')
        deps[lib.name]=sha(lib)
    if len(deps)<4:raise RuntimeError('Expected ggml, ggml-base, ggml-cpu and ggml-cuda libraries')
    save(OUT/'ggml-libraries.json',deps)
    probe=(ROOT/'scripts/correctness/fig3_board_preflight.py').read_text()
    rows=[]
    for name in BINARIES:
        rc,log=remote.run("python3 - <<'PY'\n"+probe+"\nPY",40)
        (OUT/(name+'-preflight.json')).write_text(log)
        if rc:raise RuntimeError('Board preflight failed')
        if sha(OUT/name)!=build['binaries'][name]:raise RuntimeError('Executable changed since build')
        exe=target+'/'+name
        if remote.upload(OUT/name,exe):raise RuntimeError('Llama executable deployment failed')
        libs=target+':'+setting('remote_library_path','/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu')
        cmd=('chmod +x '+shlex.quote(exe)+'; OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '
             'POLYGEIST_CUDA_GRAPH=1 LD_LIBRARY_PATH='+shlex.quote(libs)+' timeout 900 taskset -c '+
             str(int(setting('cpu',0)))+' '+shlex.quote(exe))
        if name=='llama_ggml':cmd+=' --warmup 5 --iters 5 --dump-all'
        rc,log=remote.run(cmd,940);(OUT/(name+'.log')).write_text(log)
        samples=[float(line.split(',')[-1]) for line in log.splitlines()
                 if line.startswith(('RESIDENT_SAMPLE,','TIMING_SAMPLE,'))]
        row=dict(name=name,returncode=rc,command=cmd,sha256=sha(OUT/name),samples_ms=samples)
        rows.append(row);save(OUT/'runs.json',rows)
        if rc or len(samples)!=5 or not all(math.isfinite(v) and v>0 for v in samples):
            raise RuntimeError(f'{name}: run or timing sample validation failed')

def summarize():
    sys.path.insert(0,str(ROOT/'scripts/correctness'))
    from compare_llama_outputs import read_logits,compare
    outputs={n:read_logits(OUT/(n+'.log'),32000) for n in BINARIES}
    checks={a+'_vs_'+b:compare(outputs[a],outputs[b]) for a,b in
            [('llama_resident','llama_ggml'),('llama_resident','llama_native'),('llama_ggml','llama_native')]}
    covered=json.loads((OUT/'coverage.json').read_text())
    rows=json.loads((OUT/'runs.json').read_text())
    for r in rows:
        if r['returncode'] or len(r['samples_ms'])!=5:raise RuntimeError('Invalid saved run')
        r.update(minimum_ms=min(r['samples_ms']), median_ms=statistics.median(r['samples_ms']))
    recognized=sum(r['recognized'] for r in covered)
    passed=checks['llama_resident_vs_llama_ggml']['strict_failures']==0
    result=dict(operations=len(covered),recognized=recognized,abi_lowered=sum(r['abi_ok'] for r in covered),
                coverage_matches_paper=recognized==14 and len(covered)==19, expert_comparison_passed=passed,
                comparisons=checks,timings=rows,paper_looplibrarian_ms=12.826,
                protocol='FP32, one token/layer; five warmups and five samples, minimum device interval for LL; native CPU strict comparison disclosed separately.')
    save(OUT/'summary.json',result)
    print('Llama',recognized,'/',len(covered),'expert comparison',passed)
    if not passed or not result['coverage_matches_paper']:raise RuntimeError('Llama result differs; see summary.json')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['build','run','summarize','all'],default='all');a=p.parse_args()
    if a.stage in ('build','all'):build()
    if a.stage in ('run','all'):run()
    if a.stage in ('run','summarize','all'):summarize()
