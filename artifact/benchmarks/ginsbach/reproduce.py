"""IDL: sweep 30 original PolyBench inputs, then validate actual emitted calls."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
from common import ROOT, OUT, INPUTS, PYTHON, command, required, save, sha

HERE=Path(__file__).resolve().parent

def build():
    tool=Path(required('ginsbach_root')); pb=OUT/'polybench'
    shutil.copytree(ROOT/'tools/cgeist/Test/polybench',pb)
    # The author's detector depends on the original out-of-line allocator.
    for p in (INPUTS/'polybench_original_utilities').glob('*'):
        shutil.copy2(p,pb/'utilities'/p.name)
    command([PYTHON,HERE/'sweep.py',pb,tool/'build/bin/clang',OUT/'sweep'],OUT/'sweep.log',timeout=7200)
    rows=json.loads((OUT/'sweep/summary.json').read_text())
    selected=[r for r in rows if r['detection_count']]
    command(['g++','-std=c++14','-O3','-fPIC','-shared','-I'+str(tool/'src/include'),
             '-I'+str(tool/'build/include'),tool/'build/lilac/gemm_naive.cc',
             '-o',OUT/'libgemm_naive.so'],OUT/'build.log')
    for row in selected:
        d=OUT/row['kernel'];d.mkdir();src=pb/row['source']
        for variant,cc in [('native','clang-14'),('author_naive',tool/'build/bin/clang')]:
            command([cc,'-O2','-fno-vectorize','-fno-slp-vectorize','-fno-unroll-loops',
                     '-DMINI_DATASET','-DPOLYBENCH_DUMP_ARRAYS','-I'+str(pb/'utilities'),
                     '-I'+str(src.parent),src,pb/'utilities/polybench.c','-L'+str(OUT),
                     '-lgemm_naive','-Wl,-rpath,'+str(OUT),'-lm','-o',d/variant],
                     d/(variant+'.build.log'),cwd=d)
        symbols=subprocess.check_output(['nm','-u',str(d/'author_naive')],text=True)
        if '_harness' not in symbols:raise RuntimeError('Author replacement did not emit a harness call')
        (d/'symbols.txt').write_text(symbols)
        save(d/'build.json',dict(binaries={v:sha(d/v) for v in ['native','author_naive']},
                                 source_sha256=sha(src),author_library_sha256=sha(OUT/'libgemm_naive.so')))
    save(OUT/'selected.json',selected)

def run():
    rows=[]
    for row in json.loads((OUT/'selected.json').read_text()):
        d=OUT/row['kernel'];build=json.loads((d/'build.json').read_text())
        if sha(OUT/'libgemm_naive.so')!=build['author_library_sha256']:raise RuntimeError('Author runtime changed')
        for v in ['native','author_naive']:
            if sha(d/v)!=build['binaries'][v]:raise RuntimeError('Executable changed')
            # Dump only process output, not the command preamble used by command().
            with (d/(v+'.out')).open('w') as stream:
                rc=subprocess.run([str(d/v)],stdout=stream,stderr=stream,timeout=120).returncode
            if rc:raise RuntimeError('Execution failure is not a reproduced correctness failure')
        rc=command([PYTHON,ROOT/'scripts/correctness/compare_polybench_dumps.py',d/'native.out',d/'author_naive.out',
                    '--rtol','5e-4','--atol','1.1e-2'],d/'compare.log',check=False)
        rows.append(dict(kernel=row['kernel'],detected_sites=row['detection_count'],compare_rc=rc))
    save(OUT/'correctness.json',rows)

def summarize():
    sweep=json.loads((OUT/'sweep/summary.json').read_text());rows=json.loads((OUT/'correctness.json').read_text())
    detection={r['kernel']:r['detection_count'] for r in sweep if r['detection_count']}
    reproduced=(len(sweep)==30 and all(r['returncode']==0 for r in sweep) and
                detection==dict(correlation=1,covariance=2,trmm=1) and
                len(rows)==3 and all(r['compare_rc']==1 for r in rows))
    save(OUT/'summary.json',dict(reproduced=reproduced,detections=detection,correctness=rows,
        scope='Four detected idiom instances in three programs; all three execute and fail output comparison. No speedup claimed.'))
    print('IDL reproduced:',reproduced,detection)
    if not reproduced:raise RuntimeError('IDL result differs; see summary.json')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['build','run','summarize','all'],default='all');a=p.parse_args()
    if a.stage in ('build','all'):build()
    if a.stage in ('run','all'):run()
    if a.stage in ('run','summarize','all'):summarize()
