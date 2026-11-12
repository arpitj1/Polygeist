"""Table 9: fresh native, real KernelFaRer and whole-program LL builds."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from common import ROOT, OUT, ARTIFACT, PYTHON, command, environment, polybench, required, save, setting, sha

NAMES = ['gemm', 'syr2k', '2mm', '3mm']
PAPER_LL = [8.6, 366, 115, 93]
PAPER_KF = [8.6, 346, 113, 112]
UTIL = ROOT / 'tools/cgeist/Test/polybench/utilities'

def plugin():
    source = Path(required('kernelfarer_root')).resolve()
    patch = Path(__file__).with_name('kernelfarer.patch')
    # The setup command applies this disclosed patch to a pinned fresh clone.
    command(['git', '-C', source, 'apply', '--reverse', '--check', patch], OUT/'plugin.log')
    dest = OUT/'plugin-build'
    command(['cmake', '-S', source, '-B', dest, '-G', 'Ninja',
             '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_C_COMPILER=clang-14',
             '-DCMAKE_CXX_COMPILER=clang++-14', '-DLLVM_DIR=/usr/lib/llvm-14/lib/cmake/llvm'], OUT/'plugin.log')
    command(['cmake', '--build', dest, '-j', str(setting('jobs', 3))], OUT/'plugin.log')
    so = dest/'passes/KernelFaRer.so'
    save(OUT/'plugin.json', dict(source=str(source), patch_sha256=sha(patch),
         revision=subprocess.check_output(['git', '-C', source, 'rev-parse', 'HEAD'], text=True).strip(),
         sha256=sha(so)))
    return so

def build_one(name, so):
    d=OUT/name; d.mkdir(parents=True,exist_ok=True); original=polybench()[name]
    src=d/original.name
    src.write_text(original.read_text().replace('DATA_PRINTF_MODIFIER', '"%0.17g "'))
    flags=['-O3','-DLARGE_DATASET','-DDATA_TYPE_IS_FLOAT','-DPOLYBENCH_TIME',
           '-I'+str(UTIL),'-I'+str(original.parent)]
    baseline=flags+['-ffast-math','-ffp-contract=off','-march=native','-mtune=native','-DPOLYBENCH_USE_C99_PROTO']
    driver=OUT/'repeat.c'; clang=ROOT/'llvm-project/build/bin/clang'; work=d/'work'
    env=environment(POLYGEIST_CPU_BLAS=1,POLYGEIST_CPU_BLAS_LIBS='-lopenblas',
                    POLYGEIST_CGEIST_FLAGS='--no-inline --memref-fullrank',
                    POLYGEIST_LOWER_SUBMAP_BEFORE_DEBUFFERIZE=1,POLYGEIST_KEEP_WORK=1,
                    POLYGEIST_BUILD_WORK_DIR=work)
    for key in ['POLYGEIST_POLYBENCH_REPETITIONS','POLYGEIST_CUTENSOR_ROOT','POLYGEIST_CUTENSORNET_ROOT']:
        env.pop(key,None)
    command([ROOT/'scripts/correctness/polygeist_build.sh','--target=host','--whole-program',
             '--function=kernel_'+name,'-o',d/'once',src,*flags],d/'build.log',env)
    shutil.copy2(work/'kernel.o',d/'ll.o')
    command(['objcopy','--redefine-sym','main=benchmark_once',d/'ll.o'],d/'build.log')
    command([clang,driver,d/'ll.o',*[work/(s+'.o') for s in
             ['wrapper','rt','spec_runtime','mlir_runner_utils','polybench']],
             '-lopenblas','-lm','-lpthread','-Wl,--gc-sections','-o',d/'looplibrarian'],d/'build.log')
    for variant,extra in [('native',[]),('kernelfarer',[f'-fplugin={so}',f'-fpass-plugin={so}',
                          '-mllvm','--kernelfarer-replacement-mode=cblas-interface'])]:
        obj=d/(variant+'.o')
        command(['clang-14',*baseline,*extra,'-c',src,'-o',obj],d/'build.log')
        command(['objcopy','--redefine-sym','main=benchmark_once',obj],d/'build.log')
        command(['clang-14',*baseline,driver,obj,UTIL/'polybench.c','-lopenblas','-lm','-o',d/variant],d/'build.log')
    proof=json.loads((work/'match.json').read_text())
    result=dict(kernel=name,source_sha256=sha(original),printing_source_sha256=sha(src),
                emitted=proof['emitted'],binaries={v:sha(d/v) for v in ['native','kernelfarer','looplibrarian']})
    save(d/'build.json',result); return result

def build(names):
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'repeat.c').write_text('extern int benchmark_once(int,char**); int main(void){char *a[]={"",0};for(int i=0;i<10;i++){int r=benchmark_once(i==9?43:0,a);if(r)return r;}return 0;}\n')
    so=plugin()
    with ThreadPoolExecutor(max_workers=int(setting('jobs',3))) as pool:
        rows=list(pool.map(lambda n:build_one(n,so),names))
    save(OUT/'builds.json',rows)

def run(names):
    rows=[]; cpu=str(setting('host_cpu',0)); env=environment()
    (OUT/'load-before.txt').write_text(subprocess.check_output(['uptime'],text=True))
    for name in names:
        d=OUT/name; build=json.loads((d/'build.json').read_text())
        for variant in ['native','kernelfarer','looplibrarian']:
            assert sha(d/variant)==build['binaries'][variant], 'Executable changed'
            with (d/(variant+'.samples')).open('w') as log,(d/(variant+'.out')).open('w') as output:
                rc=subprocess.run(['taskset','-c',cpu,str(d/variant)],env=env,stdout=log,stderr=output,timeout=1800).returncode
            samples=[float(v)*1000 for v in (d/(variant+'.samples')).read_text().splitlines() if re.fullmatch(r'\d+\.\d+',v)]
            cc=command([PYTHON,ROOT/'scripts/correctness/compare_polybench_dumps.py',d/'native.out',d/(variant+'.out'),
                        '--rtol','5e-4','--atol','1.1e-2'],d/(variant+'.compare.log'),check=False)
            valid=rc==cc==0 and len(samples)==10 and all(v>0 for v in samples)
            rows.append(dict(kernel=name,variant=variant,valid=valid,run_rc=rc,compare_rc=cc,
                             warmups_ms=samples[:5],samples_ms=samples[5:],minimum_ms=min(samples[5:]) if valid else None))
            save(OUT/'timing.json',rows)
            if not valid:raise RuntimeError(f'{name}/{variant} failed; timings excluded')
    (OUT/'load-after.txt').write_text(subprocess.check_output(['uptime'],text=True))
    return rows

def summarize():
    rows=json.loads((OUT/'timing.json').read_text()); out=[]
    assert all(r['valid'] for r in rows)
    for name in sorted({r['kernel'] for r in rows},key=NAMES.index):
        by={r['variant']:r['minimum_ms'] for r in rows if r['kernel']==name}
        assert set(by)=={'native','kernelfarer','looplibrarian'}
        i=NAMES.index(name)
        out.append(dict(kernel=name,**{k+'_ms':v for k,v in by.items()},
                        ll_speedup=by['native']/by['looplibrarian'],kf_speedup=by['native']/by['kernelfarer'],
                        paper_ll=PAPER_LL[i],paper_kf=PAPER_KF[i]))
    save(OUT/'summary.json',out)
    with (OUT/'summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
    print(json.dumps(out,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['build','run','summarize','all'],default='all')
    p.add_argument('--cases',nargs='+',choices=NAMES,default=NAMES);a=p.parse_args()
    if a.stage in ['build','all']:build(a.cases)
    if a.stage in ['run','all']:run(a.cases)
    if a.stage in ['run','summarize','all']:summarize()
