#!/usr/bin/env python3
"""Cached conv1d: strict checks and five pairs; uncached correctness only."""
import argparse
import csv
import json
from pathlib import Path
import re
import shlex
import shutil
import sys
from types import SimpleNamespace

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[1]
sys.path.insert(0,str(ROOT/'scripts/correctness'))
import aten_fixture_batch as F
from aten_frozen_runtime_campaign import save,transport,remote_libraries,check_harness_count

parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
output=args.output.resolve();name='aten_conv1d';dest=output/name
adapter='/home/arjaiswal/.config/polygeist-silicon/reproduction_20261109/remote.py'
ld='/home/nvidia/polygeist_aten_native_24_12:/home/nvidia/polygeist_cuda_libs:/home/nvidia/jetson-cuda-libs:/home/nvidia/cuda-12.6/lib64:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu'
built=json.loads((dest/'build.json').read_text())
assert built['rc']==0 and built['configuration']['timing_samples']==5
assert F.sha(dest/'raised-program')==built['binary_sha256']
check_harness_count(dest,5)
assert not (dest/'gpu-result.json').exists()
board=transport(adapter);env='LD_LIBRARY_PATH='+shlex.quote(ld)+' '
remote='/tmp/ll-conv1d-cache-libraries-'+built['binary_sha256'][:12]
assert board.upload(dest/'raised-program',remote)==0
assert board.run('chmod +x '+shlex.quote(remote),timeout=15)[0]==0
before=remote_libraries(board,remote,env,output,'before')

results={}
for enabled in [True,False]:
    target=dest if enabled else output/'cache-disabled-correctness'/name
    if not enabled:
        target.mkdir(parents=True,exist_ok=False)
        for file in ['raised-program','build.json','harness.c']:shutil.copy2(dest/file,target/file)
    proxy=output/('cache-on-transport.py' if enabled else 'cache-off-transport.py')
    proxy.write_text("import importlib.util\ns=importlib.util.spec_from_file_location('base_transport',"+repr(adapter)+")\nb=importlib.util.module_from_spec(s);s.loader.exec_module(b)\nupload=b.upload\ndef run(command,*args,**kwargs):\n diagnostics='1' if 'POLYGEIST_V2_TRACE=1' in command else '0'\n return b.run("+repr('POLYGEIST_CUDNN_PLAN_CACHE='+str(int(enabled))+' POLYGEIST_CUDNN_PLAN_CACHE_DIAGNOSTICS=')+"+diagnostics+' '+command,*args,**kwargs)\n")
    F.run_gpu([name],SimpleNamespace(output=target.parent,transport=str(proxy),remote_library_path=ld,
        cudnn_plan_cache=enabled,run_timeout=180,correctness_kinds=['paper'],skip_timing=not enabled,time_validated_normal_inputs=False))
    result=json.loads((target/'gpu-result.json').read_text())
    events=re.findall(r'^POLYGEIST_CUDNN_PLAN_CACHE event=(\S+).*$',(target/'paper-gpu.log').read_text(),re.M)
    result['cache_events_from_untimed_correctness']=events
    results['enabled' if enabled else 'disabled']=result
    assert enabled or 'timing' not in result
    print('CACHE',enabled,'strict',result['fully_validated'],'events',events,'timing',result.get('timing',{}).get('minimum_first_five_ms'),flush=True)
after=remote_libraries(board,remote,env,output,'after')
save(output/'dynamic-library-stability.json',dict(unchanged=before==after));assert before==after
save(output/'cache-diagnostic.json',dict(variants=results,
    cache_events_scope='Untimed correctness process: host then resident calls. No tracing in timed process.',
    output_comparison_scope='Both modes retain all strict original-C, native-ATen, guard and unchanged-input comparisons. Complete cross-variant bit dumps were not collected.',
    historical_cache_state='Unknown; old 5.805764198 ms evidence retained separately.'))
on=results['enabled'];paper=next(r for r in csv.DictReader((ROOT/'issues/aten_c_kernels/paper_analysis/aten_gpu_nonpointwise_115.csv').open()) if r['kernel']==name)
comparison=dict(kernel=name,fully_validated=on['fully_validated'],paper=paper,actual_samples=5,
    baseline_scope=built['baseline'],binary_sha256=built['binary_sha256'],cudnn_plan_cache=True,
    selected_statistic='minimum_first_five_ms',cache_disabled_correctness_passed=results['disabled']['fully_validated'])
if on['fully_validated'] and on.get('timing',{}).get('passed'):
    t=on['timing']['minimum_first_five_ms'];assert len(on['timing']['samples'])==10
    comparison.update(raised_ms=t['raised'],native_ms=t['native_aten_cuda'],current_native_over_raised=t['native_aten_cuda']/t['raised'],
        raised_over_paper=t['raised']*1000/float(paper['raised_us']),timing_statistic_matches_paper='20' not in paper['timing_protocol'],
        previous_uncached_raised_ms=5.805764198,previous_over_current=5.805764198/t['raised'])
save(dest/'paper-comparison.json',comparison)
print('COMPARISON',json.dumps(comparison),flush=True)
