#!/usr/bin/env python3
"""Isolate bounded device-workspace reuse against frozen descriptor-copy builds."""
import csv,json,re,shlex,shutil,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[1]
sys.path.insert(0,str(ROOT/'scripts/correctness'))
import aten_fixture_batch as F
from aten_frozen_runtime_campaign import prepare_one,save,transport,remote_libraries,check_harness_count
out=BASE/'workspace-cache';adapter='/home/arjaiswal/.config/polygeist-silicon/reproduction_20261109/remote.py'
ld='/home/nvidia/polygeist_aten_native_24_12:/home/nvidia/polygeist_cuda_libs:/home/nvidia/jetson-cuda-libs:/home/nvidia/cuda-12.6/lib64:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu'
board=transport(adapter);env='LD_LIBRARY_PATH='+shlex.quote(ld)+' '
unit=out/'test-workspace-cache';remote='/tmp/ll-workspace-cache-'+F.sha(unit)[:12]
assert board.upload(unit,remote)==0;assert board.run('chmod +x '+shlex.quote(remote),timeout=15)[0]==0
for enabled in [1,0]:
 path=out/('unit-enabled' if enabled else 'unit-disabled');assert not path.with_suffix('.json').exists()
 rc,log=board.run(env+'POLYGEIST_GPU_WORKSPACE_CACHE='+str(enabled)+' POLYGEIST_GPU_WORKSPACE_CACHE_BYTES=16384 POLYGEIST_GPU_WORKSPACE_DIAGNOSTICS=1 timeout 120 '+shlex.quote(remote),timeout=125)
 path.with_suffix('.log').write_text(log)
 records=re.findall(r'event=(\w+) bytes=(\d+) retained_bytes=(\d+) context_id=(\d+)',log)
 result=dict(rc=rc,passed=rc==0 and 'PASS' in log,binary_sha256=F.sha(unit),events=records)
 save(path.with_suffix('.json'),result);print('UNIT',enabled,rc,log.splitlines()[-1] if log else '',flush=True)
 assert result['passed']
 if enabled:
  assert {'hit','miss','uncached','release'} <= {r[0] for r in records}
  assert max(int(r[2]) for r in records)<=16384
  assert len({r[3] for r in records if r[3]!='0'})>=2 or any(r[0]=='context_unavailable' for r in records), 'Reset must use a new context ID or the uncached path'
paper={r['kernel']:r for r in csv.DictReader((ROOT/'issues/aten_c_kernels/paper_analysis/aten_gpu_nonpointwise_115.csv').open())}
proxies={}
for enabled in [1,0]:
 proxy=out/('cache-on-transport.py' if enabled else 'cache-off-transport.py');proxies[enabled]=str(proxy)
 proxy.write_text("import importlib.util\ns=importlib.util.spec_from_file_location('base_transport',"+repr(adapter)+")\nb=importlib.util.module_from_spec(s);s.loader.exec_module(b)\nupload=b.upload\ndef run(command,*args,**kwargs):\n diagnostics='1' if 'POLYGEIST_V2_TRACE=1' in command else '0'\n return b.run("+repr('POLYGEIST_GPU_WORKSPACE_CACHE='+str(enabled)+' POLYGEIST_GPU_WORKSPACE_DIAGNOSTICS=')+"+diagnostics+' '+command,*args,**kwargs)\n")
for name in ['aten_adaptive_avg_pool2d','aten_pixel_shuffle_cpu_backend','aten_batch_norm']:
 source=BASE/'descriptor-copy-optimized'/name
 old=json.loads((source/'build.json').read_text());prior=json.loads((source/'gpu-result.json').read_text());assert prior['fully_validated']
 prepared=prepare_one(source,out,5);dest=out/name;check_harness_count(dest,5)
 shutil.copy2(dest/'work/rt.o',dest/'work/original-rt.o');shutil.copy2(out/'rt.o',dest/'work/rt.o')
 replacements={x:str(dest/('work' if Path(x).parent.name=='work' else 'objects')/Path(x).name) for x in old['commands'][-1] if x.endswith('.o')}
 link=[replacements.get(x,x) for x in old['commands'][-1]];link[link.index('-o')+1]=str(dest/'raised-program')
 with (dest/'relink.log').open('w') as log:
  log.write(shlex.join(link)+'\n');log.flush();subprocess.run(link,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=120)
 old.update(commands=[link],binary_sha256=F.sha(dest/'raised-program'),prior_binary_sha256=old['binary_sha256'],
   runtime_replacement=dict(runtime_object_sha256=F.sha(out/'rt.o'),prior_runtime_object_sha256=F.sha(dest/'work/original-rt.o')),
   isolation='Only the main CUDA runtime object changes; compiler, ABI, optimized descriptor-copy object, C/native references and five-pair harness are frozen. Cache-on/off use the identical binary.')
 save(dest/'build.json',old)
 remote='/tmp/ll-workspace-libraries-'+old['binary_sha256'][:12];assert board.upload(dest/'raised-program',remote)==0
 assert board.run('chmod +x '+shlex.quote(remote),timeout=15)[0]==0
 before=remote_libraries(board,remote,env,dest,'before')
 records={}
 for enabled in [1,0]:
  target=dest if enabled else out/'cache-disabled-correctness'/name
  if not enabled:
   target.mkdir(parents=True,exist_ok=False)
   for filename in ['raised-program','build.json','harness.c']:shutil.copy2(dest/filename,target/filename)
  F.run_gpu([name],SimpleNamespace(output=target.parent,transport=proxies[enabled],remote_library_path=ld,cudnn_plan_cache=False,
   run_timeout=180,correctness_kinds=['paper'],skip_timing=not enabled,time_validated_normal_inputs=False))
  result=json.loads((target/'gpu-result.json').read_text());records[str(enabled)]=result
  assert result['fully_validated'],'Strict failure: stop before further cases'
  if not enabled:assert 'timing' not in result
 after=remote_libraries(board,remote,env,dest,'after');save(dest/'dynamic-library-stability.json',dict(unchanged=before==after));assert before==after
 r=records['1'];t=r['timing']['minimum_first_five_ms'];assert len(r['timing']['samples'])==10
 comparison=dict(kernel=name,fully_validated=True,paper=paper[name],actual_samples=5,baseline_scope=old['baseline'],binary_sha256=r['binary_sha256'],
   selected_statistic='minimum_first_five_ms',workspace_cache=True,cache_disabled_correctness_passed=True,prior_evidence=str(source/'gpu-result.json'),
   prior_timing=prior['timing'],raised_ms=t['raised'],native_ms=t['native_aten_cuda'],current_native_over_raised=t['native_aten_cuda']/t['raised'],
   raised_over_paper=t['raised']*1000/float(paper[name]['raised_us']),previous_first_five_over_current=prior['timing']['minimum_first_five_ms']['raised']/t['raised'],
   timing_statistic_matches_paper='20' not in paper[name]['timing_protocol'])
 save(dest/'paper-comparison.json',comparison)
 print('CASE',name,'strict on/off passed','raised_ms',t['raised'],'native_ms',t['native_aten_cuda'],'paper_ratio',comparison['raised_over_paper'],'old_over_new',comparison['previous_first_five_over_current'],flush=True)
