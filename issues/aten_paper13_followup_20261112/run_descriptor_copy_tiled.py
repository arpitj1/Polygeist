#!/usr/bin/env python3
"""Validate tiled descriptor permutations and isolate their effect on pixel shuffle."""
import csv,json,re,shlex,shutil,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[1]
sys.path.insert(0,str(ROOT/'scripts/correctness'))
import aten_fixture_batch as F
from aten_frozen_runtime_campaign import prepare_one,save,transport,remote_libraries,check_harness_count
out=BASE/'descriptor-copy-tiled';adapter='/home/arjaiswal/.config/polygeist-silicon/reproduction_20261109/remote.py';proxy=str(BASE/'workspace-cache/cache-on-transport.py')
ld='/home/nvidia/polygeist_aten_native_24_12:/home/nvidia/polygeist_cuda_libs:/home/nvidia/jetson-cuda-libs:/home/nvidia/cuda-12.6/lib64:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu'
board=transport(adapter);env='LD_LIBRARY_PATH='+shlex.quote(ld)+' '
unit=out/'test-memref-copy-bits';assert not (out/'gpu-test.json').exists()
cmd=['aarch64-linux-gnu-g++',str(out/'test.o'),str(out/'thrust_runtime.o'),'-L/usr/local/cuda-12.6/targets/sbsa-linux/lib','-lcudart','-Wl,--gc-sections','-o',str(unit)]
with (out/'test-link.log').open('w') as log:
 log.write(shlex.join(cmd)+'\n');log.flush();subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT)
remote='/tmp/ll-memref-tiled-'+F.sha(unit)[:12];assert board.upload(unit,remote)==0
assert board.run('chmod +x '+shlex.quote(remote),timeout=15)[0]==0
rc,log=board.run(env+'timeout 180 '+shlex.quote(remote),timeout=185);(out/'gpu-test.log').write_text(log)
save(out/'gpu-test.json',dict(rc=rc,passed=rc==0 and 'PASS' in log,binary_sha256=F.sha(unit),thrust_object_sha256=F.sha(out/'thrust_runtime.o')))
print('UNIT',rc,log.strip(),flush=True);assert rc==0 and 'PASS' in log
paper={r['kernel']:r for r in csv.DictReader((ROOT/'issues/aten_c_kernels/paper_analysis/aten_gpu_nonpointwise_115.csv').open())}
for name in ['aten_pixel_shuffle_cpu_backend']:
 source=BASE/'descriptor-copy-unrolled'/name
 old=json.loads((source/'build.json').read_text());prior=json.loads((source/'gpu-result.json').read_text());assert prior['fully_validated']
 prepared=prepare_one(source,out,5);dest=out/name;check_harness_count(dest,5)
 shutil.copy2(dest/'work/thrust_runtime.o',dest/'work/original-thrust_runtime.o');shutil.copy2(out/'thrust_runtime.o',dest/'work/thrust_runtime.o')
 replacements={x:str(dest/('work' if Path(x).parent.name=='work' else 'objects')/Path(x).name) for x in old['commands'][-1] if x.endswith('.o')}
 link=[replacements.get(x,x) for x in old['commands'][-1]];link[link.index('-o')+1]=str(dest/'raised-program')
 with (dest/'relink.log').open('w') as log:
  log.write(shlex.join(link)+'\n');log.flush();subprocess.run(link,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=120)
 old.update(commands=[link],binary_sha256=F.sha(dest/'raised-program'),prior_binary_sha256=old['binary_sha256'],
   runtime_replacement=dict(thrust_object_sha256=F.sha(out/'thrust_runtime.o'),prior_thrust_object_sha256=F.sha(dest/'work/original-thrust_runtime.o')),
   isolation='Only the descriptor-copy Thrust object changes; all other objects and the five-pair harness are frozen. Both current and prior runs enable the same bounded workspace cache.')
 save(dest/'build.json',old)
 remote='/tmp/ll-copy-tiled-libraries-'+old['binary_sha256'][:12];assert board.upload(dest/'raised-program',remote)==0
 assert board.run('chmod +x '+shlex.quote(remote),timeout=15)[0]==0
 before=remote_libraries(board,remote,env,dest,'before')
 F.run_gpu([name],SimpleNamespace(output=out,transport=proxy,remote_library_path=ld,cudnn_plan_cache=False,
   run_timeout=180,correctness_kinds=['paper'],skip_timing=False,time_validated_normal_inputs=False))
 after=remote_libraries(board,remote,env,dest,'after');save(dest/'dynamic-library-stability.json',dict(unchanged=before==after));assert before==after
 r=json.loads((dest/'gpu-result.json').read_text());assert r['fully_validated']
 t=r['timing']['minimum_first_five_ms'];assert len(r['timing']['samples'])==10
 comparison=dict(kernel=name,fully_validated=True,paper=paper[name],actual_samples=5,baseline_scope=old['baseline'],binary_sha256=r['binary_sha256'],
   selected_statistic='minimum_first_five_ms',workspace_cache=True,prior_evidence=str(source/'gpu-result.json'),prior_timing=prior['timing'],
   raised_ms=t['raised'],native_ms=t['native_aten_cuda'],current_native_over_raised=t['native_aten_cuda']/t['raised'],
   raised_over_paper=t['raised']*1000/float(paper[name]['raised_us']),previous_first_five_over_current=prior['timing']['minimum_first_five_ms']['raised']/t['raised'],
   timing_statistic_matches_paper='20' not in paper[name]['timing_protocol'])
 save(dest/'paper-comparison.json',comparison)
 print('CASE',name,'strict passed','raised_ms',t['raised'],'native_ms',t['native_aten_cuda'],'paper_ratio',comparison['raised_over_paper'],'old_over_new',comparison['previous_first_five_over_current'],flush=True)
