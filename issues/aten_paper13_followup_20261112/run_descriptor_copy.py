#!/usr/bin/env python3
"""Validate descriptor-copy semantics and isolate its effect on three paper cases."""
import csv,json,shlex,shutil,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[1]
sys.path.insert(0,str(ROOT/'scripts/correctness'))
import aten_fixture_batch as F
from aten_frozen_runtime_campaign import prepare_one,save,transport,remote_libraries,check_harness_count
out=BASE/'descriptor-copy-optimized';adapter='/home/arjaiswal/.config/polygeist-silicon/reproduction_20261109/remote.py'
ld='/home/nvidia/polygeist_aten_native_24_12:/home/nvidia/polygeist_cuda_libs:/home/nvidia/jetson-cuda-libs:/home/nvidia/cuda-12.6/lib64:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu'
board=transport(adapter);env='LD_LIBRARY_PATH='+shlex.quote(ld)+' '
unit=out/'test-memref-copy-bits';assert not (out/'gpu-test.json').exists()
command=['aarch64-linux-gnu-g++',str(out/'test.o'),str(out/'thrust_runtime.o'),'-L/usr/local/cuda-12.6/targets/sbsa-linux/lib','-lcudart','-Wl,--gc-sections','-o',str(unit)]
with (out/'test-link.log').open('w') as log:
 log.write(shlex.join(command)+'\n');log.flush();subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
remote='/tmp/ll-memref-copy-'+F.sha(unit)[:12];assert board.upload(unit,remote)==0
rc,log=board.run('chmod +x '+shlex.quote(remote)+' && sha256sum '+shlex.quote(remote),timeout=15)
(out/'unit-remote-hash.log').write_text(log);assert rc==0 and F.sha(unit) in log
rc,log=board.run(env+'timeout 180 '+shlex.quote(remote),timeout=185)
(out/'gpu-test.log').write_text(log);record=dict(rc=rc,passed=rc==0 and 'PASS' in log,binary_sha256=F.sha(unit),thrust_object_sha256=F.sha(out/'thrust_runtime.o'))
save(out/'gpu-test.json',record);print('UNIT',rc,log.strip(),flush=True);assert record['passed']
paper={r['kernel']:r for r in csv.DictReader((ROOT/'issues/aten_c_kernels/paper_analysis/aten_gpu_nonpointwise_115.csv').open())}
for name in ['aten_adaptive_avg_pool2d','aten_pixel_shuffle_cpu_backend','aten_batch_norm']:
 source=ROOT/'issues/aten_paper86_correctness_20261112/runtime-current'/name
 old=json.loads((source/'build.json').read_text());oldresult=json.loads((source/'gpu-result.json').read_text());assert oldresult['fully_validated']
 prepared=prepare_one(source,out,5);dest=out/name;check_harness_count(dest,5)
 shutil.copy2(dest/'work/thrust_runtime.o',dest/'work/original-thrust_runtime.o')
 shutil.copy2(out/'thrust_runtime.o',dest/'work/thrust_runtime.o')
 link=prepared['commands'][-1]
 assert link[link.index('-o')+1]==str(dest/'timing-program')
 link[link.index('-o')+1]=str(dest/'raised-program')
 with (dest/'relink.log').open('w') as log:
  log.write(shlex.join(link)+'\n');log.flush();subprocess.run(link,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=120)
 old['configuration']['timing_samples']=5
 old.update(commands=[link],binary_sha256=F.sha(dest/'raised-program'),prior_binary_sha256=old['binary_sha256'],
   runtime_replacement=dict(thrust_object_sha256=F.sha(out/'thrust_runtime.o'),
     prior_thrust_object_sha256=F.sha(dest/'work/original-thrust_runtime.o')),
   isolation='Only Thrust descriptor copy object changes; all other compiled objects frozen. Harness timing sample count changes from historical twenty to requested five.')
 save(dest/'build.json',old)
 remote='/tmp/ll-copy-libraries-'+old['binary_sha256'][:12];assert board.upload(dest/'raised-program',remote)==0
 assert board.run('chmod +x '+shlex.quote(remote),timeout=15)[0]==0
 before=remote_libraries(board,remote,env,dest,'before')
 F.run_gpu([name],SimpleNamespace(output=out,transport=adapter,remote_library_path=ld,cudnn_plan_cache=False,
   run_timeout=180,correctness_kinds=['paper'],skip_timing=False,time_validated_normal_inputs=False))
 after=remote_libraries(board,remote,env,dest,'after');save(dest/'dynamic-library-stability.json',dict(unchanged=before==after));assert before==after
 r=json.loads((dest/'gpu-result.json').read_text());comparison=dict(kernel=name,fully_validated=r['fully_validated'],paper=paper[name],actual_samples=5,
    baseline_scope=old['baseline'],binary_sha256=r['binary_sha256'],selected_statistic='minimum_first_five_ms',
    prior_evidence=str(source/'runtime-result.json'),prior_timing=oldresult.get('timing'))
 if r['fully_validated'] and r.get('timing',{}).get('passed'):
  t=r['timing']['minimum_first_five_ms'];assert len(r['timing']['samples'])==10
  previous=oldresult['timing']['minimum_first_five_ms']['raised']
  comparison.update(raised_ms=t['raised'],native_ms=t['native_aten_cuda'],current_native_over_raised=t['native_aten_cuda']/t['raised'],
    raised_over_paper=t['raised']*1000/float(paper[name]['raised_us']),previous_first_five_over_current=previous/t['raised'],
    timing_statistic_matches_paper='20' not in paper[name]['timing_protocol'])
 save(dest/'paper-comparison.json',comparison)
 print('CASE',name,'strict',r['fully_validated'],'raised_ms',comparison.get('raised_ms'),'native_ms',comparison.get('native_ms'),'paper_ratio',comparison.get('raised_over_paper'),'old_over_new',comparison.get('previous_first_five_over_current'),flush=True)
 assert r['fully_validated'],'Strict failure: stop before next timing case'
