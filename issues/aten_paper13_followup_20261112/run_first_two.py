#!/usr/bin/env python3
"""Validate frozen nested-all and conv1d builds, serially, with five pairs."""
import csv
import json
from pathlib import Path
import re
import shlex
import sys
from types import SimpleNamespace

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[1]
sys.path.insert(0,str(ROOT/'scripts/correctness'))
import aten_fixture_batch as F
from aten_frozen_runtime_campaign import save,transport,remote_libraries,check_harness_count

adapter='/home/arjaiswal/.config/polygeist-silicon/reproduction_20261109/remote.py'
ld='/home/nvidia/polygeist_aten_native_24_12:/home/nvidia/polygeist_cuda_libs:/home/nvidia/jetson-cuda-libs:/home/nvidia/cuda-12.6/lib64:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu'
board=transport(adapter)
env='LD_LIBRARY_PATH='+shlex.quote(ld)+' '
unit=BASE/'segmented-logical-unit/test-cub-segmented-logical'
assert not (unit.parent/'gpu-test.json').exists()
remote_unit='/tmp/ll-segmented-logical-unit-'+F.sha(unit)[:12]
assert board.upload(unit,remote_unit)==0
rc,log=board.run('chmod +x '+shlex.quote(remote_unit)+' && sha256sum '+shlex.quote(remote_unit),timeout=15)
(unit.parent/'remote-hash.log').write_text(log)
assert rc==0 and F.sha(unit) in log
rc,log=board.run(env+'timeout 120 '+shlex.quote(remote_unit),timeout=125)
(unit.parent/'gpu-test.log').write_text(log)
record=dict(rc=rc,passed=rc==0 and 'PASS' in log,binary_sha256=F.sha(unit),
    cub_object_sha256=F.sha(BASE/'gpu-nested-all/aten_nested_all_cpu/work/cub_runtime.o'),
    source_sha256=F.sha(ROOT/'scripts/correctness/tests/test_cub_segmented_logical_cuda.cpp'))
save(unit.parent/'gpu-test.json',record)
print('UNIT',rc,log.strip(),flush=True)
assert record['passed'],'Unit failed; no benchmark execution'

paper={r['kernel']:r for r in csv.DictReader((ROOT/'issues/aten_c_kernels/paper_analysis/aten_gpu_nonpointwise_115.csv').open())}
for folder,name,kinds in [('gpu-nested-all','aten_nested_all_cpu',['normal','edge','zeros','paper']),('gpu-conv1d','aten_conv1d',['paper'])]:
    output=BASE/folder;dest=output/name
    built=json.loads((dest/'build.json').read_text())
    assert built['rc']==0 and built['configuration']['timing_samples']==5
    assert F.sha(dest/'raised-program')==built['binary_sha256']
    check_harness_count(dest,5)
    assert not (dest/'gpu-result.json').exists()
    remote='/tmp/ll-new-aten-libraries-'+built['binary_sha256'][:12]
    assert board.upload(dest/'raised-program',remote)==0
    assert board.run('chmod +x '+shlex.quote(remote),timeout=15)[0]==0
    before=remote_libraries(board,remote,env,output,'before')
    F.run_gpu([name],SimpleNamespace(output=output,transport=adapter,remote_library_path=ld,
        cudnn_plan_cache=True,run_timeout=180,correctness_kinds=kinds,skip_timing=False,time_validated_normal_inputs=False))
    after=remote_libraries(board,remote,env,output,'after')
    save(output/'dynamic-library-stability.json',dict(unchanged=before==after))
    assert before==after
    r=json.loads((dest/'gpu-result.json').read_text())
    result=dict(kernel=name,fully_validated=r['fully_validated'],paper=paper[name],actual_samples=5,
        baseline_scope=built['baseline'],binary_sha256=r['binary_sha256'],
        selected_statistic='minimum_first_five_ms',cudnn_plan_cache=True)
    if name=='aten_nested_all_cpu':
        result.update(native_baseline_matches_paper_scope=True,
            native_inputs_match_paper=False,paper_native_scope_note=built['paper_baseline_input_caveat'])
    result['traced_library_calls']=re.findall(r'^POLYGEIST_V2_CALL .*call=(\S+)',(dest/'paper-gpu.log').read_text(),re.M)
    if r['fully_validated'] and r.get('timing',{}).get('passed'):
        assert len(r['timing']['samples'])==10
        t=r['timing']['minimum_first_five_ms']
        result.update(raised_ms=t['raised'],native_ms=t['native_aten_cuda'],
            current_native_over_raised=t['native_aten_cuda']/t['raised'],
            raised_over_paper=t['raised']*1000/float(paper[name]['raised_us']),
            timing_statistic_matches_paper='20' not in paper[name]['timing_protocol'])
    else:result['status']='strict_failed_no_timing'
    save(dest/'paper-comparison.json',result)
    print('RESULT',name,'pass',r['fully_validated'],'raised_ms',result.get('raised_ms'),'native_ms',result.get('native_ms'),'raised_over_paper',result.get('raised_over_paper'),flush=True)
