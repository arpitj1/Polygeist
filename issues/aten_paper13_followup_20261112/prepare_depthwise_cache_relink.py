#!/usr/bin/env python3
"""Retest plan-cache integration while preserving validated compute/oracle objects."""
import hashlib,json,re,shlex,shutil,subprocess
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[1]
source=BASE/'gpu-sparse-depthwise-stable/aten_depthwise_conv3x3_cpu'
output=BASE/'gpu-depthwise-cache-relink';dest=output/'aten_depthwise_conv3x3_cpu'
output.mkdir(exist_ok=False);dest.mkdir();frozen=output/'frozen-runtime';frozen.mkdir()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
files=['polygeist_cublas_rt_cuda.c','polygeist_cublas_rt.h','polygeist_cuda_graph_rt.h',
       'polygeist_gpu_workspace_cache.h','polygeist_pointwise_rounding.inc','polygeist_pointwise_division.inc']
for name in files:
 for included in re.findall(r'^#include \"([^\"]+)\"', (ROOT/'runtime'/name).read_text(), re.M):
  if (ROOT/'runtime'/included).is_file() and included not in files:files.append(included)
 shutil.copy2(ROOT/'runtime'/name,frozen/name)
save(frozen/'sha256.json',{name:sha(frozen/name) for name in files})
old=json.loads((source/'build.json').read_text());previous=json.loads((source/'gpu-result.json').read_text())
assert old['rc']==0 and sha(source/'raised-program')==old['binary_sha256'] and previous['fully_validated']
assert old['configuration']['timing_samples']==5
for name in ['build.json','raised-program','harness.c','aten_depthwise_conv3x3_cpu.c','native.cpp']:
 shutil.copy2(source/name,dest/('original-'+name if name in ['build.json','raised-program'] else name))
replacements={};hashes={}
for item in old['commands'][-1]:
 if not item.endswith('.o'):continue
 obj=Path(item);target=dest/('work' if obj.parent.name=='work' else 'objects')/obj.name
 target.parent.mkdir(exist_ok=True);shutil.copy2(obj,target)
 assert sha(obj)==sha(target)
 replacements[item]=str(target);hashes[str(target.relative_to(dest))]=sha(target)
shutil.copy2(dest/'work/rt.o',dest/'original-rt.o')
compile_command=['aarch64-linux-gnu-gcc','-O2','-ffunction-sections','-fdata-sections',
 '-DPOLYGEIST_DISABLE_CUSPARSE','-DPOLYGEIST_DISABLE_CUSOLVER',
 '-I/usr/local/cuda-12.6/targets/sbsa-linux/include',
 '-I/usr/include/aarch64-linux-gnu','-c',str(frozen/'polygeist_cublas_rt_cuda.c'),'-o',str(dest/'work/rt.o')]
link=[replacements.get(x,x) for x in old['commands'][-1]];link[link.index('-o')+1]=str(dest/'raised-program')
with (dest/'build.log').open('w') as log:
 for command in [compile_command,link]:
  log.write(shlex.join(command)+'\n');log.flush();subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=180)
for name,h in hashes.items():
 if name!='work/rt.o':assert sha(dest/name)==h
old.update(commands=[compile_command,link],binary_sha256=sha(dest/'raised-program'),original_binary_sha256=old['binary_sha256'],
 runtime_only_relink=True,frozen_object_sha256_before=hashes,frozen_object_sha256_after={name:sha(dest/name) for name in hashes},
 runtime_source_sha256=json.loads((frozen/'sha256.json').read_text()),
 provenance_scope='Previously validated compiler/kernel, ABI wrapper, native ATen, original C reference, and five-pair harness objects remain unchanged; only the main runtime object is rebuilt. Workspace cache remains disabled in both plan-cache modes.')
save(dest/'build.json',old)
print('Runtime-only depthwise relink',old['binary_sha256'])
