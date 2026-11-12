#!/usr/bin/env python3
"""Isolate a runtime change against validated conv1d compute and oracle objects."""
import hashlib,json,shlex,shutil,subprocess
from pathlib import Path
BASE=Path(__file__).resolve().parent
source=BASE/'gpu-conv1d/aten_conv1d'
output=BASE/'gpu-conv1d-cache-relink'; dest=output/'aten_conv1d'
dest.mkdir(exist_ok=False)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,data):path.write_text(json.dumps(data,indent=2)+'\n')
old=json.loads((source/'build.json').read_text())
assert old['rc']==0 and sha(source/'raised-program')==old['binary_sha256']
assert json.loads((source/'gpu-result.json').read_text())['fully_validated']
assert old['configuration']['timing_samples']==5
for name in ['build.json','raised-program','harness.c','aten_conv1d.c','native.cpp']:
 shutil.copy2(source/name,dest/('original-'+name if name in ['build.json','raised-program'] else name))
replacements={}; hashes={}
for item in old['commands'][-1]:
 if not item.endswith('.o'):continue
 obj=Path(item);target=dest/('work' if obj.parent.name=='work' else 'objects')/obj.name
 target.parent.mkdir(exist_ok=True);shutil.copy2(obj,target)
 assert sha(obj)==sha(target)
 replacements[item]=str(target);hashes[str(target.relative_to(dest))]=sha(target)
compile_command=['aarch64-linux-gnu-gcc','-O2','-ffunction-sections','-fdata-sections',
 '-DPOLYGEIST_DISABLE_CUSPARSE','-DPOLYGEIST_DISABLE_CUSOLVER','-DPOLYGEIST_ENABLE_CUTENSOR',
 '-I/home/arjaiswal/cutensor_sbsa/include','-I/usr/local/cuda-12.6/targets/sbsa-linux/include',
 '-I/usr/include/aarch64-linux-gnu','-c',str(output/'frozen-runtime/polygeist_cublas_rt_cuda.c'),
 '-o',str(dest/'work/rt.o')]
link_command=[replacements.get(x,x) for x in old['commands'][-1]]
link_command[link_command.index('-o')+1]=str(dest/'raised-program')
with (dest/'build.log').open('w') as log:
 for command in [compile_command,link_command]:
  log.write(shlex.join(command)+'\n');log.flush()
  subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=180)
for name,h in hashes.items():
 if name!='work/rt.o':assert sha(dest/name)==h
old.update(commands=[compile_command,link_command],binary_sha256=sha(dest/'raised-program'),
 original_binary_sha256=old['binary_sha256'],runtime_only_relink=True,
 frozen_object_sha256_before=hashes,
 frozen_object_sha256_after={name:sha(dest/name) for name in hashes},
 runtime_source_sha256=json.loads((output/'frozen-runtime/sha256.json').read_text()),
 provenance_scope='Reuse previously validated compiler, ABI, C reference, native ATen and five-pair harness objects; change only the runtime object. This is not a fresh compiler build or updated proof receipt.')
save(dest/'build.json',old)
print('Relinked runtime-only conv1d',old['binary_sha256'])
