#!/usr/bin/env python3
"""Prepare pinned public dependencies. Existing checkouts are never reset."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import urllib.request
from common import ARTIFACT, ROOT, PYTHON, command, environment, required, save, setting

DEPS=ARTIFACT/'dependencies'
LOG=DEPS/'setup.log'

def clone(url, revision, dest):
    dest=Path(dest).resolve()
    if not dest.exists():
        dest.mkdir(parents=True)
        command(['git','init',dest],LOG)
        command(['git','-C',dest,'remote','add','origin',url],LOG)
        command(['git','-C',dest,'fetch','--depth','1','origin',revision],LOG,timeout=7200)
        command(['git','-C',dest,'checkout','--detach','FETCH_HEAD'],LOG)
    if not (dest/'.git').exists():raise RuntimeError(f'{dest} exists but is not a checkout; move it or supply the pinned dependency')
    actual=subprocess.check_output(['git','-C',dest,'rev-parse','HEAD'],text=True).strip()
    if actual!=revision:raise RuntimeError(f'{dest}: expected {revision}, found {actual}; existing checkout left untouched')
    return dest

def patch(source, path):
    if subprocess.run(['git','-C',source,'apply','--reverse','--check',path],
                      stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:
        command(['git','-C',source,'apply','--check',path],LOG)
        command(['git','-C',source,'apply',path],LOG)

def sources():
    for row in json.loads((ARTIFACT/'dependency_sources/repos.json').read_text()):
        clone(row['url'],row['revision'],ROOT/row['path'])
    cccl=clone('https://github.com/NVIDIA/cccl.git','477f8bcb27eb80c28e18bbd97e5dde80ecfc648b',DEPS/'cccl')
    includes=ROOT/'third_party/cuda_headers/cuda_cccl/include/cccl'
    includes.mkdir(parents=True,exist_ok=True)
    for name,source in [('cub',cccl/'cub/cub'),('thrust',cccl/'thrust/thrust'),('cuda',cccl/'libcudacxx/include/cuda')]:
        target=includes/name
        if not target.exists():target.symlink_to(os.path.relpath(source,includes),target_is_directory=True)
    verify_sources()

def verify_sources():
    bad=[];checked=0
    for p in (ROOT/'generic_solver/lib_v3/manifest').glob('*.json'):
        row=json.loads(p.read_text())
        upstreams=row.get('upstream_files',[])+([row['upstream']] if row.get('upstream') else [])
        for u in upstreams:
            path=ROOT/u['path'];checked+=1
            actual=hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
            if actual!=u['sha256']:bad.append(dict(path=u['path'],actual=actual,expected=u['sha256']))
    save(DEPS/'upstream-audit.json',dict(checked=checked,mismatches=bad))
    if bad:raise RuntimeError(f'{len(bad)} upstream source hash mismatches; see {DEPS}/upstream-audit.json')
    print('Verified',checked,'upstream references')

def kernelfarer():
    source=clone('https://github.com/jaopaulolc/KernelFaRer.git','9d04406e59863512d6296f588d3fad815a01c536',
                 required('kernelfarer_root'))
    patch(source,ARTIFACT/'benchmarks/kernelfarer/kernelfarer.patch')

def ginsbach():
    target=Path(required('ginsbach_root')).resolve()
    source=clone('https://github.com/cc18ginsbach/llvm.git','113d41f6e3695045fcc6e93166b3d1f5df1a8074',target/'src')
    clone('https://github.com/asplos18ginsbach/clang.git','5e3d3c0055dc731abd506e18c320f227168f728b',source/'tools/clang')
    patch(source,ARTIFACT/'benchmarks/ginsbach/ginsbach-linearalgebra-cuda13.patch')
    (target/'bin').mkdir(exist_ok=True)
    alias=target/'bin/pypy'
    if not alias.exists():alias.symlink_to(PYTHON)
    env=environment();env['PATH']=str(target/'bin')+':'+env['PATH']
    command(['cmake','-S',source,'-B',target/'build','-G','Ninja','-DCMAKE_BUILD_TYPE=Release',
             '-DCMAKE_C_COMPILER=clang-14','-DCMAKE_CXX_COMPILER=clang++-14','-DLLVM_TARGETS_TO_BUILD=X86',
             '-DLLVM_ENABLE_ASSERTIONS=ON','-DLLVM_INCLUDE_TESTS=OFF','-DLLVM_INCLUDE_EXAMPLES=OFF',
             '-DLLVM_INCLUDE_BENCHMARKS=OFF'],LOG,env)
    command(['cmake','--build',target/'build','--target','clang','opt','lilac_target','--',
             '-j'+str(setting('jobs',3))],LOG,env,timeout=43200)

def speq():
    target=Path(required('speq_root')).resolve();target.mkdir(parents=True,exist_ok=True)
    archive=target/'lleq-artifact.tar.gz'
    if not (target/'lleq-artifact').exists():
        url='https://zenodo.org/api/records/10802555/files/lleq-artifact.tar.gz/content'
        if not archive.exists():
            print('Downloading the 6.3 GB published SpEQ archive',flush=True)
            urllib.request.urlretrieve(url,archive)
        digest=hashlib.md5()
        with archive.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
        if digest.hexdigest()!='1070e9bb41a12cf5847c1bcfa7edc267':raise RuntimeError('SpEQ archive checksum mismatch')
        # Extract the source artifact, not its retained binaries/results elsewhere.
        with tarfile.open(archive) as tar:
            members=[m for m in tar.getmembers() if Path(m.name).parts[0]=='lleq-artifact']
            for m in members:
                if not (target/m.name).resolve().is_relative_to(target):raise RuntimeError('Unsafe archive path')
                if m.issym() or m.islnk():raise RuntimeError('Inspect archive links before extraction')
            tar.extractall(target,members=members)
    command(['docker','build','--build-arg','BUILD_JOBS='+str(setting('jobs',3)),
             '-t',setting('speq_image','speq-lleq:extract'),ARTIFACT/'dependency_sources'],LOG,timeout=43200)

def ggml():
    target=Path(environment()['POLYGEIST_GGML_BUILD']).resolve()
    command(['cmake','-S',ROOT/'third_party/whisper.cpp','-B',target,'-G','Ninja',
        '-DCMAKE_TOOLCHAIN_FILE='+str(ARTIFACT/'dependency_sources/aarch64-cuda.cmake'),
        '-DCMAKE_BUILD_TYPE=Release','-DGGML_CUDA=ON','-DGGML_CUDA_NCCL=OFF','-DGGML_NATIVE=OFF',
        '-DGGML_CCACHE=OFF','-DGGML_OPENMP=OFF','-DGGML_CUDA_FA=OFF','-DWHISPER_BUILD_EXAMPLES=OFF',
        '-DWHISPER_BUILD_TESTS=OFF','-DWHISPER_BUILD_SERVER=OFF','-DCMAKE_CUDA_ARCHITECTURES=87',
        '-DCMAKE_CUDA_FLAGS=-target-dir sbsa-linux',
        '-DCMAKE_CUDA_COMPILER='+str(setting('nvcc','/usr/local/cuda/bin/nvcc'))],LOG)
    command(['cmake','--build',target,'-j',str(setting('jobs',3))],LOG,timeout=43200)

def specs():
    verify_sources()
    command([PYTHON,ROOT/'scripts/correctness/lib_v2/build_lib_v2.py','-j',str(setting('jobs',3)),
             '--timeout','300'],LOG,timeout=14400)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('component',choices=['sources','verify-sources','kernelfarer','ginsbach','speq','ggml','specs'])
    a=p.parse_args();DEPS.mkdir(parents=True,exist_ok=True)
    {'sources':sources,'verify-sources':verify_sources,'kernelfarer':kernelfarer,
     'ginsbach':ginsbach,'speq':speq,'ggml':ggml,'specs':specs}[a.component]()
