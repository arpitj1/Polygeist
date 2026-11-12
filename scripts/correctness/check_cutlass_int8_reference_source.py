"""Compare the raised int8 reference with public CUTLASS host compute_gemm."""
import argparse
import os
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--cuda-include',type=Path,default=Path('/usr/local/cuda-12.6/include'))
a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
source='#include "cutlass/util/reference/host/gemm.h"\n#include <cstdint>\n#include <cstdio>\n#include <vector>\nextern "C" void reference(int,int,int,int8_t*,int8_t*,int*);\nint main(){long checked=0;\n for(int m:{1,7,17,32})for(int n:{1,9,16,48})for(int k:{1,11,64})for(int mode=0;mode<3;++mode){\n  std::vector<int8_t>a(m*k),b(k*n);std::vector<int>c(m*n+8,913),d=c;\n  const int vals[]={0,1,-1,-128,127,2,-2,63};\n  for(int i=0;i<m*k;++i)a[i]=mode==2?0:mode==1?vals[i%8]:i%41-20;\n  for(int i=0;i<k*n;++i)b[i]=mode==2?0:mode==1?vals[(i+3)%8]:i%31-15;\n  auto olda=a,oldb=b;\n  using L=cutlass::layout::RowMajor;\n  cutlass::reference::host::compute_gemm<int8_t,L,int8_t,L,int,L,int,int>(\n   {m,n,k},1,{a.data(),L(k)},{b.data(),L(n)},0,{d.data()+4,L(n)},{d.data()+4,L(n)},0);\n  reference(m,n,k,a.data(),b.data(),c.data()+4);\n  if(c!=d||a!=olda||b!=oldb)return 1;checked+=c.size();\n }\n printf("PASS %ld CUTLASS int8 GEMM source outputs and guards\\n",checked);\n}\n'
(out/'check.cpp').write_text(source)
subprocess.run(['gcc','-O0','-DNETLIB_SPEC=reference','-c',
 str(ROOT/'generic_solver/lib_v3/ref_c/cutlass_gemm_i8.c'),'-o',str(out/'reference.o')],check=True)
# The pinned CUTLASS header redeclares TensorView as an identically named alias;
# GCC needs -fpermissive for this upstream declaration. No header is modified.
subprocess.run(['g++','-std=c++17','-fpermissive','-O0',
 '-I'+str(Path(os.environ.get('POLYGEIST_DEPENDENCY_ROOT', ROOT/'third_party'))/'cutlass/include'),
 '-I'+str(Path(os.environ.get('POLYGEIST_DEPENDENCY_ROOT', ROOT/'third_party'))/'cutlass/tools/util/include'),'-I'+str(a.cuda_include),
 str(out/'check.cpp'),str(out/'reference.o'),'-o',str(out/'check')],check=True)
subprocess.run([str(out/'check')],check=True)
