"""Compare the imported polar C reference with the pinned ATen CUDA lambda."""
import argparse
import os
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--torch-include',type=Path,default=Path(os.environ.get(
    'ATEN_NATIVE_TORCH_ROOT','/tmp/aten_native_sysroot_24_12/usr/local/lib/python3.12/dist-packages/torch'))/'include')
args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
source=(ROOT/'third_party/pytorch/aten/src/ATen/native/cuda/ComplexKernel.cu').read_text()
body=re.findall(r'return c10::complex<scalar_t>\(a \* std::cos\(b\), a \* std::sin\(b\)\);',source)
assert len(body)==1
header='''#include <c10/util/complex.h>
#include <cmath>
#include <cfloat>
#include <cstring>
#include <cstdio>
#include <vector>
using scalar_t=float;
extern "C" void reference(int,float*,float*,float*,float*);
c10::complex<float> upstream(float a,float b){ BODY }
'''.replace('BODY',body[0])
main=r'''
int main(){long checked=0;
 const float edge[]={0.f,-0.f,INFINITY,-INFINITY,NAN,FLT_MAX,-FLT_MAX,FLT_MIN,-FLT_MIN,0x1p-149f,-0x1p-149f,1.f,-1.f,0x1p+20f};
 for(int n:{1,7,65,1025})for(int mode=0;mode<3;++mode){
  std::vector<float>a(n),b(n),re(n+8,913.25f),im=re,er=re,ei=re;
  for(int i=0;i<n;++i){a[i]=mode==2?-0.f:mode==1?edge[i%14]:float(i%41-20)/16;
   b[i]=mode==2?-0.f:mode==1?edge[(i+5)%14]:float(i%59-29)/8;
   auto z=upstream(a[i],b[i]);er[i+4]=z.real();ei[i+4]=z.imag();}
  auto olda=a,oldb=b;reference(n,a.data(),b.data(),re.data()+4,im.data()+4);
  for(size_t i=0;i<re.size();++i)for(int part=0;part<2;++part){
   float actual=part?im[i]:re[i],expected=part?ei[i]:er[i];
   if(!(std::isnan(actual)&&std::isnan(expected))&&std::memcmp(&actual,&expected,4))return 1;
   ++checked;}
  if(std::memcmp(a.data(),olda.data(),n*4)||std::memcmp(b.data(),oldb.data(),n*4))return 2;
 }
 std::printf("PASS %ld ATen polar source outputs and guards\n",checked);
}
'''
(out/'source.cpp').write_text(header+main)
subprocess.run(['gcc','-O0','-ffp-contract=off','-DNETLIB_SPEC=reference','-c',
 str(ROOT/'generic_solver/lib_v3/ref_c/thrust_polar_components.c'),'-o',str(out/'reference.o')],check=True)
subprocess.run(['g++','-std=c++17','-O0','-ffp-contract=off','-I'+str(args.torch_include),
 str(out/'source.cpp'),str(out/'reference.o'),'-o',str(out/'check')],check=True)
subprocess.run([str(out/'check')],check=True)
