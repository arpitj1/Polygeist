"""Execute the pinned ATen reduction operator against its imported C reference."""
import argparse
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=True)
source = (ROOT/'third_party/pytorch/aten/src/ATen/native/SharedReduceOps.h').read_text()
start = source.index('template <typename acc_t, typename data_t>\nstruct NanSumOps')
end = source.index('\n};', start) + 3
operator = source[start:end]
header = '''#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cfloat>
#include <vector>
#define C10_DEVICE
namespace at { bool _isnan(float x) { return std::isnan(x); } }
extern "C" void reference(int,int,float*,float*);
'''
main = r'''
int main() {
 const float edge[]={0.f,-0.f,INFINITY,-INFINITY,NAN,FLT_MAX,-FLT_MAX,
                     FLT_MIN,-FLT_MIN,0x1p-149f,-0x1p-149f,1.f,-1.f};
 long count=0;
 for(int rows:{1,3,7})for(int cols:{1,5,65,127})for(int mode=0;mode<3;++mode){
  std::vector<float> x(rows*cols),actual(rows+8,123.5f),expected=actual;
  for(size_t i=0;i<x.size();++i)
   x[i]=mode==2?-0.f:mode==1?edge[i%13]:float(int(i*37%101)-50)/32;
  auto before=x;
  NanSumOps<float,float> op;
  for(int row=0;row<rows;++row){float acc=0.f;
   for(int col=0;col<cols;++col)acc=op.reduce(acc,x[row*cols+col],col);
   expected[row+4]=op.project(acc);
  }
  reference(rows,cols,x.data(),actual.data()+4);
  for(size_t i=0;i<actual.size();++i){
   if(!(std::isnan(actual[i])&&std::isnan(expected[i])) &&
      std::memcmp(&actual[i],&expected[i],sizeof(float)))return 1;
   ++count;
  }
  if(std::memcmp(x.data(),before.data(),x.size()*sizeof(float)))return 2;
 }
 std::printf("PASS %ld ATen NanSum source outputs and guards\n",count);
}
'''
(out/'source.cpp').write_text(header + operator + main)
subprocess.run(['gcc','-O0','-ffp-contract=off','-DNETLIB_SPEC=reference','-c',
                str(ROOT/'generic_solver/lib_v3/ref_c/aten_nansum.c'),'-o',str(out/'reference.o')],check=True)
subprocess.run(['g++','-std=c++17','-O0','-ffp-contract=off',str(out/'source.cpp'),
                str(out/'reference.o'),'-o',str(out/'source_check')],check=True)
subprocess.run([str(out/'source_check')],check=True)
