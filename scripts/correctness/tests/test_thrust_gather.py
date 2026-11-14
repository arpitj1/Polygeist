"""Compare gather imports with public Thrust across shapes and index patterns."""
from pathlib import Path
import shutil,subprocess,tempfile,unittest
ROOT=Path(__file__).resolve().parents[3]
SOURCE = r'''
#define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_CPP
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/execution_policy.h>
#include "polygeist_thrust_gather.h"
#include <vector>
#include <cstring>
#include <cstdio>
extern "C" void ref0(int,int,int*,float*,float*);
extern "C" void ref1(int,int,int,int*,float*,float*);
extern "C" void ref2(int,int,int,int*,float*,float*);
extern "C" void ref3(int,int,int,int*,float*,float*);
extern "C" void ref4(int,int,int*,float*,float*);
extern "C" void ref5(int,int,int,int*,int*,float*,float*);
extern "C" void ref6(int,int,int,int*,int*,int*);
template<int Mode> void actual(int n,int cols,int source,const int32_t*idx,const float*in,float*out) {
 auto first=thrust::make_transform_iterator(thrust::make_counting_iterator<int64_t>(0),PolygeistGatherIndex<Mode>{idx,cols,source});
 thrust::gather(thrust::seq,first,first+n,in,out);
}
int main() {
 int checked=0;
 for(int mode=0;mode<5;++mode) for(int rows:{1,3,7}) for(int cols:{1,4,9}) for(int source:{1,5,11}) {
  int n=mode==4?rows:rows*cols;
  int size=mode==0?source:mode==1?source*cols:mode==4?rows*cols:rows*source;
  int ni=mode==0||mode==3?n:mode==2?cols:rows;
  int limit=mode==4?cols:source;
  std::vector<int> idx(ni);
  for(int i=0;i<ni;++i)idx[i]=(i%4==0?limit-1:(i*7)%limit);
  auto savedIdx=idx;
  std::vector<float> input(size),out(n+4,-123.5f),expected=out;
  uint32_t bits[]={0,0x80000000u,0x7fc01234u,0x7f800000u,0xff800000u,0x3f800000u,0xc0100000u};
  for(int i=0;i<size;++i)std::memcpy(&input[i],&bits[i%7],4);
  auto savedInput=input;
  switch(mode) {
  case 0:ref0(n,source,idx.data(),input.data(),expected.data());actual<0>(n,cols,source,idx.data(),input.data(),out.data());break;
  case 1:ref1(rows,cols,source,idx.data(),input.data(),expected.data());actual<1>(n,cols,source,idx.data(),input.data(),out.data());break;
  case 2:ref2(rows,cols,source,idx.data(),input.data(),expected.data());actual<2>(n,cols,source,idx.data(),input.data(),out.data());break;
  case 3:ref3(rows,cols,source,idx.data(),input.data(),expected.data());actual<3>(n,cols,source,idx.data(),input.data(),out.data());break;
  case 4:ref4(rows,cols,idx.data(),input.data(),expected.data());actual<4>(n,cols,source,idx.data(),input.data(),out.data());break;
  }
  if(std::memcmp(out.data(),expected.data(),out.size()*4)||std::memcmp(input.data(),savedInput.data(),size*4)||idx!=savedIdx){std::printf("FAIL %d %d %d %d\n",mode,rows,cols,source);return 1;}
  ++checked;
 }
 for(int rows:{1,3,7}) for(int cols:{1,4,9}) for(int n:{1,5,11}) {
  std::vector<int> row(n),column(n);
  for(int i=0;i<n;++i){row[i]=(i*7+rows-1)%rows;column[i]=(i*3+cols-1)%cols;}
  auto savedRow=row,savedColumn=column;
  std::vector<float> input(rows*cols),out(n+4,-123.5f),expected=out;
  uint32_t bits[]={0,0x80000000u,0x7fc01234u,0x7f800000u,0xff800000u};
  for(int i=0;i<rows*cols;++i)std::memcpy(&input[i],&bits[i%5],4);
  auto savedInput=input;
  ref5(n,rows,cols,row.data(),column.data(),input.data(),expected.data());
  auto first=thrust::make_transform_iterator(thrust::make_counting_iterator<int64_t>(0),PolygeistGatherIndex<5>{row.data(),cols,cols,column.data()});
  thrust::gather(thrust::seq,first,first+n,input.data(),out.data());
  if(std::memcmp(out.data(),expected.data(),out.size()*4)||std::memcmp(input.data(),savedInput.data(),input.size()*4)||row!=savedRow||column!=savedColumn)return 2;
  ++checked;
  std::vector<int> integerInput(rows*cols),integerOut(n*cols+4,-123),integerExpected=integerOut;
  int values[]={INT32_MIN,INT32_MAX,-1,0,1};
  for(int i=0;i<rows*cols;++i)integerInput[i]=values[i%5];
  auto savedIntegerInput=integerInput;
  ref6(n,cols,rows,row.data(),integerInput.data(),integerExpected.data());
  auto rowFirst=thrust::make_transform_iterator(thrust::make_counting_iterator<int64_t>(0),PolygeistGatherIndex<1>{row.data(),cols,cols});
  thrust::gather(thrust::seq,rowFirst,rowFirst+n*cols,integerInput.data(),integerOut.data());
  if(integerOut!=integerExpected||integerInput!=savedIntegerInput||row!=savedRow)return 3;
  ++checked;
 }
 std::printf("%d public gather configurations passed, including output guards and exceptional-value bit patterns\n",checked);
}
'''

class ThrustGather(unittest.TestCase):
 @unittest.skipUnless(shutil.which('g++') and shutil.which('cc'),'native C/C++ compilers required')
 def test_native_index_views(self):
  with tempfile.TemporaryDirectory() as d:
   w=Path(d);objects=[]
   for mode in range(7):
    obj=w/f'ref{mode}.o';objects.append(str(obj))
    p=subprocess.run(['cc','-O0','-DMODE='+str(1 if mode == 6 else mode),'-DDATA_TYPE='+('int' if mode == 6 else 'float'),'-DNETLIB_SPEC=ref'+str(mode),'-c',str(ROOT/'generic_solver/lib_v3/ref_c/thrust_gather.c'),'-o',str(obj)],capture_output=True,text=True,timeout=60)
    self.assertEqual(p.returncode,0,p.stderr)
   (w/'check.cpp').write_text(SOURCE)
   p=subprocess.run(['g++','-O0','-std=c++17','-I'+str(ROOT/'third_party/cuda_headers/cuda_cccl/include/cccl'),'-I'+str(ROOT/'runtime'),str(w/'check.cpp'),*objects,'-o',str(w/'check')],capture_output=True,text=True,timeout=120)
   self.assertEqual(p.returncode,0,p.stderr)
   p=subprocess.run([str(w/'check')],capture_output=True,text=True,timeout=30)
   self.assertEqual(p.returncode,0,p.stdout+p.stderr)
   self.assertIn('189 public gather configurations passed',p.stdout)
if __name__=='__main__':unittest.main()
