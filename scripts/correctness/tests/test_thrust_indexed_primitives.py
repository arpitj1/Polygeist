"""Compare imported integer coordinates and predicates to public Thrust."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[3]
INCLUDE=ROOT/'third_party/cuda_headers/cuda_cccl/include/cccl'
SOURCE=r'''
#define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_CPP
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <cuda/std/functional>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <vector>
extern "C" void reference_divide(int,uint32_t,int32_t*);
extern "C" void reference_remainder(int,uint32_t,int32_t*);
extern "C" void reference_ge(int,int32_t*,int32_t,int32_t*);
extern "C" void reference_eq(int,int32_t*,int32_t*,float*);
extern "C" void reference_eq_int(int,int32_t*,int32_t*,int32_t*);
extern "C" void reference_less(int,int32_t*,int32_t*,int32_t*);
extern "C" void reference_le(int,int32_t*,int32_t,int32_t*);
int main() {
 int count=0;
 for(int n:{1,7,19,97})for(uint32_t stride:{1u,2u,7u,31u,0x7fffffffu,0x80000000u,0xffffffffu}) {
  std::vector<int32_t> actual(n+4,-999),expected=actual;
  auto first=thrust::make_counting_iterator<uint64_t>(0);
  auto divisor=thrust::make_constant_iterator<uint64_t>(stride);
  reference_divide(n,stride,expected.data()+2);
  thrust::transform(thrust::seq,first,first+n,divisor,actual.data()+2,cuda::std::divides<uint64_t>{});
  if(std::memcmp(actual.data(),expected.data(),actual.size()*4))return 1;
  reference_remainder(n,stride,expected.data()+2);
  thrust::transform(thrust::seq,first,first+n,divisor,actual.data()+2,cuda::std::modulus<uint64_t>{});
  if(std::memcmp(actual.data(),expected.data(),actual.size()*4))return 2;
  count+=2;
 }
 int32_t edges[]={INT32_MIN,INT32_MIN+1,-1,0,1,INT32_MAX-1,INT32_MAX};
 int32_t actual[11],expected[11];
 for(int32_t y:edges) {
  for(int i=0;i<11;++i)actual[i]=expected[i]=1337;
  reference_ge(7,edges,y,expected+2);
  thrust::transform(thrust::seq,edges,edges+7,thrust::make_constant_iterator(y),actual+2,cuda::std::greater_equal<int32_t>{});
  if(std::memcmp(actual,expected,sizeof(actual)))return 3;
  ++count;
  reference_le(7,edges,y,expected+2);
  thrust::transform(thrust::seq,edges,edges+7,thrust::make_constant_iterator(y),actual+2,cuda::std::less_equal<int32_t>{});
  if(std::memcmp(actual,expected,sizeof(actual)))return 5;
  ++count;
 }
 int32_t left[49],right[49];int k=0;for(auto x:edges)for(auto y:edges){left[k]=x;right[k++]=y;}
 float got[53],want[53];for(int i=0;i<53;++i)got[i]=want[i]=-3.5f;
 reference_eq(49,left,right,want+2);
 thrust::transform(thrust::seq,left,left+49,right,got+2,cuda::std::equal_to<int32_t>{});
 if(std::memcmp(got,want,sizeof(got)))return 4;
 int32_t goti[53],wanti[53];for(int i=0;i<53;++i)goti[i]=wanti[i]=-1337;
 reference_eq_int(49,left,right,wanti+2);
 thrust::transform(thrust::seq,left,left+49,right,goti+2,cuda::std::equal_to<int32_t>{});
 if(std::memcmp(goti,wanti,sizeof(goti)))return 6;
 reference_less(49,left,right,wanti+2);
 thrust::transform(thrust::seq,left,left+49,right,goti+2,cuda::std::less<int32_t>{});
 if(std::memcmp(goti,wanti,sizeof(goti)))return 7;
 std::printf("%d coordinate/predicate configurations and 147 comparison pairs passed\n",count);
}
'''

@unittest.skipUnless(shutil.which('g++') and (INCLUDE/'thrust/transform.h').exists(),
                     'public Thrust headers and host C++ compiler required')
class ThrustIndexedPrimitives(unittest.TestCase):
    def test_native_typed_functors(self):
        with tempfile.TemporaryDirectory() as folder:
            work=Path(folder);objects=[]
            for name,source,flags in [
                ('reference_divide','thrust_counting_quotient.c',['-DREMAINDER=0']),
                ('reference_remainder','thrust_counting_quotient.c',['-DREMAINDER=1']),
                ('reference_ge','thrust_integer_compare.c',['-DEQUAL=0','-DRIGHT_SCALAR=1','-DSCALAR_OUT=int']),
                ('reference_eq','thrust_integer_compare.c',['-DEQUAL=1','-DRIGHT_SCALAR=0','-DSCALAR_OUT=float']),
                ('reference_eq_int','thrust_integer_compare.c',['-DEQUAL=1','-DRIGHT_SCALAR=0','-DSCALAR_OUT=int']),
                ('reference_less','thrust_integer_compare.c',['-DEQUAL=2','-DRIGHT_SCALAR=0','-DSCALAR_OUT=int']),
                ('reference_le','thrust_integer_compare.c',['-DEQUAL=3','-DRIGHT_SCALAR=1','-DSCALAR_OUT=int']),
            ]:
                obj=work/(name+'.o');objects.append(str(obj))
                result=subprocess.run(['cc','-O0','-DNETLIB_SPEC='+name,*flags,'-c',
                    str(ROOT/'generic_solver/lib_v3/ref_c'/source),'-o',str(obj)],
                    capture_output=True,text=True,timeout=60)
                self.assertEqual(result.returncode,0,result.stderr)
            (work/'check.cpp').write_text(SOURCE)
            result=subprocess.run(['g++','-O0','-std=c++17','-I'+str(INCLUDE),
                str(work/'check.cpp'),*objects,'-o',str(work/'check')],
                capture_output=True,text=True,timeout=120)
            self.assertEqual(result.returncode,0,result.stderr)
            result=subprocess.run([str(work/'check')],capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn('70 coordinate/predicate configurations and 147 comparison pairs passed',result.stdout)
if __name__=='__main__':unittest.main()
