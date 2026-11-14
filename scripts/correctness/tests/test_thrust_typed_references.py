"""Compare adapted C imports with their pinned public Thrust implementations."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[3]
INCLUDE=ROOT/'third_party/cuda_headers/cuda_cccl/include/cccl'


@unittest.skipUnless((INCLUDE/'thrust/sequence.h').exists() and shutil.which('g++'),
                     'public Thrust headers and host C++ compiler required')
class ThrustTypedReferences(unittest.TestCase):
    def test_sequence_and_integer_clamp_native_boundaries(self):
        source=r'''
#define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_CPP
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <cuda/functional>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
extern "C" void reference_sequence(int,float,float,float*);
extern "C" void reference_clamp(int,int*,int,int,signed char*);
int main(){
 const float values[]={-0.f,0.f,1.f,-1.f,0.375f,0x1p-149f,INFINITY,-INFINITY,NAN};
 const int lengths[]={1,7,31,4096};float got[4100],want[4100];int trials=0;
 for(int n:lengths)for(float init:values)for(float step:values){
  for(int i=0;i<4100;++i)got[i]=want[i]=-777;
  reference_sequence(n,init,step,want+2);
  thrust::sequence(thrust::seq,got+2,got+2+n,init,step);
  for(int i=0;i<4100;++i){
   if(std::isnan(got[i])&&std::isnan(want[i]))continue;
   if(std::memcmp(&got[i],&want[i],sizeof(float)))return 1;
  }++trials;
 }
 int input[]={INT32_MIN,-129,-128,-127,-1,0,1,126,127,128,INT32_MAX};
 int before[11];std::memcpy(before,input,sizeof(input));
 for(int lo:input)for(int hi:input){
  signed char actual[15],expected[15];std::memset(actual,42,15);std::memset(expected,42,15);
  reference_clamp(11,input,lo,hi,expected+2);
  thrust::transform(thrust::seq,input,input+11,actual+2,[=](int x){
   return static_cast<signed char>(cuda::minimum<int>{}(hi,cuda::maximum<int>{}(lo,x)));
  });
  if(std::memcmp(actual,expected,15)||std::memcmp(before,input,sizeof(input)))return 2;
  ++trials;
 }
 std::printf("%d configurations passed\n",trials);
}
'''
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);objects=[]
            for filename,symbol in [('thrust_sequence.c','reference_sequence'),
                                    ('thrust_integer_clamp.c','reference_clamp')]:
                obj=p/(symbol+'.o');objects.append(str(obj))
                result=subprocess.run(['cc','-O0','-ffp-contract=off',
                    '-DNETLIB_SPEC='+symbol,'-c',
                    str(ROOT/'generic_solver/lib_v3/ref_c'/filename),'-o',str(obj)],
                    capture_output=True,text=True,timeout=60)
                self.assertEqual(result.returncode,0,result.stderr)
            (p/'check.cpp').write_text(source)
            result=subprocess.run(['g++','-O0','-std=c++17','-ffp-contract=off',
                '-I'+str(INCLUDE),str(p/'check.cpp'),*objects,'-o',str(p/'check')],
                capture_output=True,text=True,timeout=120)
            self.assertEqual(result.returncode,0,result.stderr)
            result=subprocess.run([str(p/'check')],capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn('445 configurations passed',result.stdout)


if __name__=='__main__':
    unittest.main()
