"""Typed traversal imports retain integer wraparound and float copy bits."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[3]
INCLUDE=ROOT/'third_party/cuda_headers/cuda_cccl/include/cccl'
SOURCE=r'''

#define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_CPP
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <cuda/std/functional>
#include <cstdint>
#include <cstring>
#include <cstdio>
extern "C" void reference_minus(int,uint32_t*,uint32_t*,int32_t*);
extern "C" void reference_copy(int,float*,float*);
int main(){
 uint32_t edges[]={0,1,2,0x7fffffffu,0x80000000u,0xfffffffeu,0xffffffffu};
 uint32_t x[49],y[49];int k=0;for(auto a:edges)for(auto b:edges){x[k]=a;y[k++]=b;}
 int32_t actual[53],expected[53];for(int i=0;i<53;++i)actual[i]=expected[i]=12345;
 reference_minus(49,x,y,expected+2);
 thrust::transform(thrust::seq,x,x+49,y,actual+2,cuda::std::minus<uint32_t>{});
 if(std::memcmp(actual,expected,sizeof(actual)))return 1;
 uint32_t bits[]={0,0x80000000u,1,0x80000001u,0x7f800000u,0xff800000u,0x7fc00001u,0xffc00001u,0x7f800001u};
 float input[9],got[13],want[13];std::memcpy(input,bits,sizeof(bits));
 for(int i=0;i<13;++i)got[i]=want[i]=-999;
 reference_copy(9,input,want+2);thrust::copy(thrust::seq,input,input+9,got+2);
 if(std::memcmp(got,want,sizeof(got))||std::memcmp(input,bits,sizeof(bits)))return 2;
 std::printf("49 integer pairs and 9 float bit patterns passed\n");
}
'''

@unittest.skipUnless(shutil.which('g++') and (INCLUDE/'thrust/copy.h').exists(),
                     'public Thrust headers and host C++ compiler required')
class ThrustScalarTraversals(unittest.TestCase):
    def test_native_integer_difference_and_float_copy(self):
        with tempfile.TemporaryDirectory() as folder:
            work=Path(folder);objects=[]
            variants=[('reference_minus','thrust_arithmetic.c',
                       ['-DSCALAR=unsigned int','-DSCALAR_OUT=int','-DTHRUST_OPERATION=1']),
                      ('reference_copy','thrust_typed_copy.c',
                       ['-DSCALAR_IN=float','-DSCALAR_OUT=float'])]
            for name,filename,flags in variants:
                obj=work/(name+'.o');objects.append(str(obj))
                result=subprocess.run(['cc','-O0','-DNETLIB_SPEC='+name,*flags,'-c',
                    str(ROOT/'generic_solver/lib_v3/ref_c'/filename),'-o',str(obj)],
                    capture_output=True,text=True,timeout=60)
                self.assertEqual(result.returncode,0,result.stderr)
            (work/'check.cpp').write_text(SOURCE)
            result=subprocess.run(['g++','-O0','-std=c++17','-I'+str(INCLUDE),
                str(work/'check.cpp'),*objects,'-o',str(work/'check')],
                capture_output=True,text=True,timeout=120)
            self.assertEqual(result.returncode,0,result.stderr)
            result=subprocess.run([str(work/'check')],capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn('49 integer pairs and 9 float bit patterns passed',result.stdout)

if __name__=='__main__':unittest.main()
