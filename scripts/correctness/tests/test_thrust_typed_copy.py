"""Check typed public Thrust imports against native iterator traversal."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
INCLUDE = ROOT / 'third_party/cuda_headers/cuda_cccl/include/cccl'
SOURCE = r'''
#define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_CPP
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <cstdint>
#include <cstring>
#include <cstdio>
extern "C" void index_i32(int,int*);
extern "C" void index_f32(int,float*);
extern "C" void copy_i32(int,int*,int*);
extern "C" void copy_u32(int,unsigned*,unsigned*);
extern "C" void copy_i32_f32(int,int*,float*);
int main(){
 int values[]={INT32_MIN,INT32_MIN+1,-16777217,-16777216,-1,0,1,16777215,16777216,16777217,INT32_MAX};
 unsigned uvalues[]={0,1,2147483647u,2147483648u,4294967295u};
 int input[270],ii[270],ia[270];float fa[270],fe[270];unsigned ui[270],ua[270],ue[270];
 for(int k=0;k<270;++k){input[k]=values[k%11];ui[k]=uvalues[k%5];}
 int before[270];unsigned ubefore[270];std::memcpy(before,input,sizeof(input));std::memcpy(ubefore,ui,sizeof(ui));
 int trials=0;for(int n:{1,7,31,257}){
  for(int k=0;k<270;++k){ii[k]=ia[k]=-777;fa[k]=fe[k]=-777;ua[k]=ue[k]=777;}
  auto it=thrust::make_counting_iterator<int>(0);
  index_i32(n,ii+3);thrust::copy(thrust::seq,it,it+n,ia+3);if(std::memcmp(ii,ia,sizeof(ii)))return 1;++trials;
  index_f32(n,fe+3);thrust::copy(thrust::seq,it,it+n,fa+3);if(std::memcmp(fe,fa,sizeof(fe)))return 2;++trials;
  copy_i32(n,input,ii+3);thrust::copy(thrust::seq,input,input+n,ia+3);if(std::memcmp(ii,ia,sizeof(ii)))return 3;++trials;
  copy_u32(n,ui,ue+3);thrust::copy(thrust::seq,ui,ui+n,ua+3);if(std::memcmp(ue,ua,sizeof(ue)))return 4;++trials;
  copy_i32_f32(n,input,fe+3);thrust::copy(thrust::seq,input,input+n,fa+3);if(std::memcmp(fe,fa,sizeof(fe)))return 5;++trials;
 }
 if(std::memcmp(input,before,sizeof(input))||std::memcmp(ui,ubefore,sizeof(ui)))return 6;
 std::printf("%d configurations passed\n",trials);
}
'''


@unittest.skipUnless((INCLUDE / 'thrust/copy.h').exists() and shutil.which('g++'),
                     'public Thrust headers and host C++ compiler required')
class ThrustTypedCopy(unittest.TestCase):
    def test_native_boundaries_and_tails(self):
        variants = [('index_i32', None, 'int', True),
                    ('index_f32', None, 'float', True),
                    ('copy_i32', 'int', 'int', False),
                    ('copy_u32', 'unsigned int', 'unsigned int', False),
                    ('copy_i32_f32', 'int', 'float', False)]
        with tempfile.TemporaryDirectory() as folder:
            work = Path(folder)
            objects = []
            for name, source_type, destination_type, counting in variants:
                obj = work / (name + '.o')
                objects.append(str(obj))
                command = ['cc', '-O0', '-ffp-contract=off',
                           '-DNETLIB_SPEC=' + name,
                           '-DSCALAR_OUT=' + destination_type]
                if source_type:
                    command += ['-DSCALAR_IN=' + source_type]
                filename = 'thrust_counting_copy.c' if counting else 'thrust_typed_copy.c'
                command += ['-c', str(ROOT / 'generic_solver/lib_v3/ref_c' / filename),
                            '-o', str(obj)]
                result = subprocess.run(command, capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stderr)
            (work / 'check.cpp').write_text(SOURCE)
            result = subprocess.run(['g++', '-O0', '-std=c++17', '-ffp-contract=off',
                                     '-I' + str(INCLUDE), str(work / 'check.cpp'),
                                     *objects, '-o', str(work / 'check')],
                                    capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run([str(work / 'check')], capture_output=True,
                                    text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('20 configurations passed', result.stdout)


if __name__ == '__main__':
    unittest.main()
