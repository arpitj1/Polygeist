"""Public CPU/CUDA scalar definitions must agree before they are matched."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
SOURCE = r'''
#include "runtime/polygeist_aten_cuda_scalar_support.cpp"
#include <cstdint>
#include <cstring>
#include <cstdio>
int main() {
 const float boundaries[] = {-INFINITY,-100.f,-4.25f,-2.f,-1.f,-0.25f,
   -0.f,0.f,0x1p-149f,0.25f,0.5f,1.f,2.f,8.f,32.f,INFINITY,NAN};
 int trials=0;
 auto check = [&](float x) {
   float cpu=::trigamma(x), cuda=calc_cuda_trigammaf(x);
   ++trials;
   return (std::isnan(cpu)&&std::isnan(cuda)) ||
          std::memcmp(&cpu,&cuda,sizeof(float))==0;
 };
 for(float x:boundaries) {
   if(!check(x)||!check(std::nextafter(x,-INFINITY))||
      !check(std::nextafter(x,INFINITY)))return 1;
 }
 uint32_t state=0x45213789u;
 for(int i=0;i<10000;++i) {
   state=1664525u*state+1013904223u;
   float x;std::memcpy(&x,&state,sizeof(x));
   if(!check(x))return 2;
 }
 std::printf("%d host comparisons passed\n",trials);
}
'''


@unittest.skipUnless(
    shutil.which('g++') and
    (ROOT/'third_party/pytorch/aten/src/ATen/native/cuda/Math.cuh').exists() and
    (ROOT/'third_party/cuda_headers/cudart/include/cuda_runtime.h').exists(),
    'public ATen/CUDA headers and host C++ compiler required')
class AtenCudaMathReference(unittest.TestCase):
    def test_trigamma_float_overloads_and_boundaries(self):
        with tempfile.TemporaryDirectory() as folder:
            work=Path(folder)
            (work/'check.cpp').write_text(SOURCE)
            includes=['.', 'generic_solver/lib_v3/support', 'third_party/pytorch',
                      'third_party/pytorch/aten/src',
                      'third_party/cuda_headers/cudart/include',
                      'third_party/cuda_headers/cuda_crt/include']
            result=subprocess.run(['g++','-O0','-std=c++17','-ffp-contract=off',
                *['-I'+str(ROOT/p) for p in includes],str(work/'check.cpp'),
                '-o',str(work/'check')],capture_output=True,text=True,timeout=120)
            self.assertEqual(result.returncode,0,result.stderr)
            result=subprocess.run([str(work/'check')],capture_output=True,
                                  text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn('10051 host comparisons passed',result.stdout)


if __name__=='__main__':
    unittest.main()
