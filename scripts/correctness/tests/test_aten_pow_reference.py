"""Compare the shared-base power import to public ATen vector code."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
SOURCE = r'''
#include <ATen/cpu/vec/vec_base.h>
#include <cmath>
#include <cstdio>
#include <cstring>
extern "C" void reference_pow(int,float,float*,float*);
int main() {
 using V=at::vec::Vectorized<float>;
 const float values[]={-INFINITY,-2.f,-1.f,-0.f,0.f,0x1p-149f,0.5f,1.f,2.f,INFINITY,NAN};
 float exponents[V::size()], actual[V::size()], expected[V::size()];int trials=0;
 for(float base:values)for(float exponent:values){
  for(int i=0;i<V::size();++i)exponents[i]=exponent;
  V result=V(base).pow(V::loadu(exponents));result.store(actual);
  reference_pow(V::size(),base,exponents,expected);
  for(int i=0;i<V::size();++i){
   if(std::isnan(actual[i])&&std::isnan(expected[i]))continue;
   if(std::memcmp(actual+i,expected+i,sizeof(float)))return 1;
  }++trials;
 }
 std::printf("%d native ATen configurations passed\n",trials);
}
'''


@unittest.skipUnless((ROOT / 'third_party/pytorch/aten/src/ATen/cpu/vec/vec_base.h').exists()
                     and shutil.which('g++'), 'public ATen headers and host C++ compiler required')
class AtenPowReference(unittest.TestCase):
    def test_native_vector_boundaries(self):
        with tempfile.TemporaryDirectory() as folder:
            work = Path(folder)
            (work / 'check.cpp').write_text(SOURCE)
            result = subprocess.run(['cc', '-O0', '-ffp-contract=off',
                '-DNETLIB_SPEC=reference_pow', '-c',
                str(ROOT / 'generic_solver/lib_v3/ref_c/aten_pow_uniform_base.c'),
                '-o', str(work / 'reference.o')], capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            includes = ['generic_solver/lib_v3/support', 'third_party/pytorch',
                        'third_party/pytorch/aten/src']
            result = subprocess.run(['g++', '-O0', '-std=c++17', '-ffp-contract=off',
                *['-I' + str(ROOT / p) for p in includes], str(work / 'check.cpp'),
                str(work / 'reference.o'), '-o', str(work / 'check')],
                capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run([str(work / 'check')], capture_output=True,
                                    text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('121 native ATen configurations passed', result.stdout)


if __name__ == '__main__':
    unittest.main()
