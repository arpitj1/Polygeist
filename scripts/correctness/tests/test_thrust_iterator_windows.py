"""Public Thrust traversal references preserve prefixes, gaps, and tails."""
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
#include <thrust/reverse.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <cuda/std/functional>
#include <cuda/std/iterator>
#include <vector>
#include <cstring>
#include <cstdio>
extern "C" void reference_copy(int,int,float*,float*);
extern "C" void reference_negate_copy(int,int,float*,float*);
extern "C" void reference_reverse(int,int,int,float*);
extern "C" void reference_negate(int,int,int,float*);
int main() {
  int checked=0;
  for(int n : {1,7,32}) for(int prefix : {0,1,3}) for(int gap : {0,2,9}) {
    std::vector<float> input(n), actual(prefix+2*n+gap+5), expected;
    for(int i=0;i<n;++i) input[i]=(i%2 ? -1.f : 1.f)*(i+0.25f);
    for(int i=0;i<(int)actual.size();++i) actual[i]=i+0.125f;
    input[0]=-0.f;
    expected=actual;
    reference_copy(n,prefix,input.data(),expected.data());
    thrust::copy(thrust::seq,input.begin(),input.end(),actual.begin()+prefix);
    if(std::memcmp(actual.data(),expected.data(),actual.size()*sizeof(float)))return 1;
    ++checked;
    reference_negate_copy(n,prefix,input.data(),expected.data());
    thrust::transform(thrust::seq,input.begin(),input.end(),actual.begin()+prefix,
                      cuda::std::negate<float>{});
    if(std::memcmp(actual.data(),expected.data(),actual.size()*sizeof(float)))return 4;
    ++checked;
    for(int negate : {0,1}) {
      for(int i=0;i<(int)actual.size();++i)actual[i]=(i%2 ? -1.f : 1.f)*(i+0.125f);
      // Signed zero must also survive the reverse and negate operations.
      actual[prefix]=-0.f;
      expected=actual;
      (negate ? reference_negate : reference_reverse)(n,prefix,gap,expected.data());
      auto first=actual.data()+prefix, last=first+n;
      if(negate)thrust::transform(thrust::seq,cuda::std::make_reverse_iterator(last),
          cuda::std::make_reverse_iterator(first),last+gap,cuda::std::negate<float>{});
      else thrust::reverse_copy(thrust::seq,first,last,last+gap);
      if(std::memcmp(actual.data(),expected.data(),actual.size()*sizeof(float)))return 2+negate;
      ++checked;
    }
  }
  std::printf("%d iterator window configurations passed\n",checked);
}
'''

@unittest.skipUnless(shutil.which('g++') and (INCLUDE/'thrust/reverse.h').exists(),
                     'public Thrust headers and host C++ compiler required')
class ThrustIteratorWindows(unittest.TestCase):
    def test_native_copy_and_disjoint_reverse_windows(self):
        with tempfile.TemporaryDirectory() as folder:
            work=Path(folder); objects=[]
            for name, source, flags in [
                ('reference_copy','thrust_copy_window.c',['-DNEGATE=0']),
                ('reference_negate_copy','thrust_copy_window.c',['-DNEGATE=1']),
                ('reference_reverse','thrust_reverse_window.c',['-DNEGATE=0']),
                ('reference_negate','thrust_reverse_window.c',['-DNEGATE=1']),
            ]:
                obj=work/(name+'.o'); objects.append(str(obj))
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
            self.assertIn('108 iterator window configurations passed',result.stdout)

if __name__=='__main__': unittest.main()
