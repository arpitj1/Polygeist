"""Boundary comparisons for public integer arithmetic and min/max imports."""
import json,shutil,subprocess,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
SOURCE='#define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_CPP\n#include <thrust/transform.h>\n#include <thrust/iterator/constant_iterator.h>\n#include <thrust/execution_policy.h>\n#include <cuda/std/functional>\n#include <cuda/functional>\n#include <cstdint>\n#include <vector>\n#include <cstdio>\nextern "C" void ref_plus(int,unsigned*,unsigned*,int*);\nextern "C" void ref_mul(int,unsigned*,unsigned,int*);\nextern "C" void ref_min(int,int*,int,int*);\nextern "C" void ref_max(int,int*,int,int*);\nextern "C" void ref_neg(int,unsigned*,int*);\nint main(){\n std::vector<uint32_t>x{0,1,2147483647u,2147483648u,4294967295u,17,4294967279u};\n std::vector<int32_t>sx(x.begin(),x.end()),out(11,99),expected=out;\n int checks=0,n=x.size();\n for(uint32_t b:x){\n  std::vector<uint32_t>y(n,b);\n  ref_plus(n,x.data(),y.data(),expected.data());\n  thrust::transform(thrust::seq,x.begin(),x.end(),y.begin(),out.begin(),cuda::std::plus<uint32_t>{});\n  if(out!=expected)return 1;checks+=n;\n  ref_mul(n,x.data(),b,expected.data());\n  thrust::transform(thrust::seq,x.begin(),x.end(),thrust::make_constant_iterator(b),out.begin(),cuda::std::multiplies<uint32_t>{});\n  if(out!=expected)return 2;checks+=n;\n  int bound=static_cast<int>(b);\n  ref_min(n,sx.data(),bound,expected.data());\n  thrust::transform(thrust::seq,sx.begin(),sx.end(),thrust::make_constant_iterator(bound),out.begin(),cuda::minimum<int32_t>{});\n  if(out!=expected)return 3;checks+=n;\n  ref_max(n,sx.data(),bound,expected.data());\n  thrust::transform(thrust::seq,sx.begin(),sx.end(),thrust::make_constant_iterator(bound),out.begin(),cuda::maximum<int32_t>{});\n  if(out!=expected)return 4;checks+=n;\n }\n ref_neg(n,x.data(),expected.data());\n thrust::transform(thrust::seq,x.begin(),x.end(),out.begin(),cuda::std::negate<uint32_t>{});\n if(out!=expected)return 5;checks+=n;\n std::printf("%d public integer boundary comparisons passed\\n",checks);\n}\n'

class IntegerCoordinateMath(unittest.TestCase):
 @unittest.skipUnless(shutil.which('g++') and shutil.which('cc'),'native compilers required')
 def test_native_unsigned_arithmetic_and_signed_bounds(self):
  names={'thrust_plus_u32_to_i32':'ref_plus','thrust_multiplies_scalar_right_u32_to_i32':'ref_mul','thrust_minimum_scalar_i32':'ref_min','thrust_maximum_scalar_i32':'ref_max','thrust_negate_u32_to_i32':'ref_neg'}
  manifest=json.loads((ROOT/'generic_solver/lib_v3/manifest/thrust_integer_coordinate_math.json').read_text())
  with tempfile.TemporaryDirectory() as folder:
   work=Path(folder);objects=[]
   for spec in manifest['specs']:
    obj=work/(spec['name']+'.o');objects.append(str(obj))
    flags=['-D'+k+'='+str(v) for k,v in spec['defines'].items()]
    p=subprocess.run(['cc','-O0','-DNETLIB_SPEC='+names[spec['name']],*flags,'-c',str(ROOT/'generic_solver/lib_v3/ref_c'/spec['ref_c']),'-o',str(obj)],capture_output=True,text=True,timeout=60)
    self.assertEqual(p.returncode,0,p.stderr)
   (work/'check.cpp').write_text(SOURCE)
   p=subprocess.run(['g++','-O0','-std=c++17','-I'+str(ROOT/'third_party/cuda_headers/cuda_cccl/include/cccl'),str(work/'check.cpp'),*objects,'-o',str(work/'check')],capture_output=True,text=True,timeout=120)
   self.assertEqual(p.returncode,0,p.stderr)
   p=subprocess.run([str(work/'check')],capture_output=True,text=True,timeout=30)
   self.assertEqual(p.returncode,0,p.stdout+p.stderr)
   self.assertIn('203 public integer boundary comparisons passed',p.stdout)
if __name__=='__main__':unittest.main()
