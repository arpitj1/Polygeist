"""Execute pinned CUB sequential/seeded source bodies and CCCL minus<int>.
Only host/device annotations and reduction dispatch are adapted for the host;
the sequential loop and seed-prefix array construction come from the header.
Parallel DeviceSegmentedReduce execution is checked separately on the GPU.
"""
import argparse
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
source=(ROOT/'third_party/cuda_headers/cuda_cccl/include/cccl/cub/thread/thread_reduce.cuh').read_text()
def body(marker):
 start=source.index('{',source.rindex(marker));depth=1;end=start+1
 while depth:
  depth += (source[end]=='{')-(source[end]=='}');end+=1
 return source[start+1:end-1]
sequential=body('AccumT ThreadReduceSequential(const Input& input, ReductionOp reduction_op)')
seeded=body('ThreadReduce(const Input& input, ReductionOp reduction_op, PrefixT prefix)')
# Type/range static assertions are satisfied by this specialization's int arrays.
seeded=seeded[seeded.index('  constexpr int length'):]
header='''#include <type_traits>
#include <cuda/std/functional>
#undef _CCCL_PRAGMA_UNROLL_FULL
#define _CCCL_PRAGMA_UNROLL_FULL()
namespace cub {
namespace detail {
template<class Input> constexpr int static_size_v=std::extent<Input>::value;
template<class AccumT,class Input,class ReductionOp>
AccumT ThreadReduceSequential(const Input& input,ReductionOp reduction_op){ SEQUENTIAL }
}
template<class Input,class ReductionOp,class ValueT,class AccumT>
AccumT ThreadReduce(const Input& input,ReductionOp reduction_op){
 return detail::ThreadReduceSequential<AccumT>(input,reduction_op); }
template<class Input,class ReductionOp,class PrefixT>
int ThreadReduce(const Input& input,ReductionOp reduction_op,PrefixT prefix){
 using namespace detail;using AccumT=int; SEEDED }
}
'''.replace('SEQUENTIAL',sequential).replace('SEEDED',seeded)
main='SOURCE_BODY\n#include <cuda/std/functional>\n#include <cstdint>\n#include <cstdio>\n#include <vector>\nextern "C" void reference(int,int,int,int8_t*,int*);\nextern "C" void subtract_reference(int,int*,int,int*);\ntemplate<int Rows> long test(){long checked=0;\n for(int cols:{1,3,48})for(int seed:{0,-37,1073741824,-1073741824})for(int mode=0;mode<3;++mode){\n  std::vector<int8_t>x(Rows*cols);std::vector<int>a(cols+8,913),b=a,c=a,d=a;\n  const int edge[]={0,1,-1,-128,127,2,-2,63};\n  for(int i=0;i<Rows*cols;++i)x[i]=mode==2?0:mode==1?edge[i%8]:i%41-20;\n  auto original=x;reference(Rows,cols,seed,x.data(),a.data()+4);\n  for(int col=0;col<cols;++col){int values[Rows];for(int r=0;r<Rows;++r)values[r]=x[r*cols+col];\n    b[col+4]=cub::ThreadReduce(values,cuda::std::plus<int>{},seed);}\n  if(a!=b||x!=original)return -1;\n  for(int offset:{-97,0,113}){\n   subtract_reference(cols,a.data()+4,offset,c.data()+4);\n   for(int col=0;col<cols;++col)d[col+4]=cuda::std::minus<int>{}(b[col+4],offset);\n   if(c!=d)return -2;checked+=c.size();\n  }\n  checked+=a.size();\n }\n return checked;\n}\nint main(){long n=0;for(long x:{test<1>(),test<7>(),test<65>(),test<127>()}){if(x<0)return 1;n+=x;}\n printf("PASS %ld actual CUB/CCCL integer source outputs and guards\\n",n);}\n'
(out/'check.cpp').write_text(main.replace('SOURCE_BODY',header))
for routine,entry,flags in [('cub_column_sum','reference',[]),('thrust_arithmetic','subtract_reference',
 ['-DSCALAR=int','-DLEFT_SCALAR=0','-DRIGHT_SCALAR=1','-DTHRUST_OPERATION=1'])]:
 subprocess.run(['gcc','-O0','-DNETLIB_SPEC='+entry,*flags,'-c',
 str(ROOT/'generic_solver/lib_v3/ref_c'/ (routine+'.c')),'-o',str(out/(entry+'.o'))],check=True)
subprocess.run(['g++','-O0','-std=c++17','-I'+str(ROOT/'third_party/cuda_headers/cuda_cccl/include/cccl'),
 str(out/'check.cpp'),str(out/'reference.o'),str(out/'subtract_reference.o'),'-o',str(out/'check')],check=True)
subprocess.run([str(out/'check')],check=True)
