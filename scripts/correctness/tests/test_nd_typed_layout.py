"""Execute the typed ABI's shape/stride guards with real memref descriptors."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'scripts/correctness'), str(ROOT/'scripts/correctness/lib_v2')]
import abi_v2 as A
import reference_execution as E
import lower_v2 as L


class NdTypedLayout(unittest.TestCase):
    def test_padded_matrix_is_packed_and_written_back(self):
        arguments = [A.SpecArg(n, 'int', ctype='int') for n in ['m', 'n']]
        arguments.append(A.SpecArg('x', 'array', rank=2, intent='inout', ctype='double'))
        params = [A.AbiParam(n, 'int') for n in ['m', 'n']]
        params += [A.AbiParam('x', 'array', elem='double', rank=2, layout='x_stride', written=True),
                   A.AbiParam('x_stride', 'int', layout_of='x')]
        binding = {n: ('arg', i) for i, n in enumerate(['m', 'n', 'x'])}
        binding['x_stride'] = ('const_num', 1)
        ca = A.CallAbi('TestMatrix', 'Test', 'TestMatrix', params, [], backends=['cuda'],
                       runtime_adapter={'kind': 'typed_c_wrapper', 'arrays': {'x': ['m', 'n']}})
        sa = A.SpecAbi('matrix_reference', 'TestMatrix', 'void', arguments, binding, {})
        text = '''module {
func.func @driver(%m:i32,%n:i32,%x:memref<?x?xf64, strided<[?, ?], offset: ?>>) {
  kernel.launch @matrix_reference(%m,%n,%x) : (i32,i32,memref<?x?xf64, strided<[?, ?], offset: ?>>) -> ()
  return
}}
'''
        with patch.object(A, 'build_table', return_value=({'TestMatrix': ca}, {'matrix_reference': sa}, {})):
            lowered = L.lower(text, do_bufferize=False, backend='cuda')
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d/'input.mlir').write_text(lowered)
            subprocess.run([E.MLIR_OPT, *E.LOWER_TO_LLVM, '--convert-cf-to-llvm', str(d/'input.mlir'), '-o', str(d/'llvm.mlir')], check=True, capture_output=True)
            subprocess.run([E.MLIR_TRANSLATE, '--mlir-to-llvmir', str(d/'llvm.mlir'), '-o', str(d/'code.ll')], check=True, capture_output=True)
            (d/'test.c').write_text('''#include "polygeist_v2_matrix_layout.inc"
static int called;
void polygeist_v2_TestMatrix(int m,int n,double*x,int64_t bytes,int stride){
 if(!x||bytes!=m*n*8||stride!=n)exit(9);called++;
 for(int r=0;r<m;++r)for(int c=0;c<n;++c)x[r*n+c]+=1;
}
void driver(int,int,double*,double*,int64_t,int64_t,int64_t,int64_t,int64_t);
int main(int argc,char**argv){double x[40];int mode=argc>1?atoi(argv[1]):0;
 int m=4,n=mode==3?1:3,s0=mode==0?3:mode==2?1:7,s1=mode>=2?5:1;
 for(int i=0;i<40;++i)x[i]=i*0.25;
 driver(m,n,x,x,0,m,n,s0,s1);
 for(int i=0;i<40;++i){double expected=i*0.25;
  for(int r=0;r<m;++r)for(int c=0;c<n;++c)if(i==r*s0+c*s1)expected+=1;
  if(x[i]!=expected)return 8;
 }
 return called!=1;
}''')
            subprocess.run([E.CLANG, '-I'+str(ROOT/'runtime'), str(d/'test.c'), str(d/'code.ll'), '-o', str(d/'test')], check=True, capture_output=True)
            for mode in range(4):
                result = subprocess.run([str(d/'test'), str(mode)], capture_output=True)
                self.assertEqual(result.returncode, 0, (mode, result.stderr))

    def test_runtime_extent_and_stride_guards(self):
        arguments=[A.SpecArg(n,'int',ctype='int') for n in ['n','h','w']]
        arguments.append(A.SpecArg('x','array',rank=3,intent='in',ctype='float'))
        params=[A.AbiParam(n,'int') for n in ['n','h','w']]
        params += [A.AbiParam('x','array',elem='float',rank=3,layout='x_stride',const=True),
                   A.AbiParam('x_stride','int',layout_of='x')]
        binding={n:('arg',i) for i,n in enumerate(['n','h','w','x'])}
        binding['x_stride']=('const_num',1)
        ca=A.CallAbi('TestReference','Test','TestReference',params,[],backends=['cuda'],
            runtime_adapter={'kind':'typed_c_wrapper','arrays':{'x':['n','h','w']}})
        sa=A.SpecAbi('nd_reference','TestReference','void',arguments,binding,{})
        text='''module {
func.func @driver(%n:i32,%h:i32,%w:i32,%x:memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>) {
  kernel.launch @nd_reference(%n,%h,%w,%x) : (i32,i32,i32,memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>) -> ()
  return
}}
'''
        with patch.object(A,'build_table',return_value=({'TestReference':ca},{'nd_reference':sa},{})):
            lowered=L.lower(text,do_bufferize=False,backend='cuda')
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);(d/'input.mlir').write_text(lowered)
            subprocess.run([E.MLIR_OPT,*E.LOWER_TO_LLVM,'--convert-cf-to-llvm',str(d/'input.mlir'),'-o',str(d/'llvm.mlir')],check=True,capture_output=True)
            subprocess.run([E.MLIR_TRANSLATE,'--mlir-to-llvmir',str(d/'llvm.mlir'),'-o',str(d/'code.ll')],check=True,capture_output=True)
            (d/'test.c').write_text('''#include <stdint.h>
#include <stdlib.h>
static int called;
void polygeist_v2_TestReference(int n,int h,int w,float*x,int64_t bytes,int flag){
 if(!x||bytes!=n*h*w*4||flag!=1)exit(9);called++;
}
void driver(int,int,int,float*,float*,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);
int main(int argc,char**argv){float x[60]={0};int mode=argc>1?atoi(argv[1]):0;
 int n=mode==3?1:2,h=3,w=5;
 driver(n,h,w,x,x,0,n,mode==1?2:h,w,mode==3?999:15,mode==2?6:5,1);
 return called!=1;
}''')
            subprocess.run([E.CLANG,str(d/'test.c'),str(d/'code.ll'),'-o',str(d/'test')],check=True,capture_output=True)
            for mode,valid in [(0,True),(1,False),(2,False),(3,True)]:
                run=subprocess.run([str(d/'test'),str(mode)],capture_output=True)
                self.assertEqual(run.returncode==0,valid,(mode,run.stderr))


if __name__=='__main__':unittest.main()
