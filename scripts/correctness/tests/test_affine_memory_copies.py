"""Exercise composed memory addresses independently of library recognition."""
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
loader = importlib.util.spec_from_file_location('lower_affine_copies', ROOT/'scripts/correctness/lib_v2/lower_v2.py')
L = importlib.util.module_from_spec(loader)
loader.loader.exec_module(L)
TYPE = 'memref<?x?xi32, strided<[?, ?], offset: ?>>'
TEXT = f'''module {{
 func.func @movement(%a: {TYPE}, %b: {TYPE}) attributes {{llvm.emit_c_interface}} {{
  linalg.generic {{indexing_maps=[affine_map<(i,j)->(2*i,j+1)>,affine_map<(i,j)->(i,j)>],iterator_types=["parallel","parallel"]}} ins(%a:{TYPE}) outs(%b:{TYPE}) {{
  ^bb0(%x:i32,%old:i32):
   linalg.yield %x : i32
  }}
  return
 }}
}}'''


class AffineMemoryCopies(unittest.TestCase):
    def test_offsets_strides_negative_source_and_guards(self):
        with tempfile.TemporaryDirectory() as folder:
            d = Path(folder)
            source = subprocess.check_output([str(ROOT/'build/bin/polygeist-opt')], input=TEXT, text=True)
            output = L.rewrite_memory_copies(source, 'cuda')
            self.assertNotIn('linalg.generic', output)
            self.assertIn('polygeist.gpu_data_operands = array<i64: 5, 6>', output)
            p = subprocess.run([str(ROOT/'llvm-project/build/bin/mlir-opt'),
                '--expand-strided-metadata','--lower-affine','--convert-arith-to-llvm',
                '--convert-index-to-llvm','--convert-cf-to-llvm','--finalize-memref-to-llvm',
                '--convert-func-to-llvm','--reconcile-unrealized-casts'],
                input=output,text=True,capture_output=True)
            self.assertEqual(p.returncode,0,p.stderr)
            ir = subprocess.check_output([str(ROOT/'llvm-project/build/bin/mlir-translate'),'--mlir-to-llvmir'],input=p.stdout,text=True)
            (d/'copy.ll').write_text(ir)
            (d/'main.c').write_text('''#include <stdint.h>
#include <string.h>
#include "polygeist_v2_matrix_layout.inc"
#include "polygeist_v2_memref_copy.inc"
typedef struct {uint32_t *allocated,*aligned;int64_t offset,sizes[2],strides[2];} Desc;
void _mlir_ciface_movement(Desc*,Desc*);
int main(){uint32_t a[96],b[96],want[96],saved[96];
for(int i=0;i<96;i++)a[i]=0x7f800001u+i*31;memcpy(saved,a,sizeof(a));
for(int reverse=0;reverse<2;reverse++){
for(int i=0;i<96;i++)b[i]=want[i]=0xdeadbeef;
Desc x={a,a,reverse?52:2,{6,7},{reverse?-10:10,1}},y={b,b,3,{3,4},{7,1}};
for(int i=0;i<3;i++)for(int j=0;j<4;j++)want[3+7*i+j]=a[x.offset+2*i*x.strides[0]+j+1];
_mlir_ciface_movement(&x,&y);if(memcmp(b,want,sizeof(b))||memcmp(a,saved,sizeof(a)))return 1;
}return 0;}
''')
            subprocess.run([str(ROOT/'llvm-project/build/bin/clang'),'-O2','-I'+str(ROOT/'runtime'),str(d/'main.c'),str(d/'copy.ll'),'-o',str(d/'check')],check=True,capture_output=True)
            subprocess.run([str(d/'check')],check=True)

    def test_computation_and_nonlinear_layout_are_not_memory_copies(self):
        arithmetic = TEXT.replace('linalg.yield %x : i32', '%sum = arith.addi %x,%old : i32\n   linalg.yield %sum : i32')
        nonlinear = TEXT.replace('(2*i,j+1)', '(i floordiv 2,j+1)')
        for text in (arithmetic, nonlinear):
            self.assertEqual(L.affine_memory_copies(text), text)


if __name__ == '__main__':
    unittest.main()
