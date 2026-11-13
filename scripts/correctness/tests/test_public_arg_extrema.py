"""Check the index-order specialization against the original CUB functors."""
import ctypes
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'scripts/correctness/lib_v2'),str(ROOT/'scripts/correctness')]
import build_lib_v2 as B
import recipe as R
import validate_specs as V
import match_v2 as M


def definition(text, name):
    start = text.index('struct '+name+'\n')
    end = text.index('{',start)+1
    depth = 1
    while depth:
        depth += (text[end]=='{')-(text[end]=='}')
        end += 1
    return text[start:end]+';'


class PublicArgExtrema(unittest.TestCase):
    def test_index_coordinates_and_offsets_require_structural_proof(self):
        for name in ('cub_segmented_argmax_f32', 'cub_segmented_argmin_f32'):
            source = (ROOT/'generic_solver/lib_v3/raised'/
                      (name+'_debuf.mlir')).read_text().replace('@'+name, '@sample')
            variants = {
                'same': source,
                'wrong-coordinate': source.replace('linalg.index 1', 'linalg.index 0'),
                'wrong-offset': source.replace('(d0 + 1)', '(d0 + 2)'),
            }
            for label, text in variants.items():
                with self.subTest(spec=name, variant=label), patch.dict(
                        os.environ, POLYGEIST_NUMERIC_MODEL='strict'):
                    _, matches, _ = M.run(text, M.DEFAULT_LIB, region=False,
                                          allowed_specs={name})
                    self.assertEqual(len(matches), int(label == 'same'))

    def test_coupled_value_and_index_reductions_execute_exactly(self):
        for spec in B.load_specs():
            if spec.routine != 'cub_segmented_arg_extrema':
                continue
            with tempfile.TemporaryDirectory(dir=ROOT/'build') as tmp:
                root=Path(tmp);src=root/'reference.c'
                src.write_text('#define NETLIB_SPEC '+spec.name+'\n'+
                    ''.join(f'#define {k} {v}\n' for k,v in spec.defines.items())+
                    '#include "'+str(spec.ref_c)+'"\n')
                stages=R.raise_c(R.PRODUCTION,str(src),spec.name,str(root/spec.name),[],120)
                self.assertTrue(stages['debuf'].ok,stages['debuf'].stderr)
                with patch.object(V,'RAISED',root),patch.object(V,'WORK',root/'validate'):
                    for cols in (1,7,64):
                        manifest=dict(spec.raw,validation=dict(scalars=dict(rows=3,cols=cols),
                            arrays=dict(input=[3,cols],output=[3])))
                        result=V.validate(spec.name,manifest,spec.ref_c,{'cols'})
                        self.assertEqual(result['verdict'],'EXACT',(spec.name,cols,result))

    def test_first_tie_index_matches_public_functors(self):
        upstream = (ROOT/'third_party/cuda_headers/cuda_cccl/include/cccl/cub/thread/thread_operators.cuh').read_text()
        prefix = '''#define _CCCL_HOST_DEVICE
#define _CCCL_FORCEINLINE inline
template<class K,class V> struct KeyValuePair {K key;V value;};
'''
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for minimum in (False,True):
                op = 'ArgMin' if minimum else 'ArgMax'
                source = prefix+definition(upstream,op)+'''
extern "C" void original(int rows,int cols,const float *x,int *out) {
 for(int r=0;r<rows;++r) {
  KeyValuePair<int,float> best{0,x[r*cols]};
  for(int c=1;c<cols;++c)best=OP{}(best,{c,x[r*cols+c]});
  out[r]=best.key;
 }
}
'''.replace('OP',op)
                (root/'upstream.cpp').write_text(source)
                upstream_so=root/(op+'_upstream.so')
                adapted_so=root/(op+'_adapted.so')
                subprocess.run(['g++','-O2','-shared','-fPIC',str(root/'upstream.cpp'),'-o',str(upstream_so)],check=True,capture_output=True)
                subprocess.run(['gcc','-O2','-shared','-fPIC','-DNETLIB_SPEC=adapted',
                    '-DARG_MIN='+str(int(minimum)),str(ROOT/'generic_solver/lib_v3/ref_c/cub_segmented_arg_extrema.c'),
                    '-o',str(adapted_so)],check=True,capture_output=True)
                original = ctypes.CDLL(str(upstream_so)).original
                adapted = ctypes.CDLL(str(adapted_so)).adapted
                for cols in (1,2,7,64,257):
                    rows=7
                    values=[float((i*37)%19-9) for i in range(rows*cols)]
                    values[:cols]=[2.0]*cols
                    values[cols:2*cols]=[-0.0 if i%2 else 0.0 for i in range(cols)]
                    x=(ctypes.c_float*len(values))(*values)
                    expected=(ctypes.c_int*rows)();actual=(ctypes.c_int*rows)()
                    original(rows,cols,x,expected);adapted(rows,cols,x,actual)
                    self.assertEqual(list(actual),list(expected),(op,cols))
                    self.assertEqual(actual[0],0)
                    self.assertEqual(actual[1],0)


if __name__=='__main__':
    unittest.main()
