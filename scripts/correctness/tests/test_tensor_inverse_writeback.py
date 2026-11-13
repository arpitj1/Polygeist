"""Dynamic tiled tensor writebacks retain prefixes/tails and reject overlap."""
from copy import deepcopy
from itertools import product
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE / 'lib_v2')]
import legalize_tensor_submaps as L
import build_lib_v2 as B
import recipe as R
import validate_specs as V

ROOT = HERE.parents[1]
SOURCE = '''module {
 func.func @scatter(%n: i32, %offset: i32, %source: memref<?x2xf32>, %out: memref<?xf32>) {
  %N = arith.index_cast %n : i32 to index
  %O = arith.index_cast %offset : i32 to index
  %two = arith.constant 2 : index
  %view = bufferization.to_tensor %source : memref<?x2xf32>
  %base = bufferization.to_tensor %out : memref<?xf32>
  %result = polygeist.submapInverse(%base, %view, %O, %N, %two) {map = affine_map<(d0,d1)[s0] -> (2*d0+d1+s0)>} : (tensor<?xf32>, tensor<?x2xf32>, index, index, index) -> tensor<?xf32>
  %updated = bufferization.to_memref %result : memref<?xf32>
  memref.copy %updated, %out : memref<?xf32> to memref<?xf32>
  return
 }
}
'''


class TensorInverseWriteback(unittest.TestCase):
    def test_dynamic_tile_offset_and_odd_unwritten_tail(self):
        lowered, count = L.legalize(SOURCE)
        self.assertEqual(count, 1)
        self.assertNotIn('polygeist.submapInverse', lowered)
        self.assertEqual(lowered.count('scf.for'), 2)
        self.assertIn('iter_args', lowered)
        with tempfile.TemporaryDirectory(dir=ROOT / 'build') as directory:
            root = Path(directory)
            (root/'scatter_debuf.mlir').write_text(SOURCE)
            reference = root/'ref.c'
            reference.write_text('''void NETLIB_SPEC(int n,int offset,float source[n][2],float out[9]) {
              for(int i=0;i<n;++i) for(int j=0;j<2;++j)
                out[offset+2*i+j]=source[i][j];
            }''')
            manifest = dict(args=['n','offset','source','out'], validation=dict(
                scalars=dict(n=3,offset=1),arrays=dict(source=[3,2],out=[9])))
            with patch.object(V, 'RAISED', root), patch.object(V, 'WORK', root/'validate'):
                result = V.validate('scatter', manifest, reference, set())
            if result['verdict'] != 'EXACT':
                self.fail(str(result)+'\n'+(root/'validate/scatter/defn/log.txt').read_text()[-3500:])

    def test_dynamic_inner_extent_controls_its_stride(self):
        source = SOURCE.replace('%offset: i32,', '%offset: i32, %width: i32,')
        source = source.replace('memref<?x2xf32>', 'memref<?x?xf32>').replace('tensor<?x2xf32>', 'tensor<?x?xf32>')
        source = source.replace('%two = arith.constant 2 : index', '%two = arith.index_cast %width : i32 to index')
        source = source.replace('%view, %O, %N, %two)', '%view, %O, %two, %N, %two)')
        source = source.replace('(d0,d1)[s0] -> (2*d0+d1+s0)', '(d0,d1)[s0,s1] -> (d0*s1+d1+s0)')
        source = source.replace('index, index, index) -> tensor<?xf32>', 'index, index, index, index) -> tensor<?xf32>')
        lowered, count = L.legalize(source)
        self.assertEqual(count, 1)
        self.assertNotIn('polygeist.submapInverse', lowered)
        unrelated = source.replace('%view, %O, %two, %N, %two)', '%view, %O, %N, %N, %two)')
        self.assertEqual(L.legalize(unrelated), (unrelated, 0))
        with tempfile.TemporaryDirectory(dir=ROOT / 'build') as directory:
            root = Path(directory)
            (root/'scatter_debuf.mlir').write_text(source)
            reference = root/'ref.c'
            reference.write_text("""void NETLIB_SPEC(int n,int offset,int width,float source[n][width],float out[11]) {
              for(int i=0;i<n;++i)for(int j=0;j<width;++j)
                out[offset+width*i+j]=source[i][j];
            }""")
            manifest = dict(args=['n','offset','width','source','out'], validation=dict(
                scalars=dict(n=2,offset=1,width=3),arrays=dict(source=[2,3],out=[11])))
            with patch.object(V, 'RAISED', root), patch.object(V, 'WORK', root/'validate'):
                result = V.validate('scatter', manifest, reference, set())
            self.assertEqual(result['verdict'], 'EXACT', result)

    def test_overlapping_map_is_not_an_overwrite_reduction(self):
        overlapping = SOURCE.replace('2*d0+d1+s0', 'd0+d1+s0')
        self.assertEqual(L.legalize(overlapping), (overlapping, 0))

    def test_symbolic_stride_without_nonzero_contract_remains(self):
        unsupported = SOURCE.replace('2*d0+d1+s0', 'd0*s0+d1')
        self.assertEqual(L.legalize(unsupported), (unsupported, 0))

    def test_unknown_inner_extent_cannot_prove_tile_disjointness(self):
        unsupported = SOURCE.replace('%O, %N, %two)', '%O, %N, %N)')
        self.assertEqual(L.legalize(unsupported), (unsupported, 0))

    def test_tile_extent_exceeding_stride_remains(self):
        overlapping = SOURCE.replace('arith.constant 2 : index',
                                     'arith.constant 3 : index')
        self.assertEqual(L.legalize(overlapping), (overlapping, 0))

    def test_projected_nonunit_axis_remains(self):
        overlapping = SOURCE.replace('2*d0+d1+s0', '2*d0+s0')
        self.assertEqual(L.legalize(overlapping), (overlapping, 0))

    def test_public_pool_reference_all_odd_tail_combinations(self):
        """Execute the raised public reference, including untouched tail cells."""
        name = 'aten_avg_pool3d_backward_k2_f32'
        spec = next(s for s in B.load_specs() if s.name == name)
        with tempfile.TemporaryDirectory(dir=ROOT / 'build') as directory:
            root = Path(directory)
            source = root / 'reference.c'
            source.write_text('#define NETLIB_SPEC ' + name + '\n#include "' +
                              str(spec.ref_c) + '"\n')
            raised = R.raise_c(R.PRODUCTION, str(source), name,
                               str(root / name), [], timeout=120)
            for stage in ('cgeist', 'raise', 'debuf'):
                self.assertTrue(raised[stage].ok,
                                (stage, raised[stage].stderr[-2000:]))
            with patch.object(V, 'RAISED', root), patch.object(V, 'WORK', root/'validate'):
                for r0, r1, r2 in product((0, 1), repeat=3):
                    with self.subTest(remainders=(r0, r1, r2)):
                        manifest = deepcopy(spec.raw)
                        manifest['validation'] = dict(
                            scalars=dict(S=2, O0=2, O1=3, O2=4,
                                         R0=r0, R1=r1, R2=r2),
                            arrays=dict(grad_output=[2, 2, 3, 4],
                                        grad_input=[2, 4+r0, 6+r1, 8+r2]))
                        result = V.validate(name, manifest, spec.ref_c,
                                            set(B.vla_leading_dims(spec.ref_c)))
                        if result['verdict'] != 'EXACT':
                            self.fail(str(result)+'\n'+(root/'validate'/name/
                                      'defn/log.txt').read_text()[-3500:])


if __name__ == '__main__':
    unittest.main()
