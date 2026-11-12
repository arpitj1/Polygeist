import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'scripts/correctness'))
import aten_fixture_batch as B
import aten_paper_shapes as S


class PaperShapes(unittest.TestCase):
    def test_copy_preserves_body_and_original(self):
        original = ROOT/'issues/aten_c_kernels/aten_conv2d.c'
        old = original.read_bytes()
        row = dict(kernel='aten_conv2d',dtype='f32',
                   dims=dict(B=4,IC=16,OC=32,H=128,W=128,KH=3,KW=3))
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'copy.c'
            S.materialize(original, output, row)
            self.assertEqual(original.read_bytes(), old)
            self.assertEqual(output.read_text().split('void ',1)[1],old.decode().split('void ',1)[1])
            result = subprocess.run(['gcc','-E','-P',str(output)],capture_output=True,text=True,check=True)
            self.assertIn('float input[4][16][128][128]',result.stdout)
            self.assertIn('float filter[32][16][3][3]',result.stdout)

    def test_refuse_mutation_or_unknown_dimensions(self):
        source = ROOT/'issues/aten_c_kernels/aten_conv2d.c'
        with self.assertRaises(ValueError):
            S.materialize(source,source,dict(dims={'B':4}))
        with tempfile.TemporaryDirectory() as tmp:
            for dims in [{'UNKNOWN':4},{'B':True},{'B':0}]:
                with self.assertRaises(ValueError):
                    S.materialize(source,Path(tmp)/'copy.c',dict(dims=dims))

    def test_dtype_and_metadata_preserved(self):
        cfg=B.configuration('aten_nested_sum_dim_cpu',
            dict(kernel='aten_nested_sum_dim_cpu',dims={'B':20,'N':128},dtype='f32'))
        self.assertEqual(cfg['integer_array_values']['len'][:4],[-1,0,1,128])
        with self.assertRaises(ValueError):
            B.configuration('aten_mm',dict(kernel='aten_mm',dims={'M':512},dtype='f32'))

    def test_samples_do_not_change_default(self):
        cfg=B.configuration('aten_polar_scalarized')
        self.assertIn('#define TIMING_SAMPLES 5',B.harness('aten_polar_scalarized',cfg))
        cfg['timing_samples']=20
        self.assertIn('#define TIMING_SAMPLES 20',B.harness('aten_polar_scalarized',cfg))


if __name__=='__main__':
    unittest.main()
