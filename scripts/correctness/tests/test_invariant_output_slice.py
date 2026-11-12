"""Scalar reductions written into higher-rank slices retain exact coordinates."""
from pathlib import Path
import os
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'scripts/correctness'),str(ROOT/'scripts/correctness/lib_v2')]
import spec_match as S

TEXT='''module {
 func.func @unrelated(%input:tensor<5xf32>,%destination:tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<f32>
  %seed = tensor.insert %zero into %empty[] : tensor<f32>
  %sum = linalg.generic {indexing_maps=[affine_map<(i)->(i)>,affine_map<(i)->()>],iterator_types=["reduction"]} ins(%input:tensor<5xf32>) outs(%seed:tensor<f32>) {
  ^bb0(%value:f32,%old:f32):
   %next = arith.addf %old,%value : f32
   linalg.yield %next : f32
  } -> tensor<f32>
  %result = polygeist.submapInverse(%destination,%sum) {map=affine_map<()->(1,2,0)>} : (tensor<2x3x4xf32>,tensor<f32>) -> tensor<2x3x4xf32>
  return %result : tensor<2x3x4xf32>
 }
}'''


class InvariantOutputSlice(unittest.TestCase):
    def match(self,text):
        with patch.dict(os.environ,{'POLYGEIST_NUMERIC_MODEL':'strict'}):
            return S.match(text,backend='cuda',only=['cub_reduce_sum_f32'],
                           region=False,max_operand_rank=1,fuse_initializers=True)

    def test_extra_fixed_axes_preserve_parent_and_coordinates(self):
        output,proof=self.match(TEXT)
        self.assertEqual(proof['emitted'],['cub_reduce_sum_f32'])
        self.assertNotIn('linalg.generic',output)
        self.assertIn('tensor.insert_slice',output)
        self.assertIn('[1, 2, 0]',output)
        # Validate typed calls and rank-reducing views without requiring the
        # semantic kernel definitions to be embedded in the application.
        S.normalize_matched_application(output,None,lambda text:
            subprocess.check_output([str(ROOT/'build/bin/polygeist-opt')],
                                    input=text,text=True))

    def test_kept_coordinate_cannot_change(self):
        _,proof=self.match(TEXT.replace('()->(1,2,0)','()->(1,2,3)'))
        self.assertFalse(proof['emitted'])

    def test_element_type_cannot_change(self):
        _,proof=self.match(TEXT.replace('f32','f64'))
        self.assertFalse(proof['emitted'])


if __name__=='__main__':unittest.main()
