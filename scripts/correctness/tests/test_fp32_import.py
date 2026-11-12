"""Generic fixes exercised by the FP32 reference import (no vendor execution)."""
import os
import re
import tempfile
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(HERE / 'lib_v2')]
import kernel_match as K
import legalize_tensor_submaps as L
import match_v2 as M
import mlir_lite as ML
import recipe as R


def comparison(predicate, reverse=False):
    a, b = ('%out', '%in') if reverse else ('%in', '%out')
    return K.GenericBody(ins_arg_names=['%in'], outs_arg_names=['%out'],
                         body_lines=[f'%v = arith.cmpf {predicate}, {a}, {b} : f32'],
                         yield_values=['%v'], captures=[], indexing_maps=[],
                         iterator_types=['parallel'], constants={},
                         block_arg_types=['f32', 'f32'])


MULTI = '''module {
  func.func @test(%x: tensor<?xf32>, %y: tensor<?xf32>, %sum: tensor<f32>, %n: index) {
    %v = polygeist.submap(%sum, %n) {map = affine_map<(d0) -> ()>} : (tensor<f32>, index) -> tensor<?xf32>
    %r:2 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["reduction"]} ins(%x : tensor<?xf32>) outs(%y, %v : tensor<?xf32>, tensor<?xf32>) {
    ^bb0(%in: f32, %out: f32, %acc: f32):
      %square = arith.mulf %in, %in : f32
      %total = arith.addf %acc, %square : f32
      linalg.yield %square, %total : f32, f32
    } -> (tensor<?xf32>, tensor<?xf32>)
    %w = polygeist.submapInverse(%sum, %r#1, %n) {map = affine_map<(d0) -> ()>} : (tensor<f32>, tensor<?xf32>, index) -> tensor<f32>
    return
  }
}
'''


class GenericImportFixes(unittest.TestCase):
    def test_comparison_reversal_is_exact_in_strict_mode(self):
        with patch.dict(os.environ, POLYGEIST_NUMERIC_MODEL='strict'):
            a = K.encode_body(comparison('ogt'))
            self.assertTrue(K.equivalent(a, K.encode_body(comparison('olt', True))))
            self.assertFalse(K.equivalent(a, K.encode_body(comparison('ugt'))))
            self.assertFalse(K.equivalent(a, K.encode_body(comparison('oge'))))

    def test_split_independent_mixed_outputs(self):
        text, count = L.split_independent_outputs(MULTI)
        self.assertEqual(count, 1)
        ops = [g for g in ML.Module(text).ops if g.name == 'linalg.generic']
        self.assertEqual([g.iters for g in ops], [['parallel'], ['reduction']])
        self.assertEqual([len(g.outs) for g in ops], [1, 1])
        self.assertNotIn('%acc', ops[0].text)

    def test_cross_output_recurrence_cannot_split(self):
        text = MULTI.replace('%in, %in', '%in, %acc')
        self.assertEqual(L.split_independent_outputs(text), (text, 0))

    def test_effectful_body_cannot_split(self):
        text = MULTI.replace('arith.mulf %in, %in : f32',
                             'func.call @side_effect(%in) : (f32) -> f32')
        self.assertEqual(L.split_independent_outputs(text), (text, 0))




if __name__ == '__main__':
    unittest.main()
