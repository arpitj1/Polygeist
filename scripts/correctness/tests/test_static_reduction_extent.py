"""Match imported reductions after bounded intermediates become static tensors."""
import os
import sys
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'scripts/correctness'), str(ROOT/'scripts/correctness/lib_v2')]
import match_v2 as M


def reduce_input(size, element='f32'):
    return f'''module {{
  func.func @reduce(%input: tensor<{size}x{element}>) -> {element} {{
    %zero = arith.constant 0.0 : {element}
    %empty = tensor.empty() : tensor<{element}>
    %initial = tensor.insert %zero into %empty[] : tensor<{element}>
    %sum = linalg.generic {{indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->()>],iterator_types=["reduction"]}} ins(%input:tensor<{size}x{element}>) outs(%initial:tensor<{element}>) {{
    ^bb0(%x:{element},%old:{element}):
      %next = arith.addf %old,%x : {element}
      linalg.yield %next : {element}
    }} -> tensor<{element}>
    %value = tensor.extract %sum[] : tensor<{element}>
    return %value : {element}
  }}
}}
'''


class StaticReductionExtent(unittest.TestCase):
    def match(self, size, element='f32'):
        with patch.dict(os.environ, {'POLYGEIST_NUMERIC_MODEL':'strict'}):
            source = subprocess.check_output([str(ROOT/'build/bin/polygeist-opt')],
                       input=reduce_input(size,element), text=True)
            return M.run(source, ROOT/'generic_solver/kernel_library_v3.mlir',
                         region=False, allowed_specs={'cub_reduce_value_sum_f32'})

    def test_static_extent_binds_imported_runtime_extent(self):
        for size in (17,1024):
            output, report, _ = self.match(size)
            self.assertEqual([r['spec'] for r in report], ['cub_reduce_value_sum_f32'])
            self.assertNotIn('linalg.generic', output)
            self.assertIn(f'constant {size} : i32', output)

    def test_integer_extent_overflow_is_rejected(self):
        _, report, _ = self.match(2147483648)
        self.assertFalse(report)

    def test_input_precision_cannot_change(self):
        _, report, _ = self.match(17,'f64')
        self.assertFalse(report)


if __name__ == '__main__':
    unittest.main()
