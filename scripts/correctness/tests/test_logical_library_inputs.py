"""Imported functions may consume tensor views without losing their offsets."""
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'scripts/correctness/lib_v2')]
import match_v2 as M


def source(element):
    return f'''module {{
 func.func @renamed_window_difference(%x: tensor<18x{element}>, %out: tensor<17x{element}>) -> tensor<17x{element}> {{
  %n = arith.constant 17 : index
  %right = polygeist.submap(%x,%n) {{map=affine_map<(i)->(i+1)>}} : (tensor<18x{element}>,index) -> tensor<17x{element}>
  %left = polygeist.submap(%x,%n) {{map=affine_map<(i)->(i)>}} : (tensor<18x{element}>,index) -> tensor<17x{element}>
  %dest = polygeist.submap(%out,%n) {{map=affine_map<(i)->(i)>}} : (tensor<17x{element}>,index) -> tensor<17x{element}>
  %r = linalg.generic {{indexing_maps=[affine_map<(i)->(i)>,affine_map<(i)->(i)>,affine_map<(i)->(i)>],iterator_types=["parallel"]}} ins(%right,%left:tensor<17x{element}>,tensor<17x{element}>) outs(%dest:tensor<17x{element}>) {{
  ^bb0(%a:{element},%b:{element},%old:{element}):
   %v = arith.subf %a,%b : {element}
   linalg.yield %v : {element}
  }} -> tensor<17x{element}>
  %updated = polygeist.submapInverse(%out,%r,%n) {{map=affine_map<(i)->(i)>}} : (tensor<17x{element}>,tensor<17x{element}>,index) -> tensor<17x{element}>
  return %updated : tensor<17x{element}>
 }}
}}'''


class LogicalLibraryInputs(unittest.TestCase):
    def match(self, element):
        text = subprocess.check_output([str(ROOT/'build/bin/polygeist-opt')],
                                       input=source(element), text=True)
        with patch.dict(os.environ, {'POLYGEIST_NUMERIC_MODEL':'strict'}):
            return M.run(text, ROOT/'generic_solver/kernel_library_v3.mlir',
                         region=False, allowed_specs={'thrust_minus_f32'})

    def test_shifted_read_remains_a_distinct_library_operand(self):
        output, report, _ = self.match('f32')
        self.assertEqual([r['spec'] for r in report], ['thrust_minus_f32'])
        self.assertNotIn('linalg.generic', output)
        self.assertIn('d0 + 1', output)
        # The shifted view stays live; matching must not bind both formals to x.
        self.assertNotEqual(report[0]['binding']['%x'], report[0]['binding']['%y'])

    def test_logical_input_does_not_change_precision(self):
        self.assertFalse(self.match('f64')[1])


if __name__ == '__main__':
    unittest.main()
