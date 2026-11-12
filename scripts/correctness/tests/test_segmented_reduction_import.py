"""Whole row reductions are recognized from imported CUB reference bodies."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'scripts/correctness'), str(ROOT/'scripts/correctness/lib_v2')]
import match_v2 as M
import recipe


class SegmentedReductionImport(unittest.TestCase):
    def test_fp64_rows_bind_as_one_call_without_precision_conversion(self):
        for element, expected in [('double', True), ('float', False)]:
            with self.subTest(element=element), tempfile.TemporaryDirectory() as tmp:
                d = Path(tmp)
                c = d/'reference.c'
                c.write_text(f'''void unrelated({element} input[3][7], {element} output[3]) {{
  for (int r=0; r<3; ++r) output[r]=0.25;
  for (int r=0; r<3; ++r)
    for (int c=0; c<7; ++c) output[r]+=input[r][c];
}}
''')
                raised = recipe.raise_c(recipe.PRODUCTION, str(c), 'unrelated',
                                        str(d/'raised'), [], 60)
                self.assertTrue(all(raised[k].ok for k in ['cgeist','raise','debuf']))
                with patch.dict(os.environ, POLYGEIST_NUMERIC_MODEL='strict'):
                    output, matches, _ = M.run(Path(raised['paths']['debuf']).read_text(),
                        ROOT/'generic_solver/kernel_library_v3.mlir', region=False,
                        allowed_specs={'cub_segmented_sum_f64'})
                self.assertEqual(bool(matches), expected)
                if expected:
                    self.assertEqual([r['spec'] for r in matches], ['cub_segmented_sum_f64'])
                    self.assertNotIn('linalg.generic', output)
                    self.assertNotIn('scf.for', output)

    def test_renamed_shapes_and_nonboolean_seed(self):
        for rows, cols, operator, spec in [
            (7, 11, '&&', 'cub_segmented_reduce_and_i32'),
            (3, 19, '||', 'cub_segmented_reduce_or_i32'),
            (5, 13, '^', 'cub_segmented_xor_i32'),
        ]:
            with self.subTest(rows=rows, cols=cols, operator=operator), tempfile.TemporaryDirectory() as tmp:
                d = Path(tmp)
                c = d/'reference.c'
                c.write_text(f'''void unrelated(int input[{rows}][{cols}], int output[{rows}]) {{
  for (int r = 0; r < {rows}; ++r) {{
    int result = -7;
    for (int c = 0; c < {cols}; ++c) result = result {operator} input[r][c];
    output[r] = result;
  }}
}}
''')
                raised = recipe.raise_c(recipe.PRODUCTION, str(c), 'unrelated', str(d/'raised'), [], 60)
                self.assertTrue(all(raised[k].ok for k in ['cgeist','raise','debuf']))
                source = Path(raised['paths']['debuf']).read_text()
                with patch.dict(os.environ, {'POLYGEIST_NUMERIC_MODEL':'strict'}):
                    output, matches, _ = M.run(source, ROOT/'generic_solver/kernel_library_v3.mlir',
                        region=False, allowed_specs={spec})
                self.assertEqual([r['spec'] for r in matches], [spec])
                self.assertNotIn('linalg.generic', output)
                self.assertNotIn('scf.for', output)


if __name__ == '__main__':
    unittest.main()
