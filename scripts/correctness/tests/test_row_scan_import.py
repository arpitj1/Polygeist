"""Prove whole seeded row scans, with a negative arithmetic case."""
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


class RowScanImport(unittest.TestCase):
    def test_seeded_product_and_negative_sum(self):
        for rows, cols, operator in [(3, 7, '*'), (5, 13, '*'), (3, 7, '+')]:
            with self.subTest(rows=rows, cols=cols, op=operator), tempfile.TemporaryDirectory() as tmp:
                d = Path(tmp)
                c = d/'reference.c'
                c.write_text(f'''void unrelated(float input[{rows}][{cols}], float output[{rows}][{cols}]) {{
  for (int r=0; r<{rows}; ++r) {{
    float value=0.5f;
    for (int c=0; c<{cols}; ++c) {{
      value=value {operator} input[r][c];
      output[r][c]=value;
    }}
  }}
}}
''')
                raised = recipe.raise_c(recipe.PRODUCTION, str(c), 'unrelated',
                                        str(d/'raised'), [], 60)
                self.assertTrue(all(raised[k].ok for k in ['cgeist','raise','debuf']))
                with patch.dict(os.environ, POLYGEIST_NUMERIC_MODEL='strict'):
                    output, matches, _ = M.run(Path(raised['paths']['debuf']).read_text(),
                        ROOT/'generic_solver/kernel_library_v3.mlir', region=False,
                        allowed_specs={'cub_row_product_scan_f32'})
                self.assertEqual(bool(matches), operator == '*')
                if matches:
                    self.assertEqual([r['spec'] for r in matches], ['cub_row_product_scan_f32'])
                    self.assertNotIn('linalg.generic', output)
                    self.assertNotIn('scf.for', output)


if __name__ == '__main__':
    unittest.main()
