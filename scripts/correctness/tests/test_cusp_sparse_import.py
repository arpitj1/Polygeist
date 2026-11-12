"""Source-derived sparse references: renamed programs and semantic negatives."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'scripts/correctness'), str(ROOT/'scripts/correctness/lib_v2')]
import recipe
from region_match import match_regions


class CuspSparseImport(unittest.TestCase):
    def test_renamed_matvec_and_wrong_index_or_type(self):
        cases = [(3, 5, 'float', 'index[p]', True),
                 (7, 4, 'float', 'index[p]', True),
                 (3, 5, 'float', 'p', False),
                 (3, 5, 'double', 'index[p]', False)]
        for rows, cols, element, column, expected in cases:
            with self.subTest(rows=rows, cols=cols, element=element, column=column), tempfile.TemporaryDirectory() as tmp:
                d = Path(tmp)
                source = d/'arithmetic.c'
                source.write_text(f'''void unrelated(int offsets[{rows+1}], int index[11],
 {element} entries[11], {element} input[{cols}], {element} output[{rows}]) {{
  for (int r=0; r<{rows}; ++r) {{
    {element} value=0;
    for (int p=offsets[r]; p<offsets[r+1]; ++p)
      value+=entries[p]*input[{column}];
    output[r]=value;
  }}
}}''')
                raised = recipe.raise_c(recipe.PRODUCTION, str(source), 'unrelated',
                                        str(d/'raised'), [], 60)
                self.assertTrue(all(raised[k].ok for k in ('cgeist','raise','debuf')))
                with patch.dict(os.environ, POLYGEIST_NUMERIC_MODEL='relaxed'):
                    matches = match_regions(Path(raised['paths']['debuf']).read_text(),
                        str(ROOT/'generic_solver/lib_v3'),
                        allowed_specs={'cusp_csr_matvec_beta0_f32'})
                self.assertEqual(bool(matches), expected)
                if matches:
                    self.assertEqual([m.spec for m in matches], ['cusp_csr_matvec_beta0_f32'])


if __name__ == '__main__':
    unittest.main()
