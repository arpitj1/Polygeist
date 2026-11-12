"""Execute a flat-prefix fill through tensorization, retaining its suffix."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE / 'lib_v2')]
import recipe
import validate_specs as V


class FlatFill(unittest.TestCase):
    def test_prefix_fill_preserves_unwritten_rows(self):
        root = HERE.parents[1]
        fixture = Path(__file__).parent / 'fixtures/flat_fill_prefix.c'
        with tempfile.TemporaryDirectory(dir=root/'issues') as directory:
            dest = Path(directory)
            raised = recipe.raise_c(recipe.PRODUCTION, str(fixture), 'flat_prefix',
                                    str(dest/'flat_prefix'),
                                    ['-DNETLIB_SPEC=flat_prefix'], 120)
            self.assertTrue(all(raised[k].ok for k in ('cgeist', 'raise', 'debuf')))
            text = Path(raised['paths']['debuf']).read_text()
            self.assertIn('bufferization.to_tensor', text)
            self.assertNotIn('memref.reinterpret_cast', text)
            manifest = dict(args=['seed', 'input', 'output'], validation=dict(
                arrays={'input': [5, 2, 3], 'output': [5, 2, 3]}))
            with patch.object(V, 'RAISED', dest), patch.object(V, 'WORK', dest/'validation'):
                result = V.validate('flat_prefix', manifest, fixture, set(), 'defn')
            self.assertEqual(result['verdict'], 'EXACT', result)


if __name__ == '__main__':
    unittest.main()
