"""Prove a complete scan without committing prematurely to its fill."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE / 'lib_v2')]
import recipe
import spec_match as S


class InitializerMatchOrder(unittest.TestCase):
    def run_case(self, product):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(__file__).parent / 'fixtures/large_scan_seed.c'
            raised = recipe.raise_c(recipe.PRODUCTION, str(fixture), 'large_scan_seed',
                                    str(Path(directory) / 'source'),
                                    ['-DSCAN_PRODUCT=' + str(int(product))], 120)
            self.assertTrue(all(raised[k].ok for k in ('cgeist', 'raise', 'debuf')))
            with patch.dict(os.environ, POLYGEIST_NUMERIC_MODEL='strict'):
                output, proof = S.match(Path(raised['paths']['debuf']).read_text(),
                    backend='cuda', only={'cub_inclusive_product_f32', 'thrust_fill_f32'},
                    region=False, max_operand_rank=1, fuse_initializers=True)
                S.verify(output, proof, backend='cuda')
            return output, proof

    def test_large_product_uses_complete_imported_scan(self):
        output, proof = self.run_case(True)
        self.assertEqual(proof['emitted'], ['cub_inclusive_product_f32'])
        self.assertNotIn('linalg.generic', output)
        self.assertEqual(proof['view_normalization']['source'], 'original')

    def test_addition_cannot_use_product_scan(self):
        output, proof = self.run_case(False)
        self.assertNotIn('cub_inclusive_product_f32', proof['emitted'])
        self.assertIn('linalg.generic', output)
        self.assertEqual(proof['view_normalization']['source'], 'matched_baseline')


if __name__ == '__main__':
    unittest.main()
