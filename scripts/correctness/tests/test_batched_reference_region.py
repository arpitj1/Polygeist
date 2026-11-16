"""A direct reference match remains visible inside a larger biased update."""
from pathlib import Path
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'scripts/correctness'),
               str(ROOT / 'scripts/correctness/lib_v2')]
import recipe
import spec_match


class BatchedReferenceRegion(unittest.TestCase):
    def test_batched_nonsquare_filter_with_separate_initialization(self):
        source = '''void filter_bank(float x[2][2][5][6][7],
 float w[3][2][2][3][4], float seed[3], float y[2][3][4][4][4]) {
 for (int b=0;b<2;b++) for (int o=0;o<3;o++)
  for (int z=0;z<4;z++) for (int h=0;h<4;h++) for (int j=0;j<4;j++) {
   y[b][o][z][h][j]=seed[o];
   for (int c=0;c<2;c++) for (int dz=0;dz<2;dz++)
    for (int dh=0;dh<3;dh++) for (int dj=0;dj<4;dj++)
     y[b][o][z][h][j]+=x[b][c][z+dz][h+dh][j+dj]*w[o][c][dz][dh][dj];
  }
}'''
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
                os.environ, POLYGEIST_SPEC_SUITE='lib_v3',
                POLYGEIST_NUMERIC_MODEL='relaxed'):
            folder = Path(tmp)
            (folder / 'input.c').write_text(source)
            raised = recipe.raise_c(recipe.PRODUCTION, str(folder/'input.c'),
                                    'filter_bank', str(folder/'raised'), [], 120)
            self.assertTrue(all(raised[k].ok for k in ('cgeist', 'raise', 'debuf')))
            matched, report = spec_match.match(
                Path(raised['paths']['debuf']).read_text(), backend='cuda',
                only=['cudnn_conv3d_valid_accumulate_f32'], region=True,
                region_budget=30)
            self.assertIn('cudnn_conv3d_valid_accumulate_f32', report['emitted'])
            self.assertIn('kernel.launch @cudnn_conv3d_valid_accumulate_f32', matched)


if __name__ == '__main__':
    unittest.main()
