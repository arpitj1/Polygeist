"""Recover library layout arguments from declared inner array dimensions."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'scripts/correctness'),str(ROOT/'scripts/correctness/lib_v2')]
import match_v2 as M
import recipe


class DeclaredDimensionMatch(unittest.TestCase):
    def run_match(self, dynamic):
        extra='int height,int width,' if dynamic else ''
        height,width=('height','width') if dynamic else ('9','11')
        source=f'''void unrelated({extra}float input[2][3][{height}][{width}],
float filter[5][3][3][2],float output[2][5][7][10]) {{
for(int n=0;n<2;++n)for(int o=0;o<5;++o)for(int y=0;y<7;++y)for(int x=0;x<10;++x){{
float v=0;for(int c=0;c<3;++c)for(int ky=0;ky<3;++ky)for(int kx=0;kx<2;++kx)
v=filter[o][c][ky][kx]*input[n][c][y+ky][x+kx]+v;
output[n][o][y][x]=v;
}}}}'''
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);(d/'input.c').write_text(source)
            raised=recipe.raise_c(recipe.PRODUCTION,str(d/'input.c'),'unrelated',str(d/'raised'),[],60)
            self.assertTrue(all(raised[k].ok for k in ['cgeist','raise','debuf']))
            with patch.dict(os.environ,{'POLYGEIST_NUMERIC_MODEL':'strict'}):
                return M.run(Path(raised['paths']['debuf']).read_text(),
                    ROOT/'generic_solver/kernel_library_v3.mlir',region=False,diag=True,
                    allowed_specs={'cudnn_conv2d_valid_d1_f32'})

    def test_static_layout_is_recovered_from_declaration(self):
        output,matches,_=self.run_match(False)
        self.assertEqual(len(matches),1)
        self.assertEqual(matches[0]['binding']['%IH'],'literal 9')
        self.assertEqual(matches[0]['binding']['%IW'],'literal 11')
        self.assertNotIn('linalg.generic',output)

    def test_unresolved_dynamic_layout_is_not_guessed(self):
        _,matches,diagnostics=self.run_match(True)
        self.assertFalse(matches)
        self.assertTrue(any('unbound parameters' in r['why'] for r in diagnostics['rejected']))


if __name__=='__main__':unittest.main()
