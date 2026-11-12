"""A proved complete region must remain visible before a partial replacement."""
from pathlib import Path
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'scripts/correctness'),str(ROOT/'scripts/correctness/lib_v2')]
import match_v2 as M
import recipe


class WholeRegionSelection(unittest.TestCase):
    def match(self, include_diagonal):
        start='i' if include_diagonal else 'i+1'
        source='''void triangular_update(int m,int n,double scale,
double left[m][m],double value[m][n]) {
 for(int i=0;i<m;++i)for(int j=0;j<n;++j){
  for(int k=START;k<m;++k)value[i][j]+=left[k][i]*value[k][j];
  value[i][j]*=scale;
 }
}'''.replace('START',start)
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);(d/'source.c').write_text(source)
            raised=recipe.raise_c(recipe.PRODUCTION,str(d/'source.c'),
                                 'triangular_update',str(d/'ir'),[],120)
            self.assertTrue(all(raised[k].ok for k in ('cgeist','raise','debuf')))
            with patch.dict(os.environ,POLYGEIST_NUMERIC_MODEL='relaxed'):
                return M.run(Path(raised['paths']['debuf']).read_text(),M.DEFAULT_LIB,
                             region=True,region_budget=15,
                             allowed_specs={'openblas_dscal','netlib_dtrmm_rutu'})

    def test_complete_candidate_wins_over_scaling_subexpression(self):
        out,report,diagnostics=self.match(False)
        self.assertIn('kernel.launch @netlib_dtrmm_rutu',out)
        self.assertNotIn('linalg.generic',out)
        self.assertNotIn('kernel.launch @openblas_dscal',out)
        self.assertTrue(diagnostics['region_alternative']['selected'])
        self.assertTrue(any(r.get('region_match')=='netlib_dtrmm_rutu' for r in report))

    def test_changed_diagonal_does_not_receive_the_complete_call(self):
        out,_,_=self.match(True)
        self.assertNotIn('kernel.launch @netlib_dtrmm_rutu',out)
        self.assertIn('linalg.generic',out)


if __name__=='__main__':unittest.main()
