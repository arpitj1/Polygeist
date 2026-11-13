"""Execute normalized input expressions against their original C reductions."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE/'lib_v2')]
import recipe as R
import validate_specs as V


class ReductionInputExpression(unittest.TestCase):
    def test_conditional_input_and_recurrence_execute_exactly(self):
        code = '''void NETLIB_SPEC(int rows,int cols,float shift,
                        float x[rows][cols],float out[rows]) {
  for(int r=0;r<rows;++r) {
    float acc=shift;
    for(int c=0;c<cols;++c) {
      float a=x[r][c];
      float value=a<0 ? -a : a;
      acc=acc+value;
    }
    out[r]=acc;
  }
}
'''
        name = 'conditional_reduce'
        with tempfile.TemporaryDirectory(dir=R.REPO/'build') as tmp:
            root = Path(tmp)
            ref = root/'reference.c';ref.write_text(code)
            src = root/'source.c'
            src.write_text('#define NETLIB_SPEC '+name+'\n#include "'+str(ref)+'"\n')
            stages = R.raise_c(R.PRODUCTION,str(src),name,str(root/name),[],120)
            self.assertTrue(stages['debuf'].ok, stages['debuf'].stderr)
            path = root/(name+'_debuf.mlir')
            subprocess.run([R.POLYGEIST_OPT,path,
                '--normalize-linalg-matching-views=split-reduction-inputs=true',
                '-o',root/'normalized.mlir'],check=True,capture_output=True)
            normalized = (root/'normalized.mlir').read_text()
            self.assertIn('iterator_types = ["parallel", "parallel"]',normalized)
            self.assertIn('arith.select',normalized)
            path.write_text(normalized)
            manifest = dict(args=['rows','cols','shift','x','out'],validation=dict(
                scalars=dict(rows=3,cols=7),arrays=dict(x=[3,7],out=[3])))
            with patch.object(V,'RAISED',root),patch.object(V,'WORK',root/'validate'):
                result = V.validate(name,manifest,ref,{'cols'})
            self.assertEqual(result['verdict'],'EXACT',json.dumps(result))


if __name__ == '__main__':
    unittest.main()
