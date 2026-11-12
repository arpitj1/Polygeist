"""Dynamic row loops must preserve their initial value and signed bounds."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'scripts/correctness'),str(ROOT/'scripts/correctness/lib_v2')]
import build_lib_v2 as B
import recipe
import validate_specs as V


class DynamicPrefixSemantics(unittest.TestCase):
    def test_nonzero_seed_and_negative_empty_row_match_original_c(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'issues') as tmp:
            d=Path(tmp);ref=d/'reference.c'
            ref.write_text('''void NETLIB_SPEC(float input[8][7],int length[8],float output[8],float seed){
 for(int row=0;row<8;++row){float value=seed;
  for(int col=0;col<length[row];++col)value+=input[row][col];
  output[row]=value;
 }}''')
            raw={'name':'row_fold','args':['input','length','output','seed'],
                 'validation':{'arrays':{'input':[8,7],'length':[8],'output':[8]},
                               'array_values':{'length':[-1,0,7,1,6,2,5,3]}}}
            spec=B.Spec('row_fold','row_fold',d/'manifest.json',ref,{},raw['args'],{},[],[],raw=raw)
            (d/'manifest.json').write_text(json.dumps({'specs':[raw]}))
            with patch.object(B,'RAISED',d/'raised'),patch.object(V,'RAISED',d/'raised'),patch.object(V,'WORK',d/'validate'):
                result=B.build_one(spec,recipe.PRODUCTION,60,True)
                self.assertNotIn('FAIL',result['status'])
                for stage in ['affine','linalg','defn']:
                    result=V.validate(spec.name,raw,ref,set(),stage)
                    self.assertEqual(result['verdict'],'EXACT',result)


if __name__=='__main__':unittest.main()
