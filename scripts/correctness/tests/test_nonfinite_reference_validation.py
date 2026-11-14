"""An executed nonfinite disagreement must reject the imported specification."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts/correctness/lib_v2'))
import validate_specs as V


class NonfiniteReferenceValidation(unittest.TestCase):
    def test_zero_maxabs_with_nan_mismatch_is_unsound(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'build') as tmp:
            d=Path(tmp);source=d/'reference.c'
            source.write_text('void NETLIB_SPEC(int n,float out[n]){for(int i=0;i<n;++i)out[i]=__builtin_nanf("");}')
            (d/'nonfinite.mlir').write_text('''module {
 func.func @nonfinite(%n:i32,%out:memref<?xf32>) {
  %size=arith.index_cast %n:i32 to index
  %zero=arith.constant 0.0:f32
  affine.for %i=0 to %size {affine.store %zero,%out[%i]:memref<?xf32>}
  return
 }
}''')
            with patch.object(V,'RAISED',d),patch.object(V,'WORK',d/'validate'):
                result=V.validate('nonfinite',{'args':['n','out']},source,set(),'affine')
            self.assertEqual(result['verdict'],'UNSOUND',result)
            self.assertEqual(result['maxabs'],0)
            self.assertGreater(result['mismatches'],0)


if __name__=='__main__':unittest.main()
