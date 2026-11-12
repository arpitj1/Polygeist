"""Fold known affine loop bounds without guessing unknown or overflowing values."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'scripts/correctness'), str(ROOT/'scripts/correctness/lib_v2')]
import match_v2 as M
import mlir_lite as ML


class AffineBoundInteger(unittest.TestCase):
    def evaluate(self, expression, values):
        source = '''module {
  func.func @bounds(%n: i32, %stride: i32) -> index {
    %size = arith.index_cast %n : i32 to index
    %step = arith.index_cast %stride : i32 to index
    %bound = affine.apply affine_map<(d0)[s0] -> (EXPR)>(%size)[%step]
    return %bound : index
  }
}'''.replace('EXPR', expression)
        matcher = M.Matcher.__new__(M.Matcher)
        matcher.L = ML.Module(source)
        matcher.A = ML.Module('module {}')
        matcher.spec = SimpleNamespace(yield_op=matcher.L.ops[-1])
        env = M.Env()
        env.vals.update({name: ('int', value, 'i32') for name, value in values.items()})
        self.assertEqual(matcher.sdef('%bound')[0].result_types, ['index'])
        return matcher.bound_integer('%bound', env)

    def test_dimension_and_symbol_arithmetic(self):
        self.assertEqual(self.evaluate('d0 - s0', {'%n':32, '%stride':1}), 31)
        self.assertEqual(self.evaluate('d0 * 3 + s0 - 7', {'%n':5, '%stride':2}), 10)

    def test_unknown_and_overflow_are_not_proofs(self):
        self.assertIsNone(self.evaluate('d0 - s0', {'%n':32}))
        self.assertIsNone(self.evaluate('d0 + s0', {'%n':2147483647, '%stride':1}))

    def test_unsupported_nonpolynomial_remains_unproved(self):
        self.assertIsNone(self.evaluate('d0 floordiv 3 + s0', {'%n':32, '%stride':1}))


if __name__ == '__main__':
    unittest.main()
