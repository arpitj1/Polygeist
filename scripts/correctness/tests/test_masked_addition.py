"""Conditional addition: exact predicate complement, relaxed zero identity."""
import math
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import kernel_match as K


class MaskedAddition(unittest.TestCase):
    def test_ieee_complement_includes_unordered_inputs(self):
        T = K.Term
        a, b, x, y = (T.In(i) for i in range(4))
        with patch.dict(os.environ, POLYGEIST_NUMERIC_MODEL='strict'):
            self.assertTrue(K.equivalent(T.Select(T.Cmp('une', a, b), x, y),
                                         T.Select(T.Cmp('oeq', a, b), y, x)))
            self.assertFalse(K.equivalent(T.Select(T.Cmp('one', a, b), x, y),
                                          T.Select(T.Cmp('oeq', a, b), y, x)))
        values = [0., -0., 1., -1., math.inf, -math.inf, math.nan]
        for a in values:
            for b in values:
                unordered_ne = math.isnan(a) or math.isnan(b) or a != b
                ordered_eq = not(math.isnan(a) or math.isnan(b)) and a == b
                self.assertEqual(unordered_ne, not ordered_eq)

    def test_masked_addition_requires_relaxed_zero_semantics(self):
        T = K.Term
        old, value = T.Out(0), T.In(0)
        p = T.Cmp('oeq', value, value)
        branch = T.Select(p, old + value, old)
        masked = old + T.Select(T.Cmp('une', value, value), T.Lit(0.), value)
        with patch.dict(os.environ, POLYGEIST_NUMERIC_MODEL='strict'):
            self.assertFalse(K.equivalent(branch, masked, False))
            self.assertIsNone(K.body_matches_template(branch, masked))
        with patch.dict(os.environ, POLYGEIST_NUMERIC_MODEL='relaxed'):
            self.assertTrue(K.equivalent(branch, masked, False))
            self.assertIsNotNone(K.body_matches_template(branch, masked))
            wrong = T.Select(p, old + value, value)
            self.assertFalse(K.equivalent(wrong, masked, False))
        # The strict rejection is necessary for a negative-zero accumulator.
        self.assertNotEqual(math.copysign(1., -0.), math.copysign(1., -0. + 0.))


if __name__ == '__main__':
    unittest.main()
