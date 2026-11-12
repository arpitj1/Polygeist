"""Exact Boolean absorption; arbitrary integers and FP predicates stay distinct."""
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import kernel_match as K


class BooleanComparison(unittest.TestCase):
    def test_exact_predicate_absorption(self):
        T = K.Term
        a, b = T.In(0), T.In(1)
        with patch.dict(os.environ, {'POLYGEIST_NUMERIC_MODEL': 'strict'}):
            for pred, complement in [('eq', 'ne'), ('ne', 'eq')]:
                p, n = T.Cmp(pred, a, T.Lit(0.0)), T.Cmp(complement, a, T.Lit(0.0))
                q = T.Cmp('ne', b, T.Lit(0.0))
                for outer, inner in [('or', 'and'), ('and', 'or')]:
                    left = T.Binary(outer, p, T.Binary(inner, n, q))
                    right = T.Binary(outer, p, q)
                    self.assertIsNotNone(K.body_matches_template(left, right))
                    wrong = T.Binary(outer, p, T.Binary(inner, n, b))
                    self.assertIsNone(K.body_matches_template(wrong, T.Binary(outer, p, b)))
            # Ordered floating equality and inequality are not complementary
            # when either input is NaN.
            p, n = T.Cmp('oeq', a, T.Lit(0.0)), T.Cmp('one', a, T.Lit(0.0))
            q = T.Cmp('ne', b, T.Lit(0.0))
            self.assertIsNone(K.body_matches_template(
                T.Binary('or', p, T.Binary('and', n, q)), T.Binary('or', p, q)))

    def test_independent_integer_truth_table(self):
        for a in range(-128, 128):
            for b in range(-128, 128):
                p, q = a != 0, b != 0
                self.assertEqual(p | ((not p) & q), p | q)
                self.assertEqual(p & ((not p) | q), p & q)


if __name__ == '__main__':
    unittest.main()
