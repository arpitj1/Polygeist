"""Ordering normalization must not erase strict NaN or signed-zero behavior."""
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import kernel_match as K


class RelaxedExtrema(unittest.TestCase):
    def setUp(self):
        K.configure_matcher('egglog')

    def test_ordered_max_and_conditional_absolute_value(self):
        T = K.Term
        x, y = T.In(0), T.In(1)
        pairs = [
            (T.Select(T.Cmp('ogt', x, y), x, y), T.Binary('max', x, y)),
            (T.Select(T.Cmp('olt', x, y), y, x), T.Binary('max', y, x)),
            (T.Select(T.Cmp('olt', x, T.Lit(0.0)), T.Unary('negf', x), x), T.Abs(x)),
        ]
        for model in ('strict', 'relaxed'):
            with patch.dict(os.environ, POLYGEIST_NUMERIC_MODEL=model):
                for left, right in pairs:
                    with self.subTest(model=model, left=str(left)):
                        result = K.body_matches_template(left, right)
                        if model == 'strict':
                            self.assertIsNone(result)
                        else:
                            self.assertEqual(result, {})

    @patch.dict(os.environ, POLYGEIST_NUMERIC_MODEL='relaxed')
    def test_minimum_and_negative_absolute_value_do_not_match(self):
        T = K.Term
        x, y = T.In(0), T.In(1)
        self.assertIsNone(K.body_matches_template(
            T.Select(T.Cmp('olt', x, y), x, y), T.Binary('max', x, y)))
        self.assertIsNone(K.body_matches_template(
            T.Select(T.Cmp('ogt', x, T.Lit(0.0)), T.Unary('negf', x), x), T.Abs(x)))


if __name__ == '__main__':
    unittest.main()
