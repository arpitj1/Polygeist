"""Exact integer reorderings cannot borrow floating-point algebra rules."""
from pathlib import Path
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import kernel_match as K


@patch.dict(os.environ, {'POLYGEIST_NUMERIC_MODEL': 'strict'})
class IntegerCommutativity(unittest.TestCase):
    def test_typed_operand_exchange_and_capture(self):
        K.configure_matcher('egglog')
        T=K.Term;a,b=T.In(0),T.In(1)
        for op in sorted(K.INTEGER_COMMUTATIVE):
            with self.subTest(op=op):
                self.assertTrue(K.equivalent(T.Binary(op,a,b),T.Binary(op,b,a)))
                self.assertIsNotNone(K.body_matches_template(
                    T.Binary(op,T.Cap('offset'),a),T.Binary(op,a,T.Cap('parameter'))))
        for op in ('subi','divsi','divui','remsi','remui'):
            self.assertFalse(K.equivalent(T.Binary('arith.'+op+':i32',a,b),
                                         T.Binary('arith.'+op+':i32',b,a)))
        self.assertFalse(K.equivalent(T.Binary('arith.addi:i32',a,b),
                                     T.Binary('arith.addi:i64',b,a)))
        self.assertFalse(K.equivalent(T.Binary('arith.addi:i32',a,b),a+b))

    def test_wraparound_boundary_values(self):
        for width in (8,16,32,64):
            mask=(1<<width)-1
            values=range(256) if width==8 else (0,1,2,mask,mask-1,mask//2,mask//2+1)
            for a in values:
                for b in values:
                    self.assertEqual((a+b)&mask,(b+a)&mask)
                    self.assertEqual((a*b)&mask,(b*a)&mask)


if __name__ == '__main__':
    unittest.main()
