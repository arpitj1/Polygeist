"""Opaque scalar calls require exact names, types, order, and explicit purity."""
from pathlib import Path
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import kernel_match as K


class PureScalarCongruence(unittest.TestCase):
    def body(self, name='helper', types='f32,f32', operands='%x, %y', pure=True):
        signature = '(' + types + ')->f32'
        return K.GenericBody(
            ins_arg_names=['%x', '%y'], outs_arg_names=['%out'],
            body_lines=[f'%r = func.call @{name}({operands}) : ({types}) -> f32'],
            yield_values=['%r'], captures=[], indexing_maps=[], iterator_types=['parallel'],
            constants={}, block_arg_types=['f32','f32','f32'],
            scalar_calls={name: signature} if pure else {})

    def test_congruence_is_exact_in_both_models(self):
        a = K.encode_body(self.body())
        for model in ('strict', 'relaxed'):
            with patch.dict(os.environ, POLYGEIST_NUMERIC_MODEL=model):
                self.assertTrue(K.equivalent(a, K.encode_body(self.body())))
                for body in (self.body(name='other'), self.body(types='f64,f64'),
                             self.body(operands='%y, %x')):
                    self.assertFalse(K.equivalent(a, K.encode_body(body)))

    def test_missing_or_mismatched_pure_declaration_is_rejected(self):
        for body in (self.body(pure=False), self.body()):
            if body.scalar_calls:
                body.scalar_calls['helper'] = '(f64,f64)->f32'
            with self.assertRaises(ValueError):
                K.encode_body(body)

    def test_only_scalar_pure_declarations_are_admitted(self):
        text = '''
func.func private @good(f32, i32) -> f32 attributes {polygeist.pure}
func.func private @effectful(f32) -> f32 attributes {llvm.linkage = #llvm.linkage<external>}
func.func private @memory(memref<?xf32>) -> f32 attributes {polygeist.pure}
'''
        self.assertEqual(K.pure_scalar_declarations(text), {'good': '(f32,i32)->f32'})

    def test_standard_extrema_join_arithmetic_only_in_relaxed_model(self):
        call = K.encode_body(self.body(name='fmaxf'))
        selection = K.Term.Select(K.Term.Cmp('olt', K.Term.In(0), K.Term.In(1)),
                                  K.Term.In(1), K.Term.In(0))
        with patch.dict(os.environ, POLYGEIST_NUMERIC_MODEL='strict'):
            self.assertFalse(K.equivalent(call, selection))
        with patch.dict(os.environ, POLYGEIST_NUMERIC_MODEL='relaxed'):
            self.assertTrue(K.equivalent(call, selection))
            for other in (self.body(name='helper'),
                          self.body(name='fmaxf', types='f64,f64')):
                self.assertFalse(K.equivalent(K.encode_body(other), selection))

    def test_standard_minimum_preserves_operand_identity(self):
        call = K.encode_body(self.body(name='fminf'))
        minimum = K.Term.Binary('min', K.Term.In(0), K.Term.In(1))
        with patch.dict(os.environ, POLYGEIST_NUMERIC_MODEL='strict'):
            self.assertFalse(K.equivalent(call, minimum))
        with patch.dict(os.environ, POLYGEIST_NUMERIC_MODEL='relaxed'):
            self.assertTrue(K.equivalent(call, minimum))
            self.assertFalse(K.equivalent(call,
                K.Term.Binary('max', K.Term.In(0), K.Term.In(1))))


if __name__ == '__main__':
    unittest.main()
