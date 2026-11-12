"""Nested scalar carry storage reuse preserves seeds, uses and FP policy."""
from pathlib import Path
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from region_match.mlir_parse import parse_module
from region_match.symexec import execute
from region_match.equations import build_system, Reject
from region_match.matcher import Binding, compare_systems

NESTED = '''module {
 func.func @nested(%x: memref<2x3x4xf32>, %out: memref<2xf32>) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %three = arith.constant 3 : index
  %seed = arith.constant 3.0 : f32
  affine.for %r = 0 to 2 {
   %result = scf.for %p = %zero to %three step %one iter_args(%outer = %seed) -> (f32) {
    %inner_result = affine.for %j = 0 to 4 iter_args(%inner = %outer) -> (f32) {
     %xv = affine.load %x[%r, %p, %j] : memref<2x3x4xf32>
     %next = arith.addf %inner, %xv : f32
     affine.yield %next : f32
    }
    scf.yield %inner_result : f32
   }
   affine.store %result, %out[%r] : memref<2xf32>
  }
  return
 }
}'''

DIRECT = '''module {
 func.func @direct(%x: memref<2x3x4xf32>, %out: memref<2xf32>) {
  %seed = arith.constant 3.0 : f32
  affine.for %r = 0 to 2 {
   affine.store %seed, %out[%r] : memref<2xf32>
   affine.for %p = 0 to 3 {
    affine.for %j = 0 to 4 {
     %xv = affine.load %x[%r, %p, %j] : memref<2x3x4xf32>
     %old = affine.load %out[%r] : memref<2xf32>
     %next = arith.addf %old, %xv : f32
     affine.store %next, %out[%r] : memref<2xf32>
    }
   }
  }
  return
 }
}'''

ARRAY_CARRY = '''module {
 func.func @array_carry(%x: memref<2x3x4xf32>, %out: memref<2xf32>) {
  %seed = arith.constant 3.0 : f32
  affine.for %r = 0 to 2 {
   affine.store %seed, %out[%r] : memref<2xf32>
   affine.for %p = 0 to 3 {
    %outer = affine.load %out[%r] : memref<2xf32>
    %result = affine.for %j = 0 to 4 iter_args(%inner = %outer) -> (f32) {
     %xv = affine.load %x[%r, %p, %j] : memref<2x3x4xf32>
     %next = arith.addf %inner, %xv : f32
     affine.yield %next : f32
    }
    affine.store %result, %out[%r] : memref<2xf32>
   }
  }
  return
 }
}'''


def system(source):
    functions, aliases = parse_module(source)
    program = execute(functions[0], aliases)
    return build_system(program, set(range(len(program.top_ops))))


def equal(a, b):
    binding = Binding({k: (k, k if k in a.outputs else None, tuple(range(v.rank)))
                       for k, v in a.arrays.items()}, {}, {})
    return compare_systems(a, b, binding).ok


class NestedScalarCarry(unittest.TestCase):
    def test_nested_and_output_element_carries_match_direct_updates(self):
        with patch.dict(os.environ, POLYGEIST_NUMERIC_MODEL='relaxed'):
            for source in (NESTED, ARRAY_CARRY):
                with self.subTest(source=source.split('@')[1].split('(')[0]):
                    self.assertTrue(equal(system(source), system(DIRECT)))
                    self.assertFalse(equal(system(source.replace('3.0', '4.0')), system(DIRECT)))

    def test_wrong_update_is_not_proved_equal(self):
        with patch.dict(os.environ, POLYGEIST_NUMERIC_MODEL='relaxed'):
            wrong = NESTED.replace('%next = arith.addf %inner, %xv',
                                   '%neg = arith.negf %xv : f32\n     %next = arith.addf %inner, %neg')
            self.assertFalse(equal(system(wrong), system(DIRECT)))

    def test_independent_old_outer_value_is_not_coalesced(self):
        with patch.dict(os.environ, POLYGEIST_NUMERIC_MODEL='relaxed'):
            for source in (NESTED, ARRAY_CARRY):
                wrong = source.replace('%next = arith.addf %inner, %xv',
                        '%captured = arith.addf %xv, %outer : f32\n     %next = arith.addf %inner, %captured')
                with self.assertRaises(Reject):
                    system(wrong)

    def test_empty_inner_loop_retains_original_seed(self):
        with patch.dict(os.environ, POLYGEIST_NUMERIC_MODEL='relaxed'):
            empty = NESTED.replace('to 4 iter_args', 'to 0 iter_args')
            direct_empty = DIRECT.replace('%j = 0 to 4', '%j = 0 to 0')
            self.assertTrue(equal(system(empty), system(direct_empty)))
            self.assertFalse(equal(system(empty.replace('3.0', '4.0')), system(direct_empty)))

    def test_strict_fp_does_not_reassociate_nested_updates(self):
        with patch.dict(os.environ, POLYGEIST_NUMERIC_MODEL='strict'):
            for source in (NESTED, ARRAY_CARRY):
                with self.assertRaises(Reject):
                    system(source)


if __name__ == '__main__':
    unittest.main()
