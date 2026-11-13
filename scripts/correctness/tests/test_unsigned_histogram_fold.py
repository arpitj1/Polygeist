"""Unsigned indirect indices and ordered nested updates retain integer meaning."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from region_match.affine import atom_info, atom_conversion, is_atom
from region_match.equations import FAcc, FBin, FConst, FFold, FRead, Reject
from region_match.matcher import compare_systems, enumerate_bindings
from region_match.symexec import Unsupported
from test_ordered_overwrite import system


HISTOGRAM = '''module {
 func.func @hist(%input: memref<2x3xi8>, %out: memref<256xi32>) {
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  affine.for %b = 0 to 256 {
   affine.store %zero, %out[%b] : memref<256xi32>
  }
  affine.for %r = 0 to 2 {
   affine.for %c = 0 to 3 {
    %byte = affine.load %input[%r, %c] : memref<2x3xi8>
    %index = arith.index_castui %byte : i8 to index
    %count = memref.load %out[%index] : memref<256xi32>
    %next = arith.addi %count, %one : i32
    memref.store %next, %out[%index] : memref<256xi32>
   }
  }
  return
 }
}'''


def affine(value, env, arrays):
    def variable(name):
        if not is_atom(name):
            return env[name]
        array, indices = atom_info(name)
        result = arrays[array]
        for index in indices:
            result = result[affine(index, env, arrays)]
        conversion = atom_conversion(name)
        if conversion:
            assert conversion.startswith('unsigned-read:i')
            result &= (1 << int(conversion.split(':i')[1])) - 1
        return result
    return value.const + sum(c * variable(v) for v, c in value.terms)


def evaluate(f, env, arrays, acc=None):
    if isinstance(f, FAcc):
        return acc
    if isinstance(f, FConst):
        return int(f.v)
    if isinstance(f, FRead):
        value = arrays[f.arr]
        for index in f.idx:
            value = value[affine(index, env, arrays)]
        return value
    if isinstance(f, FBin):
        a, b = evaluate(f.a, env, arrays, acc), evaluate(f.b, env, arrays, acc)
        operation = {'arith.addi:i32': lambda: a+b, 'arith.muli:i32': lambda: a*b}[f.op]
        return operation() & 0xffffffff
    if isinstance(f, FFold):
        value = evaluate(f.default, env, arrays, acc)
        for i in range(affine(f.lo, env, arrays), affine(f.hi, env, arrays)):
            local = dict(env, **{f.var:i})
            if all(affine(c.aff, local, arrays) == 0 if c.kind == 'eq'
                   else affine(c.aff, local, arrays) >= 0 for c in f.guard):
                value = evaluate(f.body, local, arrays, value)
        return value
    raise AssertionError(type(f))


class UnsignedHistogramFold(unittest.TestCase):
    def setUp(self):
        self.policy = patch.dict('os.environ', POLYGEIST_NUMERIC_MODEL='strict')
        self.policy.start()
        self.addCleanup(self.policy.stop)

    def test_full_byte_range_and_repeated_bins(self):
        formula = system(HISTOGRAM).equations[0].formula
        self.assertIsInstance(formula, FFold)
        # Values are the signed i8 interpretation of raw source bytes.
        arrays = {'%input': [[0, 127, -128], [-1, -1, 3]], '%out': [99]*256}
        actual = [evaluate(formula, {'v0':b}, arrays) for b in range(256)]
        expected = [0]*256
        for byte in [0, 127, 128, 255, 255, 3]:
            expected[byte] += 1
        self.assertEqual(actual, expected)

    def test_addresses_match_but_wrong_updates_and_signed_casts_do_not(self):
        app = system(HISTOGRAM)
        def matches(source):
            try:
                spec = system(source)
                return any(compare_systems(app, spec, binding).ok
                           for binding in enumerate_bindings(app, spec, zero_cost_only=True))
            except (Reject, Unsupported):
                return False
        self.assertTrue(matches(HISTOGRAM.replace('@hist', '@unrelated').replace(
            '%input[%r, %c]', '%input[0, %r * 3 + %c]')))
        self.assertFalse(matches(HISTOGRAM.replace('%one = arith.constant 1',
                                                  '%one = arith.constant 2')))
        self.assertFalse(matches(HISTOGRAM.replace('arith.index_castui', 'arith.index_cast')))

    def test_nested_noncommutative_update_preserves_iteration_order(self):
        source = '''module {
 func.func @ordered(%input: memref<2x2xi32>, %out: memref<1xi32>) {
  %zero = arith.constant 0 : i32
  %three = arith.constant 3 : i32
  affine.store %zero, %out[0] : memref<1xi32>
  affine.for %r = 0 to 2 {
   affine.for %c = 0 to 2 {
    %x = affine.load %input[%r, %c] : memref<2x2xi32>
    %acc = affine.load %out[0] : memref<1xi32>
    %scaled = arith.muli %acc, %three : i32
    %next = arith.addi %scaled, %x : i32
    affine.store %next, %out[0] : memref<1xi32>
   }
  }
  return
 }
}'''
        arrays = {'%input': [[1,2], [3,4]], '%out': [99]}
        formula = system(source).equations[0].formula
        self.assertEqual(evaluate(formula, {'v0':0}, arrays), 58)
        permuted = source.replace('%input[%r, %c]', '%input[%c, %r]')
        other = system(permuted).equations[0].formula
        self.assertEqual(evaluate(other, {'v0':0}, arrays), 64)


if __name__ == '__main__':
    unittest.main()
