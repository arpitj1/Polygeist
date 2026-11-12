"""Ordered integer recurrences preserve seeds, bounds, and typed updates."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from region_match.equations import (FAcc, FBin, FConst, FFold, FIndex, FRead,
                                    FSel, FUn, fwalk, has_acc)
from region_match.matcher import enumerate_bindings, compare_systems
from test_ordered_overwrite import affine, system

SOURCE = '''module {
 func.func @prefix(%n: index, %length: memref<?xi32>, %input: memref<?x7xi32>, %out: memref<?xi32>) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %z = arith.constant 0 : i32
  %seed = arith.constant 1 : i32
  affine.for %r = 0 to %n {
   %len = affine.load %length[%r] : memref<?xi32>
   %end = arith.index_cast %len : i32 to index
   %result = scf.for %c = %zero to %end step %one iter_args(%acc = %seed) -> (i32) {
    %x = memref.load %input[%r, %c] : memref<?x7xi32>
    %present = arith.cmpi ne, %x, %z : i32
    %value = arith.extui %present : i1 to i32
    %next = arith.andi %acc, %value : i32
    scf.yield %next : i32
   }
   affine.store %result, %out[%r] : memref<?xi32>
  }
  return
 }
}'''


def evaluate(f, env, arrays, acc=None):
    if isinstance(f, FAcc):
        return acc
    if isinstance(f, FConst):
        return int(f.v)
    if isinstance(f, FIndex):
        return affine(f.value, env, arrays)
    if isinstance(f, FRead):
        value = arrays[f.arr]
        for index in f.idx:
            value = value[affine(index, env, arrays)]
        return value
    if isinstance(f, FUn):
        assert f.op == 'arith.extui:i1 to i32'
        return evaluate(f.a, env, arrays, acc)
    if isinstance(f, FSel):
        assert f.cond[:2] == ('cmpf', 'arith.cmpi:i32:ne')
        a, b = [evaluate(x, env, arrays, acc) for x in f.cond[2:]]
        return evaluate(f.a if a != b else f.b, env, arrays, acc)
    if isinstance(f, FBin):
        assert f.op == 'arith.andi:i32'
        return evaluate(f.a, env, arrays, acc) & evaluate(f.b, env, arrays, acc)
    if isinstance(f, FFold):
        value = evaluate(f.default, env, arrays, acc)
        for i in range(affine(f.lo, env, arrays), affine(f.hi, env, arrays)):
            local = dict(env, **{f.var:i})
            if all(affine(c.aff, local, arrays) == 0 if c.kind == 'eq'
                   else affine(c.aff, local, arrays) >= 0 for c in f.guard):
                value = evaluate(f.body, local, arrays, value)
        return value
    raise AssertionError(type(f))


class OrderedFold(unittest.TestCase):
    def test_empty_mixed_lengths_and_nonboolean_values(self):
        with patch.dict('os.environ', POLYGEIST_NUMERIC_MODEL='strict'):
            result = system(SOURCE)
        formula = result.equations[0].formula
        self.assertIsInstance(formula, FFold)
        self.assertFalse(has_acc(formula))
        self.assertTrue(any(isinstance(x, FAcc) for x in fwalk(formula)))
        values = [[-7,2,3,0,5,6,7], [0,2,3,4,5,6,7],
                  [8,2,3,4,5,6,7], [2,-3,0,4,5,6,7]]
        for lengths in ([0,0,0,0], [1,7,5,3], [7,2,1,0], [-1,0,3,7]):
            arrays = {'%input':values, '%length':lengths, '%out':[99]*4}
            expected = [int(all(row[i] != 0 for i in range(n)))
                        for row, n in zip(values, lengths)]
            actual = [evaluate(formula, {'p%n':4,'v0':r}, arrays) for r in range(4)]
            self.assertEqual(actual, expected)

    def test_equivalence_rejects_wrong_seed_or_update(self):
        with patch.dict('os.environ', POLYGEIST_NUMERIC_MODEL='strict'):
            app = system(SOURCE)
            def matches(source):
                spec = system(source)
                return any(compare_systems(app, spec, b).ok
                           for b in enumerate_bindings(app, spec, zero_cost_only=True))
            self.assertTrue(matches(SOURCE.replace('@prefix','@unrelated')))
            self.assertFalse(matches(SOURCE.replace('%seed = arith.constant 1',
                                                   '%seed = arith.constant 0')))
            self.assertFalse(matches(SOURCE.replace('%next = arith.andi %acc, %value',
                                                   '%next = arith.addi %acc, %value')))


if __name__ == '__main__':
    unittest.main()
