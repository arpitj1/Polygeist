"""Ordered overwrite equations preserve overlapping, empty and sparse ranges."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from region_match.affine import atom_info, is_atom
from region_match.equations import FConst, FIndex, FLast, FRead, build_system
from region_match.mlir_parse import parse_module
from region_match.symexec import Unsupported, execute
from region_match.matcher import enumerate_bindings, compare_systems


SOURCE = '''module {
 func.func @intervals(%n: i32, %lo: memref<?xi32>, %hi: memref<?xi32>, %out: memref<?xi32>) {
  %one = arith.constant 1 : index
  %size = arith.index_cast %n : i32 to index
  affine.for %r = 0 to %size {
   %value = arith.index_cast %r : index to i32
   %a = affine.load %lo[%r] : memref<?xi32>
   %b = affine.load %hi[%r] : memref<?xi32>
   %start = arith.index_cast %a : i32 to index
   %end = arith.index_cast %b : i32 to index
   scf.for %p = %start to %end step %one {
    memref.store %value, %out[%p] : memref<?xi32>
   }
  }
  return
 }
}'''


def system(source=SOURCE):
    funcs, aliases = parse_module(source)
    program = execute(funcs[0], aliases)
    return build_system(program, {i for i, k in enumerate(program.top_kinds)
                                  if k in ('work', 'writeback')})


def affine(value, env, arrays):
    def variable(name):
        if not is_atom(name):
            return env[name]
        array, indices = atom_info(name)
        assert len(indices) == 1
        return arrays[array][affine(indices[0], env, arrays)]
    return value.const + sum(c * variable(v) for v, c in value.terms)


def evaluate(f, env, arrays):
    if isinstance(f, FConst):
        return f.v
    if isinstance(f, FIndex):
        return affine(f.value, env, arrays)
    if isinstance(f, FRead):
        return arrays[f.arr][affine(f.idx[0], env, arrays)]
    if isinstance(f, FLast):
        result = evaluate(f.default, env, arrays)
        for i in range(affine(f.lo, env, arrays), affine(f.hi, env, arrays)):
            local = dict(env, **{f.var: i})
            values = [(c.kind, affine(c.aff, local, arrays)) for c in f.guard]
            if all(v == 0 if k == 'eq' else v >= 0 for k, v in values):
                result = evaluate(f.body, local, arrays)
        return result
    raise AssertionError(type(f))


class OrderedOverwrite(unittest.TestCase):
    def test_overlaps_empty_ranges_and_untouched_elements(self):
        with patch.dict('os.environ', POLYGEIST_NUMERIC_MODEL='strict'):
            s = system()
        self.assertEqual(len(s.equations), 1)
        formula = s.equations[0].formula
        self.assertIsInstance(formula, FLast)
        for lows, highs in [([1,3,0,4], [5,4,0,7]),
                            ([6,2,1,5], [2,5,3,5]),
                            ([0,0,0,0], [0,0,0,0])]:
            initial = [19 + i for i in range(9)]
            expected = initial[:]
            for r, (lo, hi) in enumerate(zip(lows, highs)):
                for p in range(lo, hi):
                    expected[p] = r
            arrays = {'%lo': lows, '%hi': highs, '%out': initial}
            actual = [evaluate(formula, {'p%n':4, 'v0':i}, arrays)
                      for i in range(len(initial))]
            self.assertEqual(actual, expected)

    def test_integer_type_facts_do_not_authorize_truncation(self):
        with patch.dict('os.environ', POLYGEIST_NUMERIC_MODEL='strict'):
            system()
            # An i64 bound and a bound extended past INT_MAX are insufficient.
            wide = SOURCE.replace('%n: i32', '%n: i64').replace('%n : i32', '%n : i64')
            overflow = SOURCE.replace('to %size {',
                'to affine_map<(d0) -> (d0 + 2)>(%size) {')
            for source in (wide, overflow):
                with self.subTest(source=source), self.assertRaises(Unsupported):
                    system(source)

    def test_equivalence_requires_the_actual_stored_value(self):
        with patch.dict('os.environ', POLYGEIST_NUMERIC_MODEL='strict'):
            app = system()
            same = system(SOURCE.replace('@intervals', '@renamed'))
            wrong = system(SOURCE.replace('%value = arith.index_cast %r : index to i32',
                                          '%value = arith.constant 7 : i32'))
            def matches(spec):
                return any(compare_systems(app, spec, b).ok
                           for b in enumerate_bindings(app, spec, zero_cost_only=True))
            self.assertTrue(matches(same))
            self.assertFalse(matches(wrong))


if __name__ == '__main__':
    unittest.main()
