"""Clipped scatter reductions retain every boundary predicate."""
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from region_match.mlir_parse import parse_module
from region_match.symexec import execute
from region_match.equations import (build_system, Builder, FBin, FConst,
                                    FRead, FSel, FSum, Reject)
from region_match.affine import Aff
from region_match.matcher import enumerate_bindings, compare_systems
from region_match import matcher as M


SOURCE = """module {
 func.func @scatter(%n: index, %k: index, %x: memref<?xf64>, %w: memref<?xf64>, %out: memref<?xf64>) {
  %one = arith.constant 1 : index
  %zero = arith.constant 0.0 : f64
  %sum = arith.addi %n, %k : index
  %length = arith.subi %sum, %one : index
  affine.for %j = 0 to %length {
   affine.store %zero, %out[%j] : memref<?xf64>
  }
  affine.for %i = 0 to %n {
   affine.for %j = 0 to %k {
    %a = affine.load %x[%i] : memref<?xf64>
    %b = affine.load %w[%j] : memref<?xf64>
    %old = affine.load %out[%i + %j] : memref<?xf64>
    %product = arith.mulf %a, %b : f64
    %new = arith.addf %old, %product : f64
    affine.store %new, %out[%i + %j] : memref<?xf64>
   }
  }
  return
 }
}
"""


def affine(a, env):
    return a.const + sum(c * env[v] for v, c in a.terms)


def condition(c, env):
    value = affine(c.aff, env)
    return value == 0 if c.kind == 'eq' else value >= 0


def evaluate(f, env, arrays):
    if isinstance(f, FConst):
        return f.v
    if isinstance(f, FBin):
        a, b = evaluate(f.a, env, arrays), evaluate(f.b, env, arrays)
        return {'add': lambda: a+b, 'sub': lambda: a-b,
                'mul': lambda: a*b}[f.op]()
    if isinstance(f, FRead):
        assert f.ver == 'in' and len(f.idx) == 1
        index = affine(f.idx[0], env)
        # A missing clipping predicate must not silently use Python's
        # negative indexing or an out-of-range element.
        assert 0 <= index < len(arrays[f.arr]), (f, index)
        return arrays[f.arr][index]
    if isinstance(f, FSel):
        assert f.cond[0] == 'aff'
        branch = f.a if all(condition(c, env) for c in f.cond[1]) else f.b
        return evaluate(branch, env, arrays)
    if isinstance(f, FSum):
        return sum(evaluate(f.body, dict(env, **{f.var: i}), arrays)
                   for i in range(affine(f.lo, env), affine(f.hi, env)))
    raise AssertionError(type(f))


class ClippedReduction(unittest.TestCase):
    def program(self, source=SOURCE, fixed=None):
        functions, aliases = parse_module(source)
        p = execute(functions[0], aliases, fixed)
        window = {i for i, kind in enumerate(p.top_kinds)
                  if kind in ('work', 'writeback')}
        return p, window

    def test_boundary_and_interior_values(self):
        with patch.dict(os.environ, POLYGEIST_NUMERIC_MODEL='relaxed'):
            system = build_system(*self.program())
        for n in range(1, 8):
            for k in range(1, 8):
                x, w = [2*i-5 for i in range(n)], [3*i+1 for i in range(k)]
                want = [0] * (n+k-1)
                for i, a in enumerate(x):
                    for j, b in enumerate(w):
                        want[i+j] += a*b
                for q, value in enumerate(want):
                    env = {'p%n': n, 'p%k': k, 'v0': q}
                    equations = [e for e in system.equations
                                 if e.arr == '%out' and
                                 all(condition(c, env) for c in e.domain)]
                    self.assertEqual(len(equations), 1)
                    self.assertEqual(evaluate(equations[0].formula, env,
                                              {'%x': x, '%w': w}), value)

    def test_strict_reassociation_still_rejected(self):
        with patch.dict(os.environ, POLYGEIST_NUMERIC_MODEL='strict'):
            with self.assertRaisesRegex(Reject, 'relaxed numerical model'):
                build_system(*self.program())

    def test_empty_intersections_and_unbounded_ranges(self):
        builder = Builder(*self.program())
        lower = [Aff.cst(0), Aff.var('a')]
        upper = [Aff.cst(5), Aff.var('b')]
        lo, lg = builder.guarded_bound(lower, [], 'i', True)
        hi, hg = builder.guarded_bound(upper, [], 'i', False)
        for a in range(-3, 9):
            for b in range(-3, 9):
                env = {'a': a, 'b': b}
                got = [i for i in range(affine(lo, env), affine(hi, env)+1)
                       if all(condition(c, dict(env, i=i)) for c in lg+hg)]
                self.assertEqual(got, list(range(max(0, a), min(5, b)+1)))
        with self.assertRaisesRegex(Reject, 'unbounded'):
            builder.guarded_bound([], [], 'i', True)

    def test_mask_extents_bind_and_changed_arithmetic_does_not_match(self):
        with patch.dict(os.environ, POLYGEIST_NUMERIC_MODEL='relaxed'):
            spec = build_system(*self.program())
            app = build_system(*self.program(fixed={'%n': 5, '%k': 3}))
            candidates = list(enumerate_bindings(app, spec))
            self.assertTrue(candidates)
            self.assertTrue(any(compare_systems(app, spec, b).ok for b in candidates))
            wrong = build_system(*self.program(
                SOURCE.replace('arith.mulf', 'arith.addf'), {'%n': 5, '%k': 3}))
            self.assertFalse(any(compare_systems(wrong, spec, b).ok
                                 for b in enumerate_bindings(wrong, spec)))

    def test_direct_search_requires_proof_and_emittable_call(self):
        with patch.dict(os.environ, POLYGEIST_NUMERIC_MODEL='relaxed'):
            spec = build_system(*self.program())
            app = build_system(*self.program(fixed={'%n': 5, '%k': 3}))
            wrong = build_system(*self.program(SOURCE.replace('arith.mulf', 'arith.addf')))
            entries = [SimpleNamespace(name=name, system=system, fixed_ints={},
                                       entry={}, argnames={})
                       for name, system in [('wrong', wrong), ('unemittable', spec),
                                            ('available', spec)]]
            probed = []
            def can_emit(s, b):
                probed.append(s.name)
                return s.name == 'available'
            def direct(*args, **kwargs):
                self.assertTrue(kwargs.get('zero_cost_only'),
                                'a usable minimum-cost match must finish before exhaustive search')
                yield from enumerate_bindings(*args, **kwargs)
            with patch.object(M, 'enumerate_bindings', direct):
                found = M._single_search(app, entries, [], can_emit)
            self.assertEqual([r['key'] for r in found], ['available'])
            self.assertNotIn('wrong', probed)
            self.assertIn('unemittable', probed)


if __name__ == '__main__':
    unittest.main()
