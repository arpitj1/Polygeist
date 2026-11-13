"""Symbolic layout parameters bind by exact arithmetic and bounded geometry."""
import copy
from pathlib import Path
import sys
import unittest
HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE/'lib_v2')]
import abi_v2 as A
import match_v2 as M
import mlir_lite as ML
import typed_wrapper as W


class SymbolicLayoutParameters(unittest.TestCase):
    def test_constant_stride_specialization_is_exact(self):
        matcher = object.__new__(M.Matcher)
        reference = {(('d', 0), ('v', '%stride')): 2, (('d', 1),): 1}
        application = {(('d', 0),): 64, (('d', 1),): 1}
        self.assertEqual(matcher.compare_poly(reference, application), [('extent', '%stride', 32)])
        application[(('d', 0),)] = 63
        self.assertIsNone(matcher.compare_poly(reference, application))
        application[(('d', 0),)] = -64
        self.assertIsNone(matcher.compare_poly(reference, application))
        application[(('d', 0),)] = 2**33
        self.assertIsNone(matcher.compare_poly(reference, application))

    def test_different_coordinates_do_not_match(self):
        matcher = object.__new__(M.Matcher)
        reference = {(('d', 0), ('v', '%stride')): 1}
        self.assertIsNone(matcher.compare_poly(reference, {(('d', 1),): 32}))

    def geometry(self, change=None):
        args = [A.SpecArg('m','int',ctype='int'), A.SpecArg('n','int',ctype='int'),
                A.SpecArg('output','array',rank=2,ctype='float',intent='out')]
        cfg = dict(kind='typed_c_wrapper', arrays={'output':[{'product':['m','n']}, 4]},
                   bounds={name:dict(min=1,max=1024) for name in ('m','n')})
        if change:
            change(cfg)
        return W.geometry(A, {'backends':['cuda']}, args, cfg, ['m','n','output'])

    def test_product_extents_keep_checked_capacity(self):
        params, bindings, cfg = self.geometry()
        self.assertEqual(cfg['arrays']['output'][0], {'product':['m','n']})
        self.assertEqual(W.extent_expression(cfg['arrays']['output'][0]), '((int64_t)m * (int64_t)n)')
        for update in (
                lambda c:c['bounds']['m'].update(min=0),
                lambda c:c['bounds']['n'].update(max=2**30),
                lambda c:c['arrays']['output'][0].update(product=['m','unknown']),
                lambda c:c['arrays']['output'][0].update(product=['m',-1])):
            with self.subTest(update=update),self.assertRaises(A.AbiError):
                self.geometry(update)


if __name__ == '__main__':
    unittest.main()
