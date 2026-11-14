"""Recurrence expressions keep SSA sharing during matching and proof checks."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'scripts/correctness'))
import kernel_match as K


def recurrence(leaf, count=50):
    previous, current = ('Lit', 1.), leaf
    for _ in range(count):
        previous, current = current, ('Sub', ('Mul', leaf, current), previous)
    return current


class SharedScalarDAG(unittest.TestCase):
    def test_recurrence_capture_binding_and_exact_proof(self):
        template = recurrence(('Cap', 'weight'))
        application = recurrence(('Lit', 0.375))
        with patch.dict('os.environ', POLYGEIST_NUMERIC_MODEL='strict'):
            bindings = K._unify(application, template, {})
            self.assertEqual(bindings, {'weight': ('Lit', 0.375)})
            self.assertTrue(K._egglog_accepts_binding(application, template, bindings))
            self.assertFalse(K._egglog_accepts_binding(application, template,
                                                       {'weight': ('Lit', 0.5)}))
        substituted = K._substitute_caps(template, bindings)
        # Each recurrence refers to the previous recurrence's predecessor.
        self.assertIs(substituted[2], substituted[1][2][1][2])

    def test_leaf_multiplicity_is_preserved(self):
        node = ('In', 0)
        for _ in range(50):
            node = ('Add', node, node)
        self.assertEqual(K._indexed_leaf_counts(node), {('In', 0): 2**50})
        self.assertEqual(K._ast_node_count(node), 2**51-1)
        self.assertEqual(K._ordered_unique_ast_nodes(node, {'In'}), [('In', 0)])

    def test_repeated_capture_cannot_bind_different_values(self):
        cap = ('Cap', 'x')
        template = ('Sub', cap, cap)
        self.assertIsNone(K._unify(('Sub', ('Lit', 1.), ('Lit', 2.)), template, {}))

    def test_signed_zero_stays_distinct(self):
        with patch.dict('os.environ', POLYGEIST_NUMERIC_MODEL='strict'):
            self.assertFalse(K._egglog_accepts_binding(recurrence(('Lit', 0.)),
                                                       recurrence(('Lit', -0.)), {}))

    def test_region_tree_expansion_declines_large_shared_recurrence(self):
        from region_match.equations import build_system, Reject
        from region_match.symexec import EBin, EConst
        node = EConst(1.)
        for _ in range(50):
            node = EBin('add', node, node)
        program = SimpleNamespace(stmts=[SimpleNamespace(top=0, value=node)])
        with self.assertRaisesRegex(Reject, 'expanded scalar expression'):
            build_system(program, {0})


if __name__ == '__main__':
    unittest.main()
