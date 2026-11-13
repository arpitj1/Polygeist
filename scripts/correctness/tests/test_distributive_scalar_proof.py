"""Distributive capture proposals require a bounded Egglog proof."""
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import kernel_match as K


class DistributiveScalarProof(unittest.TestCase):
    def setUp(self):
        K.configure_matcher(collect_egraph_sizes=True)
        self.a, self.b, self.x = [('Cap', name) for name in ('a', 'b', 'x')]
        self.left = ('Add', ('Mul', ('Sub', ('Lit', 1.0), self.x), self.a),
                     ('Mul', self.x, self.b))
        self.right = ('Add', self.a, ('Mul', self.x, ('Sub', self.b, self.a)))

    def test_relaxed_egraph_proves_reassociation(self):
        with patch.dict(os.environ, POLYGEIST_NUMERIC_MODEL='relaxed'):
            self.assertTrue(K.prove_distributive_binding(self.left, self.right, {}))
        telemetry = K.matcher_telemetry()
        self.assertEqual(telemetry['proofs_matched'], 1)
        self.assertGreater(telemetry['egraph_nodes_max'], 0)

    def test_wrong_sign_and_strict_arithmetic_do_not_match(self):
        wrong = ('Add', self.a, ('Mul', self.x, ('Add', self.b, self.a)))
        with patch.dict(os.environ, POLYGEIST_NUMERIC_MODEL='relaxed'):
            self.assertFalse(K.prove_distributive_binding(self.left, wrong, {}))
        with patch.dict(os.environ, POLYGEIST_NUMERIC_MODEL='strict'):
            self.assertFalse(K.prove_distributive_binding(self.left, self.right, {}))

    def test_syntactic_ablation_and_timeout_cannot_accept(self):
        with patch.dict(os.environ, POLYGEIST_NUMERIC_MODEL='relaxed'):
            K.configure_matcher('syntactic')
            self.assertFalse(K.prove_distributive_binding(self.left, self.right, {}))
            K.configure_matcher()
            with patch.object(K.subprocess, 'run', side_effect=K.subprocess.TimeoutExpired('proof', 10)):
                self.assertFalse(K.prove_distributive_binding(self.left, self.right, {}))
            self.assertEqual(K.matcher_telemetry()['proofs_over_10s'], 1)

    def test_worker_obeys_rule_ablation(self):
        for mode in ('none', 'drop:distrib'):
            with patch.dict(os.environ, POLYGEIST_NUMERIC_MODEL='relaxed', EQSAT_ABLATION=mode):
                self.assertFalse(K.prove_distributive_binding(self.left, self.right, {}))
        with patch.dict(os.environ, POLYGEIST_NUMERIC_MODEL='relaxed', EQSAT_ABLATION='drop:mask'):
            self.assertTrue(K.prove_distributive_binding(self.left, self.right, {}))

    def test_worker_failure_is_not_reported_as_a_timeout(self):
        with patch.dict(os.environ, POLYGEIST_NUMERIC_MODEL='relaxed'):
            with patch.object(K.subprocess, 'run', side_effect=OSError('worker unavailable')):
                self.assertFalse(K.prove_distributive_binding(self.left, self.right, {}))
        self.assertEqual(K.matcher_telemetry()['proofs_over_10s'], 0)
        self.assertEqual(K.matcher_telemetry()['distributive_worker_errors'], 1)


if __name__ == '__main__':
    unittest.main()
