import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'lib_v2'))
import backend_contract as backend


class RecordedApproximation(unittest.TestCase):
    target = {'numerical_contract': dict(schema=1, kind='recorded_approximation',
              source='https://example.org/backend', assumptions=['Measured independently.'])}

    def test_requires_acceptance_for_the_specific_reference(self):
        for accepted, allowed in [('[]', False), ('["other"]', False), ('["reference"]', True)]:
            with patch.dict(os.environ, POLYGEIST_BACKEND_APPROXIMATIONS=accepted):
                self.assertEqual(backend.rejection('reference', self.target, {}) is None, allowed)
                if allowed:
                    evidence = backend.evidence('reference', self.target, {})
                    self.assertTrue(evidence['accepted_approximation'])
                    self.assertNotIn('max_abs_error_bound', evidence)
                    self.assertNotIn('accepted_error_budget', evidence)

    def test_error_budget_does_not_implicitly_accept_unbounded_approximation(self):
        with patch.dict(os.environ, POLYGEIST_BACKEND_APPROXIMATIONS='[]'):
            self.assertIsNotNone(backend.rejection('reference', self.target, {'reference': 255}))

    def test_invalid_acceptance_is_rejected(self):
        for value in ('{}', '[null]', '[3]', 'true', 'invalid'):
            with patch.dict(os.environ, POLYGEIST_BACKEND_APPROXIMATIONS=value):
                with self.assertRaises(ValueError):
                    backend.approximations()


if __name__ == '__main__':
    unittest.main()
