from pathlib import Path
import json
import os
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts/correctness'))
import spec_match as S


class ConfiguredPipeline(unittest.TestCase):
    def test_approximation_policy_filters_before_matching(self):
        rows = S.registry()
        for policy, expected in (({}, False), ({'opencv_box3x3_u8': 2}, False),
                                 ({'opencv_box3x3_u8': 3}, True)):
            with patch.dict(os.environ, POLYGEIST_NUMERIC_MODEL='strict',
                            POLYGEIST_BACKEND_ERROR_BUDGETS=json.dumps(policy)):
                allowed, _ = S.capabilities(rows, 'pva', only=['opencv_box3x3_u8'])
                self.assertEqual('opencv_box3x3_u8' in allowed, expected)

    def test_recovered_configuration_appears_in_proof_evidence(self):
        with patch.dict(os.environ, POLYGEIST_BACKEND_ERROR_BUDGETS='{}'):
            info = S.backend_evidence(['opencv_gaussian3x3_u8'], S.registry())
        values = info['opencv_gaussian3x3_u8']['configuration']
        self.assertEqual(values['sigma_x'], 0.8493217825889587)
        self.assertEqual(values['sigma_y'], values['sigma_x'])
        self.assertIsNone(info['opencv_gaussian3x3_u8']['numerical_contract'])


if __name__ == '__main__':
    unittest.main()
