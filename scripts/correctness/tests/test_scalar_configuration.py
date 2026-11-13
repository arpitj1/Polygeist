from pathlib import Path
import copy
import itertools
import json
import os
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts/correctness/lib_v2'))
import scalar_configuration as S
import backend_contract as B


def entry(name):
    return json.loads((ROOT / ('generic_solver/lib_v3/manifest/' + name + '.json')).read_text())['specs'][0]


class ScalarConfiguration(unittest.TestCase):
    def test_generic_configuration_has_no_operation_identity(self):
        cfg = dict(schema=1, inputs=['SCALE'], values={'gain': 'f32(1.0 / SCALE)'},
                   checks=['0 < gain <= 1'])
        self.assertEqual(S.resolve(cfg, {'SCALE': 8})['gain'], .125)
        self.assertEqual(S.resolve(cfg, {'SCALE': 16})['gain'], .0625)

    def test_gaussian_parameters_follow_imported_coefficients(self):
        spec = entry('opencv_gaussian_fixed')
        result = S.resolve(spec['target']['runtime_adapter']['configuration'], spec['defines'])
        self.assertEqual(result['sigma_x'], S.f32(.8493218))
        self.assertEqual(result['sigma_y'], result['sigma_x'])
        self.assertEqual([result[k] for k in ('corner', 'x_edge', 'y_edge', 'center')], [16, 32, 32, 64])

    def test_wrong_api_parameter_does_not_pass_coefficient_check(self):
        spec = entry('opencv_gaussian_fixed')
        cfg = spec['target']['runtime_adapter']['configuration']
        cfg['values']['sigma_x'] = 'f32(1.0)'
        with self.assertRaisesRegex(ValueError, 'configuration check failed'):
            S.resolve(cfg, spec['defines'])

    def test_changed_normalization_or_rounding_rejected(self):
        for change in ({'H_SIDE_SHIFT': 3}, {'V_ROUND': 0}, {'V_OUTPUT_SHIFT': 9}):
            spec = entry('opencv_gaussian_fixed')
            spec['defines'].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                S.resolve(spec['target']['runtime_adapter']['configuration'], spec['defines'])

    def test_api_literal_is_exactly_checked_fp32(self):
        argument = {'configuration': 'x', 'type': 'float'}
        with self.assertRaisesRegex(ValueError, 'actual FP32'):
            S.literal(argument, {'x': .1}, 'float')
        value = S.f32(.1)
        self.assertEqual(S.literal(argument, {'x': value}, 'float'), value.hex() + 'f')
        with self.assertRaises(ValueError):
            S.literal(argument, {'x': value}, 'double')

    def test_forbidden_expressions(self):
        for expression in ('__import__("os")', 'x.real', '[1, 2]', '1e999',
                           '2 ** 999999999', '1 / 0', 'sqrt(-1)', 'f32(1e38 * 10)'):
            with self.subTest(expression=expression), self.assertRaises(ValueError):
                S.evaluate(expression, {'x': 1})

    def test_unknown_input_and_unchecked_configuration_rejected(self):
        cfg = dict(schema=1, inputs=['X'], values={'x': 'X'}, checks=['x > 0'])
        with self.assertRaises(ValueError):
            S.resolve(cfg, {})
        cfg['checks'] = []
        with self.assertRaises(ValueError):
            S.resolve(cfg, {'X': 1})


class BackendNumerics(unittest.TestCase):
    def test_box_bound_and_policy(self):
        spec = entry('opencv_box_fixed')
        info = B.analyze(spec['target']['numerical_contract'])
        self.assertEqual(info['max_abs_error_bound'], 3)
        self.assertIn('explicit error budget', B.rejection(spec['name'], spec['target'], {}))
        self.assertIsNotNone(B.rejection(spec['name'], spec['target'], {spec['name']: 2}))
        self.assertIsNone(B.rejection(spec['name'], spec['target'], {spec['name']: 3}))

    def test_box_reference_normalization_for_all_possible_sums(self):
        for total in range(9 * 255 + 1):
            self.assertEqual(((total + 4) * 932068) >> 23, (total + 4) // 9)

    def test_box_bound_contains_extreme_inputs_and_preserves_counterexample(self):
        cfg = entry('opencv_box_fixed')['target']['numerical_contract']
        largest = 0
        for values in itertools.product((0, 255), repeat=9):
            source = (sum(values) + 4) // 9
            vendor = (sum(a*b for a, b in zip(values, cfg['backend_weights'])) + 128) // 256
            largest = max(largest, abs(source - vendor))
        self.assertEqual(largest, 3)
        self.assertLessEqual(largest, B.analyze(cfg)['max_abs_error_bound'])

    def test_malformed_or_different_normalization_rejected(self):
        cfg = entry('opencv_box_fixed')['target']['numerical_contract']
        cfg['backend_weights'][0] += 1
        with self.assertRaisesRegex(ValueError, 'normalization'):
            B.analyze(cfg)

    def test_unscoped_tolerance_is_not_a_contract(self):
        self.assertIsNotNone(B.rejection('x', {'numerical_contract': {'max_abs_error': 3}}, {'x': 3}))
        for value in ('[]', '{"x": true}', '{"x": -1}', '{"x": NaN}', '3'):
            with patch.dict(os.environ, POLYGEIST_BACKEND_ERROR_BUDGETS=value):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    B.budgets()


if __name__ == '__main__':
    unittest.main()
