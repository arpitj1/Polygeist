"""The tensor emitter derives geometry from references and rejects unsafe views."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE / 'lib_v2')]
import abi_v2 as A
import gen_v2_runtime as G
import mlir_lite as M
import runtime_adapter as R
import tensor_pointwise as T


class TensorPointwiseReference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.name = 'pytorch_batchnorm_saved_f32'
        cls.calls, cls.specs, errors = A.build_table([cls.name])
        if errors:
            raise AssertionError(errors)
        cls.sa = cls.specs[cls.name]
        cls.target = A.manifest_entry(cls.name)[1]['target']
        cls.shapes = cls.target['runtime_adapter']['arrays']
        cls.text = next(f.text for f in M.Module(
            (A.SPEC_DIR.parent / 'kernel_library_v3.mlir').read_text()).funcs if f.name == cls.name)

    def analyze(self, text=None, name=None, args=None, shapes=None):
        text = self.text if text is None else text
        with patch.object(R, '_parsed_reference_module', return_value=M.Module('module {\n' + text + '\n}')):
            return T.reference(A, name or self.name, args or self.sa.args, shapes or self.shapes)

    def test_geometry_comes_from_the_public_reference(self):
        result = self.analyze()
        self.assertEqual(result['extents'], [['N'], ['C'], ['H', 'W']])
        self.assertEqual(result['input_masks'], [[False, True, False]] * 4 + [[True] * 3])
        graph = result['graph']
        self.assertEqual(graph['inputs'], ['weight', 'bias', 'mean', 'invstd', 'input'])
        self.assertEqual(graph['nodes'], 4)
        self.assertEqual((graph['words'][0] >> 24) & 255, 3)

    def test_names_do_not_select_the_emission(self):
        self.assertEqual(self.analyze(), self.analyze(
            self.text.replace(self.name, 'unrelated_library_name'), 'unrelated_library_name'))

    def test_computation_changes_the_descriptor(self):
        before = self.analyze()['graph']['words']
        after = self.analyze(self.text.replace('arith.subf', 'arith.addf'))['graph']['words']
        self.assertNotEqual(before, after)

    def test_inconsistent_extent_and_inverse_rejected(self):
        for text in (self.text.replace('arith.index_cast %C', 'arith.index_cast %H'),
                     self.text.replace('submapInverse(%output', 'submapInverse(%input'),
                     self.text.replace('"parallel", "parallel"', '"parallel", "reduction"', 1)):
            with self.subTest(text=text[-200:]), self.assertRaises(A.AbiError):
                self.analyze(text)

    def test_shape_mismatch_rejected(self):
        shapes = copy.deepcopy(self.shapes)
        shapes['mean'] = ['H']
        with self.assertRaises(A.AbiError):
            self.analyze(shapes=shapes)

    def test_partial_input_or_output_mutation_rejected(self):
        args = copy.deepcopy(self.sa.args)
        next(a for a in args if a.name == 'mean').intent = 'inout'
        with self.assertRaises(A.AbiError):
            self.analyze(args=args)

    def test_unsupported_vendor_operation_rejected(self):
        with self.assertRaises(A.AbiError):
            self.analyze(self.text.replace('arith.subf', 'arith.divf'))

    def test_generated_abi_keeps_capacity_overlap_and_device_checks(self):
        source = G.gen_cuda(self.calls)
        for token in ('polygeist_v2_overlaps', 'array extent overflow',
                      'array layout or capacity mismatch', 'polygeist_v2_map_to_device',
                      'polygeist_cudnn_tensor_pointwise_f32', 'pv2_tensor.input_dims[0][0] = 1'):
            self.assertIn(token, source)
        self.assertEqual(G.gen_cpu(self.calls), '')

    def test_backend_and_unbounded_arguments_rejected(self):
        target = copy.deepcopy(self.target)
        target['backends'] = ['cpu']
        with self.assertRaises(A.AbiError):
            R.contract(A, self.name, target, self.sa.ret, self.sa.args)
        target = copy.deepcopy(self.target)
        target['runtime_adapter']['bounds']['N']['min'] = 0
        with self.assertRaises(A.AbiError):
            R.contract(A, self.name, target, self.sa.ret, self.sa.args)


if __name__ == '__main__':
    unittest.main()
