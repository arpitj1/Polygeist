"""Exact aliasing requires the validated per-element reference/backend path."""
from pathlib import Path
import copy
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE / 'lib_v2')]
import abi_v2 as A
import runtime_adapter as R
import lower_v2 as L
import mlir_lite as ML


class PointwiseAliasContract(unittest.TestCase):
    def contract(self, spec, policy):
        calls, specs, errors = A.build_table([spec])
        self.assertFalse(errors)
        reference = specs[spec]
        target = copy.deepcopy(A.manifest_entry(spec)[1]['target'])
        target['runtime_adapter']['alias_policy'] = policy
        return R.contract(A, spec, target, reference.ret, reference.args)

    def test_identity_pointwise_body_accepts_exact_alias_contract(self):
        _, _, contract = self.contract('aten_add_f32', 'exact_or_disjoint')
        self.assertEqual(contract['alias_policy'], 'exact_or_disjoint')

    def test_other_runtime_adapters_cannot_assert_alias_safety(self):
        for spec in ['cutlass_rmsnorm_f32', 'aten_tan_f32']:
            with self.subTest(spec=spec), self.assertRaises(A.AbiError):
                self.contract(spec, 'exact_or_disjoint')

    def test_typed_elementwise_reference_can_validate_vendor_alias_contract(self):
        for spec in ['thrust_plus_f32', 'thrust_multiplies_scalar_left_f32',
                     'thrust_minus_scalar_right_i32']:
            with self.subTest(spec=spec):
                _, _, contract = self.contract(spec, 'exact_or_disjoint')
                self.assertTrue(contract['elementwise_exact_alias'])

    def test_typed_alias_contract_rejects_cross_element_access_and_effects(self):
        spec = 'thrust_plus_f32'
        _, specs, errors = A.build_table([spec])
        self.assertFalse(errors)
        reference = specs[spec]
        target = copy.deepcopy(A.manifest_entry(spec)[1]['target'])
        target['runtime_adapter']['alias_policy'] = 'exact_or_disjoint'
        text = (A.SPEC_DIR.parent/'kernel_library_v3.mlir').read_text()
        for mutation in ['shift', 'reduction', 'effect']:
            module = ML.Module(text)
            function = next(f for f in module.funcs if f.name == spec)
            generic = next(op for op in function.ops if op.name == 'linalg.generic')
            if mutation == 'shift':
                generic.maps[0] = 'affine_map<(d0) -> (d0 + 1)>'
            elif mutation == 'reduction':
                generic.iters[:] = ['reduction']
            else:
                function.ops.append(SimpleNamespace(name='func.call', parent=None, results=[]))
            with self.subTest(mutation=mutation), patch.object(R, '_parsed_reference_module', return_value=module):
                with self.assertRaises(A.AbiError):
                    R.contract(A, spec, target, reference.ret, reference.args)

    def test_lowering_derives_alias_capability_from_validated_contract(self):
        calls, specs, errors = A.build_table(['aten_add_f32'])
        self.assertFalse(errors)
        text = ("module {\n  %r = kernel.launch @aten_add_f32(%n, %x, %y, %out) : "
                "(i32, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>\n}\n")
        prepared, _ = L.prepare(text, specs, calls)
        self.assertIn("{polygeist.elementwise_exact_alias}", prepared)
        self.assertEqual(len([m for line in prepared.splitlines()
                              if (m := L.LAUNCH_RE.match(line))]), 1)

    def test_input_cannot_assert_unvalidated_alias_capability(self):
        calls, specs, errors = A.build_table(['aten_mul_f32'])
        self.assertFalse(errors)
        text = ("module {\n  %r = kernel.launch @aten_mul_f32(%n, %x, %y, %out) "
                "{polygeist.elementwise_exact_alias, keep = 1 : i64} : "
                "(i32, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>\n}\n")
        prepared, _ = L.prepare(text, specs, calls)
        self.assertNotIn("polygeist.elementwise_exact_alias", prepared)
        self.assertIn("keep = 1 : i64", prepared)

    def test_partial_overlap_is_not_an_available_policy(self):
        with self.assertRaises(A.AbiError):
            self.contract('aten_add_f32', 'arbitrary_overlap')

    def scale_module(self, inplace=True, keep_original=False):
        destination = '%xmem' if inplace else '%ymem'
        preserve = ('    %old = bufferization.to_memref %x : memref<16xf32>\n'
                    '    memref.copy %old, %ymem : memref<16xf32> to memref<16xf32>\n'
                    if keep_original else '')
        return ('''module {
  func.func @scale(%xmem: memref<16xf32>, %ymem: memref<16xf32>, %alpha: f32) {
    %n = arith.constant 16 : i32
    %x = bufferization.to_tensor %xmem : memref<16xf32>
    %empty = tensor.empty() : tensor<16xf32>
    %r = kernel.launch @thrust_multiplies_scalar_left_f32(%n, %alpha, %x, %empty) : (i32, f32, tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
    %out = bufferization.to_memref %r : memref<16xf32>
''' + preserve + f'    memref.copy %out, {destination} : memref<16xf32> to memref<16xf32>\n'
                '    return\n  }\n}\n')

    def test_readonly_input_is_not_selected_as_destination(self):
        calls, specs, errors = A.build_table(['thrust_multiplies_scalar_left_f32'])
        self.assertFalse(errors)
        prepared, _ = L.prepare(self.scale_module(inplace=False), specs, calls)
        self.assertIn('%n, %alpha, %x, %empty)', prepared)
        self.assertIn('to_tensor %xmem restrict :', prepared)

    def test_external_writable_input_keeps_its_destination(self):
        calls, specs, errors = A.build_table(['thrust_multiplies_scalar_left_f32'])
        self.assertFalse(errors)
        text = self.scale_module().replace('to_tensor %xmem :', 'to_tensor %xmem restrict writable :')
        prepared, _ = L.prepare(text, specs, calls)
        self.assertIn('%n, %alpha, %x, %empty)', prepared)

    def chain_module(self, keep_original=False):
        text = self.scale_module(inplace=False)
        text = text.replace('    %out = bufferization.to_memref %r :',
            '    %second = tensor.empty() : tensor<16xf32>\n'
            '    %r2 = kernel.launch @thrust_multiplies_scalar_left_f32(%n, %alpha, %r, %second) : '
            '(i32, f32, tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>\n'
            '    %out = bufferization.to_memref %r2 :')
        if keep_original:
            text = text.replace('    return',
                '    %old = bufferization.to_memref %r : memref<16xf32>\n'
                '    memref.copy %old, %xmem : memref<16xf32> to memref<16xf32>\n'
                '    return')
        return text

    def test_private_last_use_result_reuses_its_allocation(self):
        calls, specs, errors = A.build_table(['thrust_multiplies_scalar_left_f32'])
        self.assertFalse(errors)
        prepared, _ = L.prepare(self.chain_module(), specs, calls)
        self.assertIn('%n, %alpha, %r, %r)', prepared)
        buffered = L.bufferize(prepared, [])
        self.assertEqual(buffered.count('= memref.alloc('), 1)

    def test_a_live_original_result_keeps_separate_storage(self):
        calls, specs, errors = A.build_table(['thrust_multiplies_scalar_left_f32'])
        self.assertFalse(errors)
        prepared, _ = L.prepare(self.chain_module(keep_original=True), specs, calls)
        self.assertIn('%n, %alpha, %r, %second)', prepared)
        buffered = L.bufferize(prepared, [])
        self.assertEqual(buffered.count('= memref.alloc('), 2)

    def test_shared_empty_destination_does_not_prevent_reuse(self):
        calls, specs, errors = A.build_table(['thrust_multiplies_scalar_left_f32'])
        self.assertFalse(errors)
        text = self.chain_module().replace('%r, %second)', '%r, %empty)')
        prepared, _ = L.prepare(text, specs, calls)
        self.assertIn('%n, %alpha, %r, %r)', prepared)
        buffered = L.bufferize(prepared, [])
        self.assertEqual(buffered.count('= memref.alloc('), 1)


if __name__ == '__main__':
    unittest.main()
