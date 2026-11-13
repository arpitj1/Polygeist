"""Dead-copy elimination respects whole-buffer writes, reads and control flow."""
from pathlib import Path
import sys
import unittest

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / 'lib_v2')]
import lower_v2 as L

TEMPLATE = '''module {
  func.func @different_algorithm(%x: memref<8xf32>, %y: memref<8xf32>, %old: memref<8xf32>) {
    %n = arith.constant 8 : i32
    %a = memref.alloc() : memref<8xf32>
    memref.copy %old, %a : memref<8xf32> to memref<8xf32>
    %v = memref.cast %a : memref<8xf32> to memref<?xf32>
    %vx = memref.cast %x : memref<8xf32> to memref<?xf32>
    %vy = memref.cast %y : memref<8xf32> to memref<?xf32>
    kernel.launch @aten_mul_f32(%n, %vx, %vy, %v) : (i32, memref<?xf32>, memref<?xf32>, memref<?xf32>) -> ()
    memref.copy %a, %old : memref<8xf32> to memref<8xf32>
    return
  }
}
'''


class OutputInitialization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calls, cls.specs, errors = L.A.build_table(['aten_mul_f32'])
        assert not errors, errors

    def drop(self, text):
        return L.drop_overwritten_materializations(text, self.specs, {}, self.calls)

    def test_complete_overwrite_removes_only_initial_copy(self):
        text, count = self.drop(TEMPLATE)
        self.assertEqual(count, 1)
        self.assertNotIn('memref.copy %old, %a', text)
        self.assertIn('memref.copy %a, %old', text)

    def test_prefix_and_dynamic_extent_keep_initialization(self):
        for constant in ['4', '0', '-1']:
            with self.subTest(constant=constant):
                self.assertEqual(self.drop(TEMPLATE.replace('constant 8 :', 'constant '+constant+' :'))[1], 0)
        self.assertEqual(self.drop(TEMPLATE.replace('%n = arith.constant 8 : i32', '%n = func.call @unknown() : () -> i32'))[1], 0)

    def test_read_alias_and_escape_keep_initialization(self):
        self.assertEqual(self.drop(TEMPLATE.replace('(%n, %vx, %vy, %v)', '(%n, %v, %vy, %v)'))[1], 0)
        read = '    func.call @observe(%a) : (memref<8xf32>) -> ()\n'
        self.assertEqual(self.drop(TEMPLATE.replace('    kernel.launch', read+'    kernel.launch'))[1], 0)

    def test_conditional_call_keeps_initialization(self):
        text = TEMPLATE.replace('    kernel.launch', '    scf.if %condition {\n    kernel.launch')
        text = text.replace('    memref.copy %a,', '    }\n    memref.copy %a,')
        self.assertEqual(self.drop(text)[1], 0)

    def test_partial_view_and_external_buffer_are_not_private(self):
        text = TEMPLATE.replace('%v = memref.cast %a : memref<8xf32> to memref<?xf32>',
            '%v = memref.subview %a[0] [8] [1] : memref<8xf32> to memref<8xf32, strided<[1]>>')
        self.assertEqual(self.drop(text)[1], 0)
        self.assertEqual(self.drop(TEMPLATE.replace('%a = memref.alloc() : memref<8xf32>',
            '%a = memref.cast %old : memref<8xf32> to memref<8xf32>'))[1], 0)

    def test_missing_validated_contract_keeps_copy(self):
        self.assertEqual(L.drop_overwritten_materializations(TEMPLATE, self.specs, {}, {})[1], 0)

    def test_plain_vendor_api_uses_imported_memory_effect(self):
        text=TEMPLATE.replace('aten_mul_f32','openblas_scopy_f32').replace(
            '(%n, %vx, %vy, %v)','(%n, %vx, %v)').replace(
            '(i32, memref<?xf32>, memref<?xf32>, memref<?xf32>)',
            '(i32, memref<?xf32>, memref<?xf32>)')
        calls,specs,errors=L.A.build_table(['openblas_scopy_f32'])
        self.assertFalse(errors)
        self.assertEqual(L.drop_overwritten_materializations(text,specs,{},calls)[1],1)
        self.assertEqual(L.drop_overwritten_materializations(
            text.replace('constant 8 :','constant 4 :'),specs,{},calls)[1],0)
        self.assertEqual(L.drop_overwritten_materializations(
            text.replace('(%n, %vx, %v)','(%n, %v, %v)'),specs,{},calls)[1],0)


if __name__ == '__main__':
    unittest.main()
