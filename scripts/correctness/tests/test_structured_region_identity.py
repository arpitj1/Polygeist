"""Whole-region identity must retain recurrences, predicates and scalar types."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'scripts/correctness'), str(ROOT/'scripts/correctness/lib_v2')]
from region_match.structural import match_regions


REFERENCE = '''module {
func.func @reference(%n: i32, %x: memref<?xf32>, %y: memref<?xf32>) {
 %c0 = arith.constant 0 : index
 %c1 = arith.constant 1 : index
 %one = arith.constant 1.0 : f32
 %size = arith.index_cast %n : i32 to index
 affine.for %i = 0 to %size {
   %v = memref.load %x[%i] : memref<?xf32>
   %r:2 = scf.while (%a = %v, %j = %c0) : (f32,index) -> (f32,index) {
     %keep = arith.cmpi slt, %j, %size : index
     scf.condition(%keep) %a, %j : f32,index
   } do {
   ^bb0(%a: f32, %j: index):
     %next = arith.addf %a, %one : f32
     %k = arith.addi %j, %c1 : index
     scf.yield %next, %k : f32,index
   }
   memref.store %r#0, %y[%i] : memref<?xf32>
 }
 return
}
}'''
APPLICATION = REFERENCE.replace('@reference(%n: i32, ', '@application(').replace(
    ' %c0 =', ' %n = arith.constant 7 : i32\n %c0 =')


class StructuredRegionIdentity(unittest.TestCase):
    def match(self, application=APPLICATION, reference=REFERENCE, validated=True, args=None):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);(d/'manifest').mkdir();(d/'raised').mkdir()
            (d/'manifest/spec.json').write_text(json.dumps({'specs':[{
                'name':'reference','args':args or ['n','x','y'],
                'assumptions':['positive n <= INT_MAX'], 'deviations':[]}]}))
            (d/'import_manifest.json').write_text(json.dumps({'specifications':{
                'reference':{'numerical_validation':{'defn':'EXACT' if validated else 'UNSOUND'}}}}))
            (d/'raised/reference_debuf.mlir').write_text(reference)
            return match_regions(application,str(d),{'reference'})

    def test_exact_nested_recurrence(self):
        matches=self.match()
        self.assertEqual(len(matches),1)
        self.assertEqual(matches[0].binding,{'n':'literal 7','x':'%x','y':'%y'})
        self.assertIn('exact structured-region identity',matches[0].evidence)

    def test_changed_recurrence_rejected(self):
        self.assertFalse(self.match(APPLICATION.replace('arith.addf','arith.subf')))

    def test_runtime_scalar_with_different_argument_order(self):
        reference = REFERENCE.replace('%n: i32,', '%n: i32, %step: f32,').replace(
            'arith.addf %a, %one', 'arith.addf %a, %step')
        application = APPLICATION.replace('%x: memref<?xf32>,',
            '%x: memref<?xf32>, %increment: f32,').replace(
            'arith.addf %a, %one', 'arith.addf %a, %increment')
        result = self.match(application, reference, args=['n','step','x','y'])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].binding,
                         {'n':'literal 7','step':'%increment','x':'%x','y':'%y'})
        self.assertIn('%increment', result[0].launch_text)

    def test_runtime_scalar_cannot_be_replaced_with_fixed_constant(self):
        reference = REFERENCE.replace('%n: i32,', '%n: i32, %step: f32,').replace(
            'arith.addf %a, %one', 'arith.addf %a, %step')
        application = APPLICATION.replace('%x: memref<?xf32>,',
            '%x: memref<?xf32>, %increment: f32,')
        self.assertFalse(self.match(application, reference, args=['n','step','x','y']))

    def test_runtime_integer_distinguished_from_extent(self):
        reference = REFERENCE.replace('%n: i32,', '%limit: i32, %n: i32,').replace(
            ' affine.for', ' %bound = arith.index_cast %limit : i32 to index\n affine.for').replace(
            'cmpi slt, %j, %size', 'cmpi slt, %j, %bound')
        application = APPLICATION.replace('@application(', '@application(%stop: i32,').replace(
            ' affine.for', ' %bound = arith.index_cast %stop : i32 to index\n affine.for').replace(
            'cmpi slt, %j, %size', 'cmpi slt, %j, %bound')
        result = self.match(application, reference, args=['limit','n','x','y'])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].binding['limit'], '%stop')
        self.assertEqual(result[0].binding['n'], 'literal 7')

    def test_constant_order_is_irrelevant(self):
        first=' %c0 = arith.constant 0 : index\n %c1 = arith.constant 1 : index'
        second=' %c1 = arith.constant 1 : index\n %c0 = arith.constant 0 : index'
        self.assertEqual(len(self.match(APPLICATION.replace(first,second))),1)

    def test_changed_termination_rejected(self):
        self.assertFalse(self.match(APPLICATION.replace('cmpi slt','cmpi sle')))

    def test_changed_memory_target_rejected(self):
        self.assertFalse(self.match(APPLICATION.replace('store %r#0, %y','store %r#0, %x')))

    def test_typed_arithmetic_rejected(self):
        self.assertFalse(self.match(APPLICATION.replace('f32','f64')))

    def test_unvalidated_reference_rejected(self):
        self.assertFalse(self.match(validated=False))

    def test_opaque_helper_rejected(self):
        def helper(text):
            return text.replace('module {','module {\nfunc.func private @user_helper(f32) -> f32').replace(
                'arith.addf %a, %one : f32','func.call @user_helper(%a) : (f32) -> f32')
        self.assertFalse(self.match(helper(APPLICATION),helper(REFERENCE)))

    def test_fastmath_is_part_of_identity(self):
        self.assertFalse(self.match(APPLICATION.replace('arith.addf %a, %one :',
                                                       'arith.addf %a, %one fastmath<fast> :')))

    @staticmethod
    def with_coefficients(text, value='1.0', constant=True):
        qualifier = 'constant ' if constant else ''
        return text.replace('module {', 'module {\n memref.global "private" '+
            qualifier+'@coefficients : memref<8xf32> = dense<'+value+'>').replace(
            ' %one = arith.constant 1.0 : f32',
            ' %table = memref.get_global @coefficients : memref<8xf32>').replace(
            '     %next = arith.addf %a, %one',
            '     %one = memref.load %table[%j] : memref<8xf32>\n'
            '     %next = arith.addf %a, %one')

    def test_exact_immutable_coefficient_table(self):
        self.assertEqual(len(self.match(self.with_coefficients(APPLICATION),
                                      self.with_coefficients(REFERENCE))), 1)

    def test_changed_coefficient_table_rejected(self):
        self.assertFalse(self.match(self.with_coefficients(APPLICATION, '2.0'),
                                    self.with_coefficients(REFERENCE)))

    def test_mutable_coefficient_table_rejected(self):
        self.assertFalse(self.match(self.with_coefficients(APPLICATION, constant=False),
                                    self.with_coefficients(REFERENCE, constant=False)))

    def test_private_table_address_is_not_an_equivalence(self):
        def expose_address(text):
            return self.with_coefficients(text).replace(' return', '''
 %address = memref.extract_aligned_pointer_as_index %table : memref<8xf32> -> index
 %integer = arith.index_cast %address : index to i64
 %value = arith.uitofp %integer : i64 to f32
 memref.store %value, %y[%c0] : memref<?xf32>
 return''')
        self.assertFalse(self.match(expose_address(APPLICATION), expose_address(REFERENCE)))


if __name__=='__main__':unittest.main()
