"""Typed integer indices and pure scalar branches retain full semantics."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).parent)]
from test_ordered_overwrite import system
from test_unsigned_histogram_fold import HISTOGRAM
from region_match.matcher import enumerate_bindings, compare_systems
from region_match.symexec import Unsupported, _elem_of


def matches(left, right):
    a, b = system(left), system(right)
    return any(compare_systems(a, b, binding).ok
               for binding in enumerate_bindings(a, b, zero_cost_only=True))


class IntegerImages(unittest.TestCase):
    def setUp(self):
        policy = patch.dict('os.environ', POLYGEIST_NUMERIC_MODEL='strict')
        policy.start(); self.addCleanup(policy.stop)

    def histogram(self, shift=8, extension='extui'):
        return HISTOGRAM.replace('xi8>', 'xi16>').replace(
            '%index = arith.index_castui %byte : i8 to index',
            '%wide = arith.' + extension + ' %byte : i16 to i32\n'
            '    %amount = arith.constant ' + str(shift) + ' : i32\n'
            '    %bin = arith.shrsi %wide, %amount : i32\n'
            '    %index = arith.index_castui %bin : i32 to index')

    def test_i16_element_type(self):
        self.assertEqual(_elem_of('memref<?x?xi16>'), 'i16')
        self.assertEqual(_elem_of('memref<4xi16, strided<[1], offset: ?>>'), 'i16')

    def test_shifted_unsigned_indices(self):
        source = self.histogram()
        self.assertTrue(matches(source, source))
        self.assertFalse(matches(source, self.histogram(9)))
        self.assertFalse(matches(source, source.replace('%count, %one', '%count, %zero')))
        with self.assertRaises(Unsupported):
            system(self.histogram(extension='extsi'))
        with self.assertRaises(Unsupported):
            system(self.histogram(32))

    def test_pure_integer_if_matches_select(self):
        prefix = '''module {
 func.func @f(%x: memref<1xi16>, %out: memref<1xi32>) {
  %zero = arith.constant 0 : index
  %low = arith.constant 5 : i32
  %xv = memref.load %x[%zero] : memref<1xi16>
  %v = arith.extsi %xv : i16 to i32
  %c = arith.cmpi slt, %v, %low : i32
'''
        suffix = '''  memref.store %result, %out[%zero] : memref<1xi32>
  return
 }
}'''
        branch = '''  %result = scf.if %c -> (i32) {
   scf.yield %low : i32
  } else {
   scf.yield %v : i32
  }
'''
        select = '  %result = arith.select %c, %low, %v : i32\n'
        self.assertTrue(matches(prefix+branch+suffix, prefix+select+suffix))
        self.assertFalse(matches(prefix+branch+suffix,
                                 prefix+select.replace('%low, %v', '%v, %low')+suffix))
        effectful = branch.replace('scf.yield %low',
                                   'memref.store %low, %out[%zero] : memref<1xi32>\n   scf.yield %low')
        with self.assertRaisesRegex(Unsupported, 'effectful'):
            system(prefix+effectful+suffix)

if __name__ == '__main__': unittest.main()
