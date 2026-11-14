"""Scalar branch normalization must preserve guards and every result."""
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[3]
OPT = ROOT / 'build/bin/polygeist-opt'


def module(body, kind='f32', extra=''):
    return '''module {
func.func @check(%input: memref<4x'''+kind+'''>, %output: memref<4x'''+kind+'''>, %condition: i1, %j: index'''+extra+''') {
 linalg.generic {indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
 iterator_types = ["parallel"]} ins(%input : memref<4x'''+kind+'''>) outs(%output : memref<4x'''+kind+'''>) {
 ^bb0(%x: '''+kind+''', %old: '''+kind+'''):
 '''+body+'''
 }
 return
}
}'''


class LinalgScalarBranches(unittest.TestCase):
    def normalize(self, body, kind='f32', extra=''):
        p = subprocess.run([str(OPT), '--raise-affine-to-linalg'],
            input=module(body, kind, extra), capture_output=True, text=True, timeout=60)
        self.assertEqual(p.returncode, 0, p.stderr)
        return p.stdout

    def test_multiple_pure_results(self):
        text = self.normalize('''
 %r:2 = scf.if %condition -> (f32, f32) {
   %s = math.sin %x : f32
   %t = arith.mulf %x, %x : f32
   scf.yield %s, %t : f32, f32
 } else {
   scf.yield %x, %old : f32, f32
 }
 %result = arith.addf %r#0, %r#1 : f32
 linalg.yield %result : f32''')
        self.assertNotIn('scf.if', text)
        self.assertEqual(text.count('arith.select'), 2)
        self.assertIn('math.sin', text)

    def test_guarded_integer_division_stays_guarded(self):
        text = self.normalize('''
 %r = scf.if %condition -> (i32) {
   %d = arith.divsi %x, %old : i32
   scf.yield %d : i32
 } else {
   scf.yield %x : i32
 }
 linalg.yield %r : i32''', 'i32')
        self.assertIn('scf.if', text)
        self.assertIn('arith.divsi', text)

    def test_guarded_read_is_not_speculated(self):
        text = self.normalize('''
 %r = scf.if %condition -> (f32) {
   %v = memref.load %input[%j] : memref<4xf32>
   scf.yield %v : f32
 } else {
   scf.yield %old : f32
 }
 linalg.yield %r : f32''')
        self.assertIn('scf.if', text)
        self.assertIn('memref.load', text)

    def test_intervening_write_blocks_read_reuse(self):
        text = self.normalize('''
 %before = memref.load %other[%j] : memref<4xf32>
 memref.store %x, %other[%j] : memref<4xf32>
 %r = scf.if %condition -> (f32) {
   %after = memref.load %other[%j] : memref<4xf32>
   scf.yield %after : f32
 } else {
   scf.yield %old : f32
 }
 %sum = arith.addf %before, %r : f32
 linalg.yield %sum : f32''', extra=', %other: memref<4xf32>')
        self.assertIn('scf.if', text)
        self.assertEqual(text.count('memref.load'), 2)


if __name__ == '__main__':
    unittest.main()
