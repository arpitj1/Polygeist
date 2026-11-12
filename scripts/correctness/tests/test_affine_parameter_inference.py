"""Infer coupled library extents without relaxing the domain/value proof."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from region_match.mlir_parse import parse_module
from region_match.symexec import execute
from region_match.equations import build_system
from region_match.matcher import enumerate_bindings, compare_systems

SOURCE = '''module {
 func.func @window(%n: index, %k: index, %x: memref<?xf64>, %w: memref<?xf64>, %out: memref<?xf64>) {
  %one = arith.constant 1 : index
  %zero = arith.constant 0.0 : f64
  %delta = arith.subi %n, %k : index
  %length = arith.addi %delta, %one : index
  affine.for %i = 0 to %length {
   affine.store %zero, %out[%i] : memref<?xf64>
   affine.for %j = 0 to %k {
    %a = affine.load %x[%i + %j] : memref<?xf64>
    %b = affine.load %w[%j] : memref<?xf64>
    %old = affine.load %out[%i] : memref<?xf64>
    %product = arith.mulf %a, %b : f64
    %value = arith.addf %old, %product : f64
    affine.store %value, %out[%i] : memref<?xf64>
   }
  }
  return
 }
}'''


def system(source, fixed=None):
    functions, aliases = parse_module(source)
    program = execute(functions[0], aliases, fixed)
    return build_system(program, {i for i,k in enumerate(program.top_kinds)
                                  if k in ('work','writeback')})


class AffineParameterInference(unittest.TestCase):
    def test_singleton_extent_from_canonical_equality(self):
        template = '''module {
 func.func @copy(%n: index, %x: memref<?xf32>, %out: memref<?xf32>) {
  affine.for %i = 0 to %n {
   %value = affine.load %x[%i] : memref<?xf32>
   affine.store %value, %out[%i] : memref<?xf32>
  }
  return
 }
}'''
        application = '''module {
 func.func @single(%x: memref<?xf32>, %out: memref<?xf32>) {
  %value = affine.load %x[0] : memref<?xf32>
  affine.store %value, %out[0] : memref<?xf32>
  return
 }
}'''
        spec = system(template)
        app = system(application)
        matches = [b for b in enumerate_bindings(app, spec, zero_cost_only=True)
                   if compare_systems(app, spec, b).ok]
        self.assertTrue(matches)
        self.assertTrue(all(b.params['p%n'].const == 1 for b in matches))
        # Inferring an extent must not authorize a shifted read or wider write.
        wrong = system(application.replace('%x[0]', '%x[1]'))
        self.assertFalse(any(compare_systems(wrong,spec,b).ok for b in
                             enumerate_bindings(wrong,spec,zero_cost_only=True)))
        for binding in matches:
            binding.params['p%n'] = type(binding.params['p%n']).cst(2)
            self.assertFalse(compare_systems(app, spec, binding).ok)

    def test_coupled_lengths_and_wrong_index(self):
        with patch.dict('os.environ', POLYGEIST_NUMERIC_MODEL='relaxed'):
            spec = system(SOURCE)
            for n,k in [(11,3),(19,5)]:
                app = system(SOURCE, {'%n':n,'%k':k})
                matches = [b for b in enumerate_bindings(app,spec,zero_cost_only=True)
                           if compare_systems(app,spec,b).ok]
                self.assertTrue(matches)
                self.assertTrue(any(b.params['p%n'].const == n and
                                    b.params['p%k'].const == k for b in matches))
                wrong = system(SOURCE.replace('%x[%i + %j]', '%x[%i + %j + 1]'))
                self.assertFalse(any(compare_systems(app,wrong,b).ok for b in
                                     enumerate_bindings(app,wrong,zero_cost_only=True)))


if __name__ == '__main__':
    unittest.main()
