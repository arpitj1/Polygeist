"""Shape-only casts retain ordered computation and all live consumers."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'scripts/correctness'), str(ROOT/'scripts/correctness/lib_v2')]
import match_v2 as M


class WritebackCastMatch(unittest.TestCase):
    def source(self, shared=False):
        extra = 'call @observe(%cast) : (tensor<?xf32>) -> ()' if shared else ''
        return '''#map = affine_map<(d0) -> (d0)>
module {
func.func private @observe(tensor<?xf32>)
func.func @renamed(%x: tensor<37xf32>, %scale: f32, %output: tensor<?xf32>) -> tensor<?xf32> {
  %n = arith.constant 37 : index
  %view = polygeist.submap(%output, %n) {map = #map} : (tensor<?xf32>, index) -> tensor<37xf32>
  %computed = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%x : tensor<37xf32>) outs(%view : tensor<37xf32>) {
    ^bb0(%a: f32, %b: f32):
      %v = arith.divf %a, %scale : f32
      linalg.yield %v : f32
  } -> tensor<37xf32>
  %cast = tensor.cast %computed : tensor<37xf32> to tensor<?xf32>
  EXTRA
  %result = polygeist.submapInverse(%output, %cast, %n) {map = #map} : (tensor<?xf32>, tensor<?xf32>, index) -> tensor<?xf32>
  return %result : tensor<?xf32>
}}
'''.replace('EXTRA', extra)

    def match(self, text):
        return M.run(text, ROOT/'generic_solver/kernel_library_v3.mlir', region=False,
                     allowed_specs={'thrust_divide_scalar_right_f32'})[:2]

    def test_shape_cast_allows_reference_match_and_dead_producer_removal(self):
        out, matches = self.match(self.source())
        self.assertEqual([m['spec'] for m in matches], ['thrust_divide_scalar_right_f32'])
        self.assertNotIn('linalg.generic', out)
        self.assertNotIn('%cast =', out)
        self.assertIn('%scale', out)

    def test_shared_cast_consumer_keeps_original_producer(self):
        out, matches = self.match(self.source(shared=True))
        self.assertEqual(len(matches), 1)
        self.assertIn('call @observe(%cast)', out)
        self.assertIn('%computed = linalg.generic', out)
        self.assertIn('%cast = tensor.cast %computed', out)

    def test_element_conversion_is_not_a_shape_cast(self):
        text = self.source().replace('tensor<?xf32>', 'tensor<?xf64>')
        text = text.replace('%view = polygeist.submap(%output, %n) {map = #map} : (tensor<?xf64>, index) -> tensor<37xf32>',
            '%view = tensor.empty() : tensor<37xf32>')
        text = text.replace('%cast = tensor.cast %computed : tensor<37xf32> to tensor<?xf64>',
            '%wide = arith.extf %computed : tensor<37xf32> to tensor<37xf64>\n'
            '  %cast = tensor.cast %wide : tensor<37xf64> to tensor<?xf64>')
        _, matches = self.match(text)
        self.assertEqual(matches, [])


if __name__ == '__main__':
    unittest.main()
