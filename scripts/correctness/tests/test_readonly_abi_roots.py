"""A read-only view must not become a host gather during CUDA ABI lowering."""
import sys
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lib_v2'))
import lower_v2 as L


class ReadonlyAbiRoots(unittest.TestCase):
    def mark(self, operation):
        text = '''module {
  func.func @test(%a: memref<8xf32>, %b: memref<8xf32>) {
    %x = bufferization.to_tensor %a : memref<8xf32>
    %y = bufferization.to_tensor %b : memref<8xf32>
    %v = tensor.cast %x : tensor<8xf32> to tensor<?xf32>
'''+operation+'''
    return
  }
}
'''
        spec = types.SimpleNamespace(args=[types.SimpleNamespace(intent='in'),
                                           types.SimpleNamespace(intent='out')])
        return L.mark_tensor_roots(text, {'copy': spec}, {})

    def test_library_roles_propagate_through_view(self):
        text = self.mark('    %r = kernel.launch @copy(%v, %y) : (tensor<?xf32>, tensor<8xf32>) -> tensor<8xf32>')
        self.assertIn('to_tensor %a restrict :', text)
        self.assertIn('to_tensor %b restrict writable :', text)

    def test_unknown_consumer_retains_write_permission(self):
        text = self.mark('    func.call @opaque(%v) : (tensor<?xf32>) -> ()')
        self.assertIn('to_tensor %a restrict writable :', text)

    def test_a_second_write_use_prevents_readonly_promotion(self):
        text = self.mark('''    %r = kernel.launch @copy(%v, %y) : (tensor<?xf32>, tensor<8xf32>) -> tensor<8xf32>
    %s = kernel.launch @copy(%y, %x) : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>''')
        self.assertIn('to_tensor %a restrict writable :', text)


if __name__ == '__main__':
    unittest.main()
