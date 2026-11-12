"""A dominating contiguous view may provide proved operand coordinates."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from region_match.mlir_parse import parse_module
from region_match.symexec import execute
from region_match.equations import build_system
from region_match.matcher import enumerate_bindings, compare_systems

SOURCE = '''module {
 func.func @view(%x: memref<3x4xf32>, %out: memref<12xf32>) {
  %v = memref.reinterpret_cast %out to offset: [0], sizes: [3, 4], strides: [4, 1] : memref<12xf32> to memref<3x4xf32>
  affine.for %r = 0 to 3 {
   affine.for %c = 0 to 4 {
    %xv = affine.load %x[%r, %c] : memref<3x4xf32>
    affine.store %xv, %v[%r, %c] : memref<3x4xf32>
   }
  }
  return
 }
}'''


def program(source):
    functions, aliases = parse_module(source)
    return execute(functions[0], aliases, coordinate_views=True)


def system(p):
    return build_system(p, {i for i, kind in enumerate(p.top_kinds)
                            if kind in ('work', 'writeback')})


class RegionCoordinateViews(unittest.TestCase):
    def test_existing_view_preserves_addresses_and_original_identity(self):
        p = program(SOURCE)
        self.assertEqual(p.arg_buffers['%out'].name, '%out')
        self.assertEqual(p.arg_buffers['%out'].rank, 2)
        self.assertEqual(p.region_views['%out']['ssa'], '%v')
        self.assertEqual(p.region_views['%out']['defined_at'], 0)
        self.assertEqual(p.region_views['%out']['base_type'], 'memref<12xf32>')
        reference = SOURCE.replace('%out: memref<12xf32>', '%out: memref<3x4xf32>')
        reference = '\n'.join(line for line in reference.splitlines()
                              if '%v = memref.reinterpret_cast' not in line)
        reference = reference.replace('%v[%r, %c]', '%out[%r, %c]')
        a, b = system(p), system(program(reference))
        self.assertTrue(any(compare_systems(a, b, binding).ok
                            for binding in enumerate_bindings(a, b, zero_cost_only=True)))

    def test_shifted_or_strided_view_keeps_original_coordinates(self):
        for text in (SOURCE.replace('offset: [0]', 'offset: [1]'),
                     SOURCE.replace('strides: [4, 1]', 'strides: [8, 2]')):
            self.assertEqual(program(text).region_views, {})

    def test_unproved_flat_access_does_not_use_smaller_view(self):
        text = SOURCE.replace('%v[%r, %c] : memref<3x4xf32>',
                              '%out[%r * 4 + %c + 1] : memref<12xf32>')
        self.assertEqual(program(text).region_views, {})
        text = SOURCE.replace('sizes: [3, 4]', 'sizes: [2, 4]')
        self.assertEqual(program(text).region_views, {})

    def test_view_must_dominate_every_access(self):
        text = SOURCE.replace('  %v = memref.reinterpret_cast',
            '  %seed = arith.constant 0.0 : f32\n'
            '  affine.store %seed, %out[0] : memref<12xf32>\n'
            '  %v = memref.reinterpret_cast')
        self.assertEqual(program(text).region_views, {})

    def test_tensor_writeback_path_is_not_normalized(self):
        text = '''module {
 func.func @tensor_view(%out: memref<12xf32>) {
  %v = memref.reinterpret_cast %out to offset: [0], sizes: [3, 4], strides: [4, 1] : memref<12xf32> to memref<3x4xf32>
  %t = bufferization.to_tensor %v : memref<3x4xf32>
  %zero = arith.constant 0 : index
  %seed = arith.constant 1.0 : f32
  %changed = tensor.insert %seed into %t[%zero, %zero] : tensor<3x4xf32>
  %m = bufferization.to_memref %changed : memref<3x4xf32>
  memref.copy %m, %v : memref<3x4xf32> to memref<3x4xf32>
  return
 }
}'''
        self.assertEqual(program(text).region_views, {})


BLOCK_INPUT = '''module {
 func.func @blocks(%col: memref<3xi32>, %x: memref<?xf32>, %out: memref<3x4xf32>) {
  affine.for %p = 0 to 3 {
   %column = affine.load %col[%p] : memref<3xi32>
   %outer = arith.index_cast %column : i32 to index
   affine.for %j = 0 to 4 {
    %v = affine.load %x[%outer * 4 + %j] : memref<?xf32>
    affine.store %v, %out[%p, %j] : memref<3x4xf32>
   }
  }
  return
 }
}'''


class CompleteBlockInputViews(unittest.TestCase):
    def test_full_indirect_blocks_get_derived_capacity_view(self):
        for width in (2, 3, 5, 7):
            with self.subTest(width=width):
                source = BLOCK_INPUT.replace('4', str(width))
                p = program(source)
                view = p.region_views['%x']
                self.assertTrue(view['generated'])
                self.assertEqual(view['inner_extent'], width)
                self.assertEqual(p.params['pviewdim%x_0'], 'viewdim(%x,0)')
                self.assertEqual(p.params['pdim%x_0'], 'dim(%x,0)')
                # The derived view leaves an incomplete unused capacity tail.
                self.assertEqual(view['outer_dimension'], 'floor(source_capacity / inner_extent)')
                reference = source.replace('%x: memref<?xf32>',
                                            f'%x: memref<?x{width}xf32>')
                reference = reference.replace(f'%x[%outer * {width} + %j] : memref<?xf32>',
                                              f'%x[%outer, %j] : memref<?x{width}xf32>')
                a, b = system(p), system(program(reference))
                self.assertTrue(any(compare_systems(a, b, binding).ok
                                    for binding in enumerate_bindings(a, b, zero_cost_only=True)))

    def test_partial_blocks_and_extra_scalar_reads_remain_flat(self):
        partial = BLOCK_INPUT.replace('%j = 0 to 4', '%j = 0 to 3')
        self.assertNotIn('%x', program(partial).region_views)
        extra = BLOCK_INPUT.replace('  return',
            '  %first = affine.load %x[0] : memref<?xf32>\n'
            '  affine.store %first, %out[0, 0] : memref<3x4xf32>\n  return')
        self.assertNotIn('%x', program(extra).region_views)

    def test_outer_index_cannot_depend_on_inner_residue(self):
        text = BLOCK_INPUT.replace(
            '   %column = affine.load %col[%p] : memref<3xi32>\n'
            '   %outer = arith.index_cast %column : i32 to index\n', '')
        text = text.replace('   affine.for %j = 0 to 4 {',
            '   affine.for %j = 0 to 4 {\n'
            '    %column = affine.load %col[%j] : memref<3xi32>\n'
            '    %outer = arith.index_cast %column : i32 to index')
        self.assertNotIn('%x', program(text).region_views)

    def test_read_write_operand_is_not_reinterpreted(self):
        text = BLOCK_INPUT.replace('    affine.store %v, %out[%p, %j] : memref<3x4xf32>',
            '    affine.store %v, %x[%outer * 4 + %j] : memref<?xf32>')
        self.assertNotIn('%x', program(text).region_views)

    def test_contiguous_vector_and_default_executor_keep_original_rank(self):
        plain = BLOCK_INPUT.replace('%x[%outer * 4 + %j]', '%x[%j]')
        self.assertNotIn('%x', program(plain).region_views)
        functions, aliases = parse_module(BLOCK_INPUT)
        self.assertEqual(execute(functions[0], aliases).region_views, {})

    def test_emitter_preserves_physical_and_derived_dimensions(self):
        from region_match.matcher import _Emit
        p = program(BLOCK_INPUT)
        emitter = _Emit(p, 'viewcheck')
        emitter.prepare({'%col': None, '%x': None, '%out': None}, {'%out'})
        generated = emitter.mem('%x')
        self.assertNotEqual(generated, '%x')
        emitter.dimension('%x', 0)
        emitter.dimension('%x', 0, logical=True)
        source = '\n'.join(emitter.lines)
        self.assertIn('memref.dim %x,', source)
        self.assertIn('memref.dim ' + generated + ',', source)
        self.assertIn('arith.divui', source)
        self.assertIn('memref.reinterpret_cast %x to offset: [0]', source)
        self.assertIn('strides: [4, 1] : memref<?xf32> to memref<?x4xf32>', source)


if __name__ == '__main__':
    unittest.main()
