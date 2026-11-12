"""A reference's dynamic dimensions must not erase known application extents."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from region_match.matcher import Binding, _Emit, _LaunchSpec, _launch

class LaunchShapeTests(unittest.TestCase):
    def emit(self, actual, expected, rank=2):
        program = SimpleNamespace(func=SimpleNamespace(args=[('%array', actual)]))
        emitter = _Emit(program, 'test')
        spec = _LaunchSpec('reference', [('%x', expected)], {'%x': 'x'}, {}, {})
        binding = Binding({'%x': ('%array', None, tuple(range(rank)))}, {}, {})
        _launch(emitter, spec, binding)
        return '\n'.join(emitter.lines)

    def test_static_dimensions_are_kept_for_dynamic_reference(self):
        for actual in ['memref<3x7xf32>', 'memref<?x7xf32>']:
            with self.subTest(actual=actual):
                text = self.emit(actual, 'memref<?x?xf32>')
                self.assertNotIn('memref.cast', text)
                self.assertIn(f'kernel.launch @reference(%array) : ({actual})', text)

    def test_existing_static_requirement_still_needs_checked_cast(self):
        text = self.emit('memref<?x7xf32>', 'memref<3x7xf32>')
        self.assertIn('memref.cast %array : memref<?x7xf32> to memref<3x7xf32>', text)

    def test_strided_layout_is_preserved(self):
        actual = 'memref<3x7xf32, strided<[9, 1], offset: 2>>'
        expected = 'memref<?x?xf32, strided<[9, 1], offset: 2>>'
        text = self.emit(actual, expected)
        self.assertNotIn('memref.cast', text)
        self.assertIn(actual, text)

    def test_unused_outer_extent_binds_capacity_not_stride(self):
        program=SimpleNamespace(func=SimpleNamespace(args=[('%array','memref<?xf32, strided<[3], offset: 2>>')]))
        emitter=_Emit(program,'test')
        spec=_LaunchSpec('reference',[('%capacity','i32'),('%x','memref<?xf32>')],
                        {'%capacity':'capacity','%x':'x'},{'capacity':('x',0)},{})
        binding=Binding({'%x':('%array',None,(0,))},{},{})
        description=_launch(emitter,spec,binding)
        self.assertEqual(description['capacity'],'size0(%array)')
        self.assertIn('arith.index_cast %test_md1#2 : index to i32','\n'.join(emitter.lines))

if __name__ == '__main__':
    unittest.main()
