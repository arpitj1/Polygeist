"""A previous-iteration input load is usable as an index only with a valid base."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from region_match.mlir_parse import parse_module
from region_match.symexec import Unsupported, execute
from test_ordered_overwrite import affine, evaluate, system


SOURCE = '''module {
 func.func @scatter(%indices: memref<5xi32>, %out: memref<9xi32>) {
  %initial = affine.load %indices[0] : memref<5xi32>
  %result = affine.for %i = 0 to 4 iter_args(%current = %initial) -> (i32) {
   %next = affine.load %indices[%i + 1] : memref<5xi32>
   %index = arith.index_cast %current : i32 to index
   %value = arith.index_cast %i : index to i32
   memref.store %value, %out[%index] : memref<9xi32>
   affine.yield %next : i32
  }
  return
 }
}'''

SPILLED = SOURCE.replace(
    '  %result = affine.for %i = 0 to 4 iter_args(%current = %initial) -> (i32) {',
    '  %state = memref.alloca() : memref<i32>\n'
    '  affine.store %initial, %state[] : memref<i32>\n'
    '  affine.for %i = 0 to 4 {\n'
    '   %current = affine.load %state[] : memref<i32>').replace(
    '   affine.yield %next : i32', '   affine.store %next, %state[] : memref<i32>')


class CarriedInputLoad(unittest.TestCase):
    def test_scatter_preserves_repeated_and_unordered_indices(self):
        for source in (SOURCE, SPILLED):
            equations = system(source).equations
            output = next(eq for eq in equations if eq.arr == '%out')
            for indices in ([4, 1, 4, 2, 0], [0, 0, 0, 0, 0], [8, 7, 6, 5, 4]):
                initial = [10 + i for i in range(9)]
                expected = initial[:]
                for i in range(4):
                    expected[indices[i]] = i
                actual = [evaluate(output.formula, {'v0': i},
                                   {'%indices': indices, '%out': initial})
                          for i in range(9)]
                self.assertEqual(actual, expected)

    def test_spilled_state_load_after_write_is_not_substituted(self):
        text = SPILLED.replace('   affine.store %next, %state[] : memref<i32>',
            '   affine.store %next, %state[] : memref<i32>\n'
            '   %later = affine.load %state[] : memref<i32>')
        functions, aliases = parse_module(text)
        with self.assertRaises(Unsupported):
            execute(functions[0], aliases)

    def test_spilled_state_alias_write_is_not_substituted(self):
        text = SPILLED.replace('  affine.for %i = 0 to 4 {',
            '  %alias = memref.cast %state : memref<i32> to memref<i32>\n'
            '  affine.for %i = 0 to 4 {').replace(
            '   affine.store %next, %state[] : memref<i32>',
            '   affine.store %next, %state[] : memref<i32>\n'
            '   affine.store %initial, %alias[] : memref<i32>')
        functions, aliases = parse_module(text)
        with self.assertRaises(Unsupported):
            execute(functions[0], aliases)

    def test_nonzero_lower_bound_preserves_previous_iteration(self):
        text = SOURCE.replace('%indices[0]', '%indices[2]').replace(
            '%i = 0 to 4', '%i = 2 to 4')
        functions, aliases = parse_module(text)
        program = execute(functions[0], aliases)
        write = next(st for st in program.stmts if st.buf.name == '%out')
        for i in (2, 3):
            self.assertEqual(affine(write.idx[0], {'i1': i},
                                    {'%indices': [4, 1, 7, 6, 0]}),
                             [4, 1, 7, 6, 0][i])

    def test_wrong_initial_value_is_not_replaced(self):
        text = SOURCE.replace('%indices[0]', '%indices[1]')
        functions, aliases = parse_module(text)
        with self.assertRaises(Unsupported):
            execute(functions[0], aliases)

    def test_write_after_load_is_not_treated_as_readonly(self):
        text = SOURCE.replace('   affine.yield %next',
            '   memref.store %value, %indices[%index] : memref<5xi32>\n'
            '   affine.yield %next')
        functions, aliases = parse_module(text)
        with self.assertRaisesRegex(Unsupported, 'written buffer'):
            execute(functions[0], aliases)

    def test_write_between_initial_load_and_loop_is_rejected(self):
        text = SOURCE.replace('  %result = affine.for',
            '  affine.store %initial, %indices[0] : memref<5xi32>\n'
            '  %result = affine.for')
        functions, aliases = parse_module(text)
        with self.assertRaises(Unsupported):
            execute(functions[0], aliases)

    def test_loaded_integer_width_is_preserved_through_index(self):
        source = '''module {
 func.func @roundtrip(%input: memref<1xi32>, %out: memref<1xi32>) {
  %value = affine.load %input[0] : memref<1xi32>
  %index = arith.index_cast %value : i32 to index
  %back = arith.index_cast %index : index to i32
  affine.store %back, %out[0] : memref<1xi32>
  return
 }
}'''
        self.assertEqual(len(system(source).equations), 1)
        wider = source.replace('%input: memref<1xi32>', '%input: memref<1xi64>')
        wider = wider.replace('%input[0] : memref<1xi32>', '%input[0] : memref<1xi64>')
        wider = wider.replace('%value : i32 to index', '%value : i64 to index')
        functions, aliases = parse_module(wider)
        with self.assertRaisesRegex(Unsupported, 'may truncate'):
            execute(functions[0], aliases)


if __name__ == '__main__':
    unittest.main()
