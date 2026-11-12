"""Check the BSR C import against the actual public CUSP implementation."""
import ctypes
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]


@unittest.skipUnless(shutil.which('cc') and shutil.which('c++'), 'requires C/C++ compilers')
class CuspBsrReference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='cusp-bsr-reference-')
        dest = Path(cls.temp.name)
        source = dest / 'upstream.cpp'
        source.write_text('''#include <cassert>
#include "cusp/detail/host/reference/bsr.h"
extern "C" void upstream(int rows, int cols, int block_dim,
    const int *offsets, const int *indices, const float *values,
    const float *x, float *y) {
  bsr_matvec<int, float>(rows, cols, block_dim, block_dim,
                         offsets, indices, values, x, y);
}
''')
        subprocess.run(['c++', '-std=c++17', '-O0', '-ffp-contract=off', '-fPIC',
                        '-shared', '-I', str(ROOT / 'third_party/AMGX/include'),
                        str(source), '-o', str(dest / 'upstream.so')],
                       check=True, capture_output=True)
        subprocess.run(['cc', '-std=c99', '-O0', '-ffp-contract=off', '-fPIC',
                        '-shared', '-DNETLIB_SPEC=imported',
                        str(ROOT / 'generic_solver/lib_v3/ref_c/cusp_bsr_matvec.c'),
                        '-o', str(dest / 'imported.so')], check=True, capture_output=True)
        cls.upstream_lib = ctypes.CDLL(str(dest / 'upstream.so'))
        cls.imported_lib = ctypes.CDLL(str(dest / 'imported.so'))
        cls.upstream = cls.upstream_lib.upstream
        cls.imported = cls.imported_lib.imported
        i, ip, fp = ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float)
        cls.upstream.argtypes = [i, i, i, ip, ip, fp, fp, fp]
        cls.imported.argtypes = [i, i, i, ip, i, ip, i, fp, i, fp, i, fp]
        cls.upstream.restype = cls.imported.restype = None

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_dynamic_block_sizes_empty_rows_and_output_capacity(self):
        rows, cols, nnz = 3, 4, 5
        offsets = (ctypes.c_int * 4)(0, 2, 2, 5)
        indices = (ctypes.c_int * nnz)(0, 2, 3, 1, 0)
        for block_dim in (1, 2, 3, 5, 7):
            with self.subTest(block_dim=block_dim):
                values = (ctypes.c_float * (nnz * block_dim * block_dim))(
                    *[((k * 17) % 37 - 18) / 16 for k in range(nnz * block_dim * block_dim)])
                x = (ctypes.c_float * (cols * block_dim))(
                    *[((k * 11) % 29 - 14) / 8 for k in range(cols * block_dim)])
                before = (bytes(offsets), bytes(indices), bytes(values), bytes(x))
                # CUSP accumulates; beta=0 supplies a zeroed output. A spare
                # block verifies that the imported wrapper preserves capacity
                # outside the mathematical output domain.
                reference = (ctypes.c_float * ((rows + 1) * block_dim))(
                    *([0.0] * (rows * block_dim) + [123.5] * block_dim))
                imported = (ctypes.c_float * ((rows + 1) * block_dim))(
                    *([42.0] * (rows * block_dim) + [123.5] * block_dim))
                self.upstream(rows, cols, block_dim, offsets, indices, values, x, reference)
                self.imported(rows, block_dim, 4, offsets, nnz, indices,
                              nnz, values, cols, x, rows + 1, imported)
                self.assertEqual(bytes(imported), bytes(reference))
                self.assertEqual(list(imported)[block_dim:2 * block_dim], [0.0] * block_dim)
                self.assertEqual(before, (bytes(offsets), bytes(indices), bytes(values), bytes(x)))


if __name__ == '__main__':
    unittest.main()
