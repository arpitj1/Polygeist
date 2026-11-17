"""Cached ABI preparation must follow changes to current reference inputs."""
import dataclasses
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lib_v2'))
import abi_cache as C
import abi_v2 as A
import gen_v2_runtime as G


class PersistentABI(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        suite = self.root / 'generic_solver/lib_v3'
        for folder in ('manifest', 'raised'):
            (suite / folder).mkdir(parents=True)
        (suite / 'manifest/example.json').write_text('{"specs":[{"name":"example"}]}')
        (suite / 'raised/example.c').write_text('#include "types.h"\nvoid example(TYPE a) {}\n')
        self.header = suite / 'raised/types.h'
        self.header.write_text('#define TYPE float\n')
        (suite / 'raised/example.mlir').write_text('reference')
        (self.root / 'runtime').mkdir()
        (self.root / 'runtime/helper.h').write_text('header')
        (self.root / 'scripts/correctness').mkdir(parents=True)
        (self.root / 'scripts/correctness/helper.py').write_text('parser')
        (self.root / 'headers').mkdir()
        (self.root / 'headers/vendor.h').write_text('vendor')
        (self.root / 'netlib/SRC').mkdir(parents=True)
        (self.root / 'netlib/SRC/entry.f').write_text('fortran')
        self.cache = self.root / 'cache'
        self.patches = [
            patch.dict(os.environ, POLYGEIST_ABI_CACHE_DIR=str(self.cache),
                       POLYGEIST_ABI_NO_CACHE='0', POLYGEIST_ABI_CACHE_LOG=''),
            patch.multiple(A, REPO=self.root, SPEC_DIR=suite, CLANG=shutil.which('cc'),
                           CUDA_HEADERS=self.root / 'headers', NETLIB=self.root / 'netlib',
                           HEADER_FAMILIES={'test': ['vendor.h']}),
            patch.object(A, 'bind_spec', side_effect=self.bind),
            patch.object(A, '_build_call', side_effect=self.call),
        ]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    def bind(self, name):
        ret, args = A.spec_c_prototype(name)
        return A.SpecAbi(name, 'example', ret, args, {'a': ('arg', 0)}, {})

    def call(self, name, specs):
        return A.CallAbi(name, 'test', name, [A.AbiParam('a', 'scalar', elem=specs[0].args[0].ctype)], [])

    def test_hit_preserves_nested_types_and_is_independent(self):
        first = A.build_table()
        second = A.build_table()
        self.assertEqual(first, second)
        self.assertEqual(A._build_call.call_count, 1)
        first[1]['example'].binding['a'] = ('arg', 10)
        self.assertEqual(A.build_table()[1]['example'].binding['a'], ('arg', 0))

    def test_transitive_header_change_rechecks_prototype(self):
        first = A.build_table()
        self.header.write_text('#define TYPE double\n')
        second = A.build_table()
        self.assertNotEqual(first, second)
        self.assertEqual(second[0]['example'].params[0].elem, 'double')
        self.assertEqual(A._build_call.call_count, 2)

    def test_each_file_dependency_invalidates(self):
        A.build_table()
        dependencies = [self.root / relative for relative in (
            'scripts/correctness/helper.py', 'runtime/helper.h', 'headers/vendor.h',
            'netlib/SRC/entry.f', 'generic_solver/lib_v3/raised/example.mlir',
            'generic_solver/lib_v3/manifest/example.json')]
        for count, path in enumerate(dependencies, 2):
            path.write_text(path.read_text() + '\n')
            A.build_table()
            self.assertEqual(A._build_call.call_count, count, str(path))

    def test_subset_does_not_reuse_full_table(self):
        A.build_table()
        self.assertEqual(A.build_table([]), ({}, {}, {}))

    def test_policy_change_invalidates(self):
        with patch.dict(os.environ, POLYGEIST_NUMERIC_MODEL='strict'):
            A.build_table()
        with patch.dict(os.environ, POLYGEIST_NUMERIC_MODEL='relaxed'):
            A.build_table()
        self.assertEqual(A._build_call.call_count, 2)

    def test_corrupt_cache_rebuilds(self):
        expected = A.build_table()
        next(self.cache.glob('*.json')).write_text('{broken')
        self.assertEqual(A.build_table(), expected)
        self.assertEqual(A._build_call.call_count, 2)

    def test_disable_and_read_only_cache_fall_back(self):
        with patch.dict(os.environ, POLYGEIST_ABI_NO_CACHE='1'):
            expected = A.build_table()
        self.assertFalse(self.cache.exists())
        self.cache.write_text('not a directory')
        self.assertEqual(A.build_table(), expected)

    def test_serialization_preserves_dataclasses_tuples_and_integer_keys(self):
        table = A.build_table()
        value = (table, {1: [True, None, ('x', 3.5)]})
        classes = {name: cls for name, cls in vars(A).items()
                   if isinstance(cls, type) and dataclasses.is_dataclass(cls)}
        self.assertEqual(C.decode(C.encode(value), classes), value)


class RequiredRuntime(unittest.TestCase):
    def test_only_referenced_adapters_and_builtin_helper(self):
        calls = {'foo': 1, 'bar': 2}
        self.assertEqual(G.required_calls(calls,
            'func.call @polygeist_v2_foo()\nfunc.call @polygeist_v2_memref_copy()'), {'foo': 1})

    def test_address_taken_adapter_is_kept(self):
        self.assertEqual(G.required_calls({'foo': 1},
            '%f = func.constant @polygeist_v2_foo'), {'foo': 1})

    def test_unknown_adapter_is_rejected(self):
        with self.assertRaises(ValueError):
            G.required_calls({}, 'func.call @polygeist_v2_missing()')


if __name__ == '__main__':
    unittest.main()
