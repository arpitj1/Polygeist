"""Skipping empty searches and reusing prepared references must preserve matching."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

CORRECTNESS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORRECTNESS))
from region_match import matcher as M
from lib_v2 import match_v2


COPY = '''module {
  func.func @copy(%n: index, %x: memref<?xf64>, %y: memref<?xf64>) {
    affine.for %i = 0 to %n {
      %v = affine.load %x[%i] : memref<?xf64>
      affine.store %v, %y[%i] : memref<?xf64>
    }
    return
  }
}
'''

LAUNCH = '''module {
  func.func @application(%x: memref<?xf64>, %y: memref<?xf64>) {
    %a = bufferization.to_tensor %x : memref<?xf64>
    %b = bufferization.to_tensor %y : memref<?xf64>
    %r = kernel.launch @copy(%a, %b) : (tensor<?xf64>, tensor<?xf64>) -> tensor<?xf64>
    %m = bufferization.to_memref %r : memref<?xf64>
    memref.copy %m, %y : memref<?xf64> to memref<?xf64>
    return
  }
}
'''


class RegionPreparation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.specs = self.root/'lib_v3'
        for name in ('manifest', 'ref_c', 'raised'):
            (self.specs/name).mkdir(parents=True)
        meta = dict(spec='copy', params=['x', 'y'], outputs=['y'])
        (self.root/'kernel_library_v3.mlir').write_text('// lib_v2-meta: '+json.dumps(meta))

    def test_launch_writeback_does_not_prepare_catalog(self):
        with patch.object(M, 'load_specs', side_effect=AssertionError('unneeded catalog')):
            self.assertFalse(M.has_matchable_work(LAUNCH, str(self.specs)))
            self.assertEqual(M.match_text(LAUNCH, str(self.specs)), [])
        with patch.object(match_v2, 'SPEC_DIR', self.specs), \
                patch('region_match.load_specs', side_effect=AssertionError('unneeded catalog')) as load:
            report = []
            self.assertEqual(match_v2.splice_region_matches(LAUNCH, report), LAUNCH)
            load.assert_not_called()
            self.assertEqual(report, [{'region_match_skipped': 'no remaining computation'}])

    def test_independent_copy_remains_work(self):
        independent = LAUNCH.replace('memref.copy %m, %y', 'memref.copy %m, %x')
        self.assertTrue(M.has_matchable_work(independent, str(self.specs)))
        with patch.object(M, 'load_specs', return_value=[]) as load:
            M.match_text(independent, str(self.specs))
            load.assert_called_once()

    def test_loop_and_additional_function_remain_work(self):
        self.assertTrue(M.has_matchable_work(COPY, str(self.specs)))
        self.assertTrue(M.has_matchable_work(LAUNCH+COPY, str(self.specs)))

    def test_store_remains_work(self):
        text = LAUNCH.replace('    return', '''    %z = arith.constant 0 : index
    %v = arith.constant 0.0 : f64
    memref.store %v, %x[%z] : memref<?xf64>
    return''')
        self.assertTrue(M.has_matchable_work(text, str(self.specs)))

    def test_unknown_launch_or_syntax_keeps_normal_path(self):
        for text in (LAUNCH.replace('@copy(', '@unknown('),
                     LAUNCH.replace('    return', '    unknown.operation\n    return'),
                     LAUNCH.replace('func.func', '"func.func"'),
                     LAUNCH.replace('@application(', '@application(\n')):
            with self.subTest(text=text):
                self.assertTrue(M.has_matchable_work(text, str(self.specs)))

    def populate(self):
        manifest = dict(routine='copy', specs=[dict(name='copy', args=['n', 'x', 'y'])])
        (self.specs/'manifest/copy.json').write_text(json.dumps(manifest))
        (self.specs/'raised/copy.mlir').write_text(COPY)
        (self.specs/'raised/copy_debuf.mlir').write_text(COPY)
        (self.specs/'ref_c/copy.c').write_text('void copy(int n, double x[n], double y[n]);')

    def child(self, seed, *, no_cache=False):
        code = '''
import json,sys
sys.path.insert(0,sys.argv[1])
from region_match import matcher as M
from region_match.affine import Aff
specs=M.load_specs(sys.argv[2])
matches=M.match_text(open(sys.argv[2]+'/raised/copy.mlir').read(),sys.argv[2])
print(json.dumps(dict(specs=[(s.name,s.validation) for s in specs],
                     matches=[(m.spec,m.binding,m.launch_text) for m in matches])))
'''
        env = dict(os.environ, PYTHONHASHSEED=str(seed),
                   REGION_MATCH_CACHE_DIR=str(self.root/'cache'),
                   REGION_MATCH_NO_CACHE='1' if no_cache else '0',
                   POLYGEIST_NUMERIC_MODEL='relaxed', POLYGEIST_SATURATION_TIMEOUT='10',
                   POLYGEIST_MATCHER_TELEMETRY_JSONL=str(self.root/f'{seed}.jsonl'))
        result = subprocess.run([sys.executable, '-c', code, str(CORRECTNESS), str(self.specs)],
                                env=env, text=True, capture_output=True, check=True, timeout=60)
        events = [json.loads(line) for line in (self.root/f'{seed}.jsonl').read_text().splitlines()]
        return json.loads(result.stdout), events

    def test_fresh_process_cache_preserves_real_match(self):
        self.populate()
        cold, cold_events = self.child(11)
        warm, warm_events = self.child(22)
        uncached, _ = self.child(33, no_cache=True)
        self.assertTrue(cold['matches'])
        self.assertEqual(cold, warm)
        self.assertEqual(cold, uncached)
        self.assertEqual(next(e['status'] for e in cold_events if e['phase']=='region_catalog_cache'), 'miss')
        self.assertEqual(next(e['status'] for e in warm_events if e['phase']=='region_catalog_cache'), 'disk_hit')

    def test_affine_hash_is_rebuilt_in_new_process(self):
        # Cached Aff expressions are dictionary/set keys. Their string-based
        # hash must be computed under the reader's Python hash seed.
        code = '''
import pickle,sys
sys.path.insert(0,sys.argv[1])
from region_match.affine import Aff
p=sys.argv[2]
if sys.argv[3]=='write':
    with open(p,'wb') as f: pickle.dump({Aff.var('i'): 'present'},f)
else:
    with open(p,'rb') as f: values=pickle.load(f)
    assert values[Aff.var('i')]=='present'
'''
        for seed, mode in [(1, 'write'), (2, 'read')]:
            subprocess.run([sys.executable, '-c', code, str(CORRECTNESS),
                            str(self.root/'aff.pkl'), mode], check=True, timeout=10,
                           env=dict(os.environ, PYTHONHASHSEED=str(seed)))

    def test_cache_key_changes_with_inputs_and_policy(self):
        self.populate()
        with patch.dict(os.environ, POLYGEIST_NUMERIC_MODEL='relaxed',
                        EQSAT_ABLATION='full', POLYGEIST_SATURATION_TIMEOUT='10'):
            original = M._cache_key(str(self.specs))
            for name in ('raised/copy.mlir', 'manifest/copy.json', 'ref_c/copy.c'):
                path = self.specs/name
                before = path.read_bytes()
                path.write_bytes(before+b'\n')
                self.assertNotEqual(M._cache_key(str(self.specs)), original, name)
                path.write_bytes(before)
            for variable, value in [('POLYGEIST_NUMERIC_MODEL', 'strict'),
                                    ('EQSAT_ABLATION', 'none'),
                                    ('POLYGEIST_SATURATION_TIMEOUT', '1')]:
                with patch.dict(os.environ, {variable: value}):
                    self.assertNotEqual(M._cache_key(str(self.specs)), original, variable)
            self.assertEqual(M._cache_key(str(self.specs)), original)

    def test_loaded_cache_uses_current_reference_directory(self):
        self.populate()
        moved = self.root/'other_v3'
        shutil.copytree(self.specs, moved)
        with patch.dict(os.environ, REGION_MATCH_CACHE_DIR=str(self.root/'cache')), \
                patch.object(M, '_SPEC_CACHE', {}), patch.object(M, '_cache_key', return_value='same-inputs'):
            original = M.load_specs(str(self.specs))
            with patch.object(M, '_load_specs_uncached', side_effect=AssertionError('cache miss')):
                restored = M.load_specs(str(moved))
            self.assertEqual(restored[0].source_dir, str(moved))
            self.assertEqual(original[0].source_dir, str(self.specs))

    def test_allowed_specifications_have_separate_caches(self):
        self.populate()
        with patch.dict(os.environ, REGION_MATCH_CACHE_DIR=str(self.root/'cache')), \
                patch.object(M, '_SPEC_CACHE', {}):
            self.assertEqual([s.name for s in M.load_specs(str(self.specs), allowed_specs={'copy'})], ['copy'])
            self.assertEqual(M.load_specs(str(self.specs), allowed_specs={'missing'}), [])

    def test_cache_key_tracks_proof_code_and_library_version(self):
        code = self.root/'code'
        region = code/'region_match'
        region.mkdir(parents=True)
        (code/'lib_v2').mkdir()
        dependencies = ('region_match/egg.py', 'kernel_match.py', 'eqsat_ablation.py',
                        'proof_metrics.py', 'proof_worker.py', 'lib_v2/spec_provenance.py')
        for name in dependencies:
            (code/name).write_text('# initial\n')
        with patch.object(M, '_HERE', str(region)):
            original = M._cache_key(str(self.specs))
            for name in dependencies:
                (code/name).write_text('# changed\n')
                self.assertNotEqual(M._cache_key(str(self.specs)), original, name)
                (code/name).write_text('# initial\n')
            with patch.object(M.importlib.metadata, 'version', return_value='different'):
                self.assertNotEqual(M._cache_key(str(self.specs)), original)
            (self.specs/'import_manifest.json').write_text('{}')
            self.assertNotEqual(M._cache_key(str(self.specs)), original)


if __name__ == '__main__':
    unittest.main()
