"""External source roots preserve input identity and reject changed content."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lib_v2'))
import spec_provenance as P


class DependencyProvenance(unittest.TestCase):
    def test_relocation_and_content_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            compiler = root/'compiler'
            compiler.mkdir()
            original = root/'sources'
            relocated = root/'new-sources'
            for base in (original, relocated):
                (base/'library').mkdir(parents=True)
                (base/'library/ref.c').write_text('void reference(void) {}\n')
            with patch.object(P, 'ROOT', compiler), patch.dict(os.environ, POLYGEIST_DEPENDENCY_ROOT=str(original)):
                recorded = P.record([original/'library/ref.c'])
                self.assertEqual(list(recorded), ['third_party/library/ref.c'])
                self.assertIsNone(P.check(recorded))
                with patch.dict(os.environ, POLYGEIST_DEPENDENCY_ROOT=str(relocated)):
                    self.assertIsNone(P.check(recorded))
                    (relocated/'library/ref.c').write_text('changed source\n')
                    self.assertIsNotNone(P.check(recorded))

    def test_escape_and_absolute_paths_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            deps = root/'deps'
            deps.mkdir()
            (root/'outside.c').write_text('outside')
            (deps/'escape.c').symlink_to(root/'outside.c')
            with patch.dict(os.environ, POLYGEIST_DEPENDENCY_ROOT=str(deps)):
                for name in ('third_party/../outside.c', str(root/'outside.c'), 'third_party/escape.c'):
                    with self.subTest(name=name):
                        with self.assertRaises(ValueError):
                            P.input_path(name)
                        self.assertIsNotNone(P.check({name: P.file_hash(root/'outside.c')}))


if __name__ == '__main__':
    unittest.main()
