"""Public pooling configuration matches only after common source raising.

The input fixture is read unchanged. No matching result is manufactured from
its name, and an otherwise corresponding three-element library configuration
must not be accepted for the two-element application window.
"""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]),
               str(Path(__file__).resolve().parents[1] / 'lib_v2')]
import build_lib_v2
import match_v2
import recipe

REPO = Path(__file__).resolve().parents[3]


class PublicPoolingMatch(unittest.TestCase):
    def test_dynamic_reference_and_wrong_window(self):
        name = 'aten_avg_pool3d_backward_k2_f32'
        spec = next(s for s in build_lib_v2.load_specs() if s.name == name)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wrapper = root/(name+'.c')
            wrapper.write_text('#define NETLIB_SPEC '+name+'\n#include "'+
                               str(spec.ref_c)+'"\n')
            sources = [('library', wrapper, name),
                       ('app', REPO/'issues/aten_c_kernels/aten_avg_pool3d_backward_cpu.c',
                        'aten_avg_pool3d_backward_cpu')]
            for stem, source, function in sources:
                result = recipe.raise_c(recipe.PRODUCTION, str(source), function,
                                        str(root/stem), [], timeout=120)
                for stage in ('cgeist', 'raise', 'debuf'):
                    self.assertTrue(result[stage].ok, (stage, result[stage].stderr))
            application = (root/'app_debuf.mlir').read_text()
            self.assertNotIn('affine.for', application)
            self.assertNotIn('scf.for', application)
            with patch.object(build_lib_v2, 'RAISED', root):
                definition, metadata = build_lib_v2.to_defn(
                    (root/'library_debuf.mlir').read_text(), spec)
            text = ('module {\n// lib_v2-meta: '+json.dumps(metadata)+'\n'+
                    definition+'\n}\n')
            library = root/'library.mlir'
            library.write_text(text)
            with patch.dict(os.environ, POLYGEIST_NUMERIC_MODEL='relaxed'):
                _, report, diagnostic = match_v2.run(application, library, region=False,
                                                     diag=True)
                self.assertEqual([item['spec'] for item in report], [name])
                self.assertEqual(diagnostic['residual_generics'], [])
                self.assertEqual(report[0]['binding']['%R1'], 'literal 1')
                self.assertEqual(report[0]['binding']['%R0'], 'literal 0')
                self.assertEqual(report[0]['binding']['%R2'], 'literal 0')

                wrong = text.replace('arith.constant 2 : i32', 'arith.constant 3 : i32')
                wrong = wrong.replace('arith.constant 2 : index', 'arith.constant 3 : index')
                wrong = wrong.replace('8.000000e+00 : f32', '2.700000e+01 : f32')
                self.assertNotEqual(text, wrong)
                library.write_text(wrong)
                _, report, _ = match_v2.run(application, library, region=False)
                self.assertEqual(report, [])


if __name__ == '__main__':
    unittest.main()
