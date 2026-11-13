"""Execute math lowering at inputs where expansion loses meaningful bits."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts/correctness/lib_v2'))
import reference_execution as E


class PreciseMathLowering(unittest.TestCase):
    def test_small_inputs_preserve_libm_primitives(self):
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            (d/'input.mlir').write_text('''module {
func.func @test_expm1(%x: f32) -> f32 {
  %y = math.expm1 %x : f32
  return %y : f32
}
func.func @test_log1p(%x: f32) -> f32 {
  %y = math.log1p %x : f32
  return %y : f32
}
}''')
            subprocess.run([E.R.POLYGEIST_OPT, '--lower-supported-math-to-libm',
                            str(d/'input.mlir'), '-o', str(d/'libm.mlir')],
                           check=True, capture_output=True)
            subprocess.run([E.MLIR_OPT, *E.LOWER_TO_LLVM, str(d/'libm.mlir'),
                            '-o', str(d/'llvm.mlir')], check=True, capture_output=True)
            subprocess.run([E.MLIR_TRANSLATE, '--mlir-to-llvmir', str(d/'llvm.mlir'),
                            '-o', str(d/'code.ll')], check=True, capture_output=True)
            (d/'test.c').write_text('''#include <math.h>
#include <string.h>
float test_expm1(float), test_log1p(float);
int main(void) {
  const float x[] = {0x1p-30f,-0x1p-30f,0x1p-100f,-0x1p-100f,0.f,-0.f};
  for (unsigned i=0; i<sizeof(x)/sizeof(x[0]); ++i) {
    float a=test_expm1(x[i]), b=expm1f(x[i]);
    if (memcmp(&a,&b,sizeof(a))) return 1;
    a=test_log1p(x[i]); b=log1pf(x[i]);
    if (memcmp(&a,&b,sizeof(a))) return 2;
  }
  return 0;
}''')
            subprocess.run([E.CLANG, str(d/'test.c'), str(d/'code.ll'), '-lm',
                            '-o', str(d/'test')], check=True, capture_output=True)
            result = subprocess.run([str(d/'test')], capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
