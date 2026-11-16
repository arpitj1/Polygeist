"""Raising must retain a strict FP addition after a nested reduction."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'scripts/correctness'), str(ROOT/'scripts/correctness/lib_v2')]
import recipe as R
import reference_execution as E


class ReductionEpilogueOrder(unittest.TestCase):
    def test_epilogue_rounding_and_special_values(self):
        with tempfile.TemporaryDirectory() as folder:
            d = Path(folder)
            source = d/'fold.c'
            source.write_text('''void fold(float x[2][3][4], float y[2]) {
  for (int i=0;i<2;++i) {
    float sum=0;
    for (int j=0;j<3;++j) for (int k=0;k<4;++k) sum+=x[i][j][k];
    y[i]=sum+y[i];
  }
}
''')
            raised = R.raise_c(R.PRODUCTION, str(source), 'fold',
                               str(d/'source'), [], 60)
            for stage in ['cgeist', 'raise', 'debuf']:
                self.assertTrue(raised[stage].ok, raised[stage].stderr)
            log = d/'lower.log'
            log.write_text('')
            llvm = E.lower(Path(raised['paths']['debuf']), d, 'fold', 'fold', log)
            self.assertIsNotNone(llvm, log.read_text()[-3000:])
            wrapper = subprocess.check_output([
                sys.executable, str(ROOT/'scripts/correctness/gen_wrapper.py'),
                str(source), 'fold'], text=True)
            (d/'wrapper.c').write_text(wrapper)
            (d/'harness.c').write_text('''#include <string.h>
#include <math.h>
#include <stdio.h>
void fold(float[2][3][4],float[2]);
void original(float[2][3][4],float[2]);
int main(void) {
  const float values[]={1.f,0x1p24f,-0x1p24f,3.f,-3.f,0.f,-0.f,
                        INFINITY,-INFINITY,NAN,0x1p-149f,-0x1p-149f};
  float x[2][3][4],saved[2][3][4],want[2],got[2];
  for(int trial=0;trial<49;++trial) for(int s=0;s<12;++s) {
    for(int i=0;i<24;++i) ((float*)x)[i]=trial==0 ? 1.f : values[(i+trial)%12];
    memcpy(saved,x,sizeof(x));
    for(int i=0;i<2;++i) want[i]=got[i]=values[s];
    original(x,want);fold(x,got);
    for(int i=0;i<2;++i)
      if(!(isnan(want[i])&&isnan(got[i]))&&memcmp(want+i,got+i,sizeof(float))) {
        printf("epilogue trial%d seed%d want%a got%a\\n",trial,s,want[i],got[i]);
        return 1;
      }
    if(memcmp(x,saved,sizeof(x))) return 2;
  }
  return 0;
}
''')
            flags = ['-O0', '-ffp-contract=off', '-fno-fast-math']
            subprocess.run([E.CLANG, *flags, '-Dfold=original', '-c', str(source),
                            '-o', str(d/'original.o')], check=True,
                           capture_output=True, text=True)
            subprocess.run([E.CLANG, *flags, str(d/'original.o'), str(d/'wrapper.c'),
                            str(d/'harness.c'), str(llvm), '-lm', '-o', str(d/'check')],
                           check=True, capture_output=True, text=True)
            result = subprocess.run([str(d/'check')], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)


if __name__ == '__main__':
    unittest.main()
