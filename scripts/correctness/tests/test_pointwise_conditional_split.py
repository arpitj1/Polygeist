"""Conditional splitting preserves branch order, exceptional values and guards."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'scripts/correctness'), str(ROOT/'scripts/correctness/lib_v2')]
import recipe as R
import reference_execution as E


class PointwiseConditionalSplit(unittest.TestCase):
    def test_unsigned_remainder_prefix_preserves_bits_and_tail(self):
        with tempfile.TemporaryDirectory() as folder:
            d=Path(folder);source=d/'source.c'
            source.write_text('''void translate(unsigned bits[16], int range,
int base, int out[16]) {
 for(int i=0;i<16;++i)out[i]=(int)(bits[i]%(unsigned)range)+base;
}
''')
            raised=R.raise_c(R.PRODUCTION,str(source),'translate',str(d/'raised'),[],60)
            self.assertTrue(all(raised[k].ok for k in ('cgeist','raise','debuf')))
            text=subprocess.check_output([R.POLYGEIST_OPT,
                '--normalize-linalg-matching-views=split-pointwise-inputs=true',
                '--cse',raised['paths']['debuf']],text=True,timeout=30)
            self.assertEqual(text.count('linalg.generic'),2)
            self.assertIn('arith.remui',text)
            normalized=d/'normalized.mlir';normalized.write_text(text)
            log=d/'lower.log';log.write_text('')
            llvm=E.lower(normalized,d,'integer','translate',log)
            self.assertIsNotNone(llvm,log.read_text()[-2000:])
            (d/'wrapper.c').write_text(subprocess.check_output([sys.executable,
                str(ROOT/'scripts/correctness/gen_wrapper.py'),str(source),'translate'],text=True))
            (d/'harness.c').write_text('''#include <stdint.h>
#include <string.h>
void translate(unsigned*,int,int,int*);
void original(unsigned*,int,int,int*);
int main(void){
 uint32_t values[]={0,1,16,17,255,256,2147483647u,2147483648u,4294967295u};
 unsigned ranges[]={1,17,257,1024};int bases[]={-100,0,1000000};
 for(int r=0;r<4;++r)for(int b=0;b<3;++b){
  unsigned input[20],copy[20];int want[20],got[20];
  for(int i=0;i<20;++i){input[i]=values[i%9];want[i]=got[i]=-777;}
  memcpy(copy,input,sizeof(input));
  original(input+2,ranges[r],bases[b],want+2);
  translate(input+2,ranges[r],bases[b],got+2);
  if(memcmp(copy,input,sizeof(input))||memcmp(want,got,sizeof(want)))return 1;
 }return 0;}
''')
            subprocess.run([E.CLANG,'-O0','-Dtranslate=original','-c',str(source),
                '-o',str(d/'original.o')],check=True,capture_output=True)
            subprocess.run([E.CLANG,'-O0',str(d/'original.o'),str(d/'wrapper.c'),
                str(d/'harness.c'),str(llvm),'-lm',f'-L{E.MLIR_LIBDIR}',
                f'-Wl,-rpath,{E.MLIR_LIBDIR}','-lmlir_c_runner_utils','-o',str(d/'check')],
                check=True,capture_output=True)
            self.assertEqual(subprocess.run([str(d/'check')]).returncode,0)

    def test_nested_conditions_preserve_all_float_bits(self):
        with tempfile.TemporaryDirectory() as folder:
            d = Path(folder)
            source = d/'source.c'
            source.write_text('''void piecewise(float x[16], float value[16],
float out[16], float lower, float upper) {
  for (int i=0; i<16; ++i)
    out[i] = x[i] < lower ? value[i] : (x[i] > upper ? upper : x[i]);
}
''')
            raised = R.raise_c(R.PRODUCTION,str(source),'piecewise',str(d/'raised'),[],60)
            self.assertTrue(all(raised[k].ok for k in ('cgeist','raise','debuf')))
            text = subprocess.check_output([R.POLYGEIST_OPT,
                '--normalize-linalg-matching-views=split-pointwise-inputs=true',
                '--cse',raised['paths']['debuf']],text=True,timeout=30)
            self.assertEqual(text.count('linalg.generic'),2)
            normalized = d/'normalized.mlir'; normalized.write_text(text)
            log = d/'lower.log'; log.write_text('')
            llvm = E.lower(normalized,d,'conditional','piecewise',log)
            self.assertIsNotNone(llvm,log.read_text()[-2000:])
            (d/'wrapper.c').write_text(subprocess.check_output([sys.executable,
                str(ROOT/'scripts/correctness/gen_wrapper.py'),str(source),'piecewise'],text=True))
            (d/'harness.c').write_text('''#include <stdint.h>
#include <string.h>
void piecewise(float*,float*,float*,float,float);
void original(float*,float*,float*,float,float);
int main(void) {
 uint32_t bits[]={0,0x80000000,0x3f800000,0xbf800000,0x7f800000,
                  0xff800000,0x7fc00019,0x7f800001,1,0x80000001};
 float values[10]; memcpy(values,bits,sizeof(bits));
 for(int lo=0;lo<10;++lo)for(int hi=0;hi<10;++hi){
  float x[20],v[20],copyx[20],copyv[20],want[20],got[20];
  for(int i=0;i<20;++i){x[i]=values[i%10];v[i]=values[(i+3)%10];want[i]=got[i]=19;}
  memcpy(copyx,x,sizeof(x));memcpy(copyv,v,sizeof(v));
  original(x+2,v+2,want+2,values[lo],values[hi]);
  piecewise(x+2,v+2,got+2,values[lo],values[hi]);
  if(memcmp(want,got,sizeof(want))||memcmp(x,copyx,sizeof(x))||memcmp(v,copyv,sizeof(v)))return 1;
 }
 return 0;
}
''')
            flags = ['-O0','-ffp-contract=off','-fno-fast-math']
            subprocess.run([E.CLANG,*flags,'-Dpiecewise=original','-c',str(source),
                '-o',str(d/'original.o')],check=True,capture_output=True)
            subprocess.run([E.CLANG,*flags,str(d/'original.o'),str(d/'wrapper.c'),
                str(d/'harness.c'),str(llvm),'-lm',f'-L{E.MLIR_LIBDIR}',
                f'-Wl,-rpath,{E.MLIR_LIBDIR}','-lmlir_c_runner_utils','-o',str(d/'check')],
                check=True,capture_output=True)
            self.assertEqual(subprocess.run([str(d/'check')]).returncode,0)


if __name__ == '__main__':
    unittest.main()
