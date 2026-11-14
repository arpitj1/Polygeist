"""Pointwise decomposition preserves both outputs and opaque-call execution."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'scripts/correctness'), str(ROOT/'scripts/correctness/lib_v2')]
import recipe as R
import reference_execution as E


class PointwiseMultipleOutputs(unittest.TestCase):
    def test_effectful_calls_are_not_split_or_duplicated(self):
        template = '''module {
func.func private @helper(f32) -> f32 ATTR
func.func @pair(%x: tensor<8xf32>, %a: tensor<8xf32>, %b: tensor<8xf32>) -> (tensor<8xf32>,tensor<8xf32>) {
 %r:2 = linalg.generic {indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>], iterator_types = ["parallel"]} ins(%x: tensor<8xf32>) outs(%a,%b: tensor<8xf32>,tensor<8xf32>) {
 ^bb0(%xv:f32,%av:f32,%bv:f32):
  %h = func.call @helper(%xv) : (f32)->f32
  %p = arith.addf %h,%xv : f32
  %q = arith.subf %h,%xv : f32
  linalg.yield %p,%q : f32,f32
 } -> (tensor<8xf32>,tensor<8xf32>)
 return %r#0,%r#1 : tensor<8xf32>,tensor<8xf32>
}
}'''
        for pure in (False, True):
            source = template.replace('ATTR','attributes {polygeist.pure}' if pure else '')
            p = subprocess.run([R.POLYGEIST_OPT,
                '--normalize-linalg-matching-views=split-pointwise-inputs=true'],
                input=source,text=True,capture_output=True,timeout=30)
            self.assertEqual(p.returncode,0,p.stderr)
            self.assertEqual(p.stdout.count('func.call @helper'),1)
            if pure:
                self.assertGreater(p.stdout.count('linalg.generic'),1)
            else:
                self.assertEqual(p.stdout.count('linalg.generic'),1)

    def test_two_outputs_preserve_exceptional_values_and_guards(self):
        with tempfile.TemporaryDirectory() as folder:
            d = Path(folder); source = d/'source.c'
            source.write_text('''extern __attribute__((const)) float hypotf(float,float);
void pair(float x[16],float y[16],float a[16],float b[16]) {
 for(int i=0;i<16;++i){float h=hypotf(x[i],y[i]);
  a[i]=h==0?0:x[i]/h; b[i]=h==0?0:y[i]/h;}
}
''')
            raised = R.raise_c(R.PRODUCTION,str(source),'pair',str(d/'raised'),[],60)
            self.assertTrue(all(raised[k].ok for k in ('cgeist','raise','debuf')))
            text = subprocess.check_output([R.POLYGEIST_OPT,
                '--normalize-linalg-matching-views=split-pointwise-inputs=true',
                '--cse',raised['paths']['debuf']],text=True,timeout=30)
            self.assertEqual(text.count('func.call @hypotf'),1)
            self.assertGreaterEqual(text.count('linalg.generic'),5)
            normalized=d/'normalized.mlir';normalized.write_text(text)
            log=d/'lower.log';log.write_text('');llvm=E.lower(normalized,d,'pair','pair',log)
            self.assertIsNotNone(llvm,log.read_text()[-2000:])
            (d/'wrapper.c').write_text(subprocess.check_output([sys.executable,
                str(ROOT/'scripts/correctness/gen_wrapper.py'),str(source),'pair'],text=True))
            (d/'harness.c').write_text('''#include <stdint.h>
#include <string.h>
void pair(float*,float*,float*,float*);void original(float*,float*,float*,float*);
int main(void){
 uint32_t bits[]={0,0x80000000,0x3f800000,0xbf800000,0x7f800000,0xff800000,
  0x7fc00019,0x7f800001,1,0x80000001,0x7f7fffff,0x00800000};
 float values[12];memcpy(values,bits,sizeof(bits));
 for(int mode=0;mode<12;++mode){
  float x[20],y[20],cx[20],cy[20],a[20],b[20],wa[20],wb[20];
  for(int i=0;i<20;++i){x[i]=values[i%12];y[i]=values[(i+mode)%12];a[i]=b[i]=wa[i]=wb[i]=19;}
  memcpy(cx,x,sizeof(x));memcpy(cy,y,sizeof(y));
  original(x+2,y+2,wa+2,wb+2);pair(x+2,y+2,a+2,b+2);
  if(memcmp(a,wa,sizeof(a))||memcmp(b,wb,sizeof(b))||memcmp(x,cx,sizeof(x))||memcmp(y,cy,sizeof(y)))return 1;
 }return 0;}
''')
            flags=['-O0','-ffp-contract=off','-fno-fast-math']
            subprocess.run([E.CLANG,*flags,'-Dpair=original','-c',str(source),'-o',str(d/'original.o')],check=True,capture_output=True)
            subprocess.run([E.CLANG,*flags,str(d/'original.o'),str(d/'wrapper.c'),str(d/'harness.c'),str(llvm),'-lm',f'-L{E.MLIR_LIBDIR}',f'-Wl,-rpath,{E.MLIR_LIBDIR}','-lmlir_c_runner_utils','-o',str(d/'check')],check=True,capture_output=True)
            self.assertEqual(subprocess.run([str(d/'check')]).returncode,0)


if __name__ == '__main__':
    unittest.main()
