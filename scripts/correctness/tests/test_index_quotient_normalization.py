"""Only exact-width unsigned arithmetic may cross the integer/index boundary."""
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[3]
OPT = ROOT / 'build/bin/polygeist-opt'


def module(op='divui', index_bits=64, integer_bits=64, guarded=False):
    arithmetic = f'''%wide = arith.index_cast %i : index to i{integer_bits}
      %d = arith.extui %stride : i32 to i{integer_bits}
      %q = arith.{op} %wide, %d : i{integer_bits}
      %r = arith.trunci %q : i{integer_bits} to i32'''
    if guarded:
        arithmetic = '''%value = scf.if %guard -> (i32) {
      '''+arithmetic+'''
      scf.yield %r : i32
    } else { scf.yield %old : i32 }
    linalg.yield %value : i32'''
    else:
        arithmetic += '\n    linalg.yield %r : i32'
    return f'''module attributes {{dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, {index_bits} : i64>>}} {{
 func.func @check(%output: memref<7xi32>, %stride: i32, %guard: i1) {{
  linalg.generic {{indexing_maps = [affine_map<(d0)->(d0)>], iterator_types = ["parallel"]}}
      outs(%output : memref<7xi32>) {{
   ^bb0(%old: i32):
    %i = linalg.index 0 : index
    {arithmetic}
  }}
  return
 }}
}}'''


class IndexQuotientNormalization(unittest.TestCase):
    def normalize(self, **kwargs):
        result = subprocess.run([str(OPT), '--normalize-linalg-matching-views'],
            input=module(**kwargs), capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def test_unsigned_division_and_remainder_use_native_index_width(self):
        for op in ['divui', 'remui']:
            with self.subTest(op=op):
                result = self.normalize(op=op)
                self.assertRegex(result, r'arith\.'+op+r' .* : index')
                self.assertNotIn('arith.trunci', result)
                self.assertIn('arith.index_castui', result)
                self.assertRegex(result, r'arith.index_cast .* : index to i32')

    def test_narrow_target_index_leaves_wide_arithmetic_unchanged(self):
        result = self.normalize(index_bits=32)
        self.assertRegex(result, r'arith.divui .* : i64')
        self.assertIn('arith.trunci', result)

    def test_signed_arithmetic_is_not_rewritten(self):
        for op in ['divsi', 'remsi']:
            result = self.normalize(op=op)
            self.assertRegex(result, r'arith\.'+op+r' .* : i64')
            self.assertIn('arith.trunci', result)

    def test_division_keeps_its_control_flow_guard(self):
        result = self.normalize(guarded=True)
        self.assertIn('scf.if', result)
        self.assertLess(result.index('scf.if'), result.index('arith.divui'))
        self.assertRegex(result, r'arith.divui .* : index')

    def test_normalized_execution_matches_c_for_unsigned_divisors(self):
        import sys
        import tempfile
        sys.path[:0] = [str(ROOT/'scripts/correctness'), str(ROOT/'scripts/correctness/lib_v2')]
        import recipe as R
        import reference_execution as E
        with tempfile.TemporaryDirectory() as folder:
            d = Path(folder)
            source = d/'source.c'
            source.write_text('''void coordinates(int n,unsigned int stride,int out[n]) {
 for(long i=0;i<n;++i) out[i]=(int)((unsigned long)i / (unsigned long)stride)
                            +(int)((unsigned long)i % (unsigned long)stride);
}''')
            raised=R.raise_c(R.PRODUCTION,str(source),'coordinates',str(d/'raised'),[],60)
            self.assertTrue(all(raised[k].ok for k in ['cgeist','raise','debuf']))
            text=subprocess.check_output([str(OPT),'--normalize-linalg-matching-views',
                raised['paths']['debuf']],text=True)
            self.assertRegex(text,r'arith.divui .* : index')
            self.assertRegex(text,r'arith.remui .* : index')
            normalized=d/'normalized.mlir';normalized.write_text(text)
            log=d/'lower.log';log.write_text('')
            llvm=E.lower(normalized,d,'coordinates','coordinates',log)
            self.assertIsNotNone(llvm,log.read_text()[-2000:])
            (d/'wrapper.c').write_text('''#include <stdint.h>
void coordinates_impl(int32_t,uint32_t,int32_t*,int32_t*,int64_t,int64_t,int64_t);
void coordinates(int n,uint32_t stride,int32_t* out){
 coordinates_impl(n,stride,out,out,0,n,1);
}''')
            (d/'harness.c').write_text('''#include <stdint.h>
#include <string.h>
void original(int,unsigned int,int*);void coordinates(int,unsigned int,int*);
int main(){
 unsigned int strides[]={1,2,7,31,0x7fffffffU,0x80000000U,0xffffffffU};
 int expected[103],actual[103];
 for(int n=1;n<=97;n+=3)for(int j=0;j<7;++j){
  for(int i=0;i<103;++i)expected[i]=actual[i]=-123456;
  original(n,strides[j],expected+2);coordinates(n,strides[j],actual+2);
  if(memcmp(expected,actual,sizeof(actual)))return 1;
 }
 return 0;
}''')
            subprocess.run([E.CLANG,'-O0','-Dcoordinates=original','-c',str(source),
                '-o',str(d/'original.o')],check=True,capture_output=True,text=True)
            compiled=subprocess.run([E.CLANG,'-O0',str(d/'original.o'),str(d/'wrapper.c'),
                str(d/'harness.c'),str(llvm),'-lm',f'-L{E.MLIR_LIBDIR}',
                f'-Wl,-rpath,{E.MLIR_LIBDIR}','-lmlir_c_runner_utils','-o',str(d/'check')],
                capture_output=True,text=True)
            self.assertEqual(compiled.returncode,0,compiled.stderr)
            self.assertEqual(subprocess.run([str(d/'check')]).returncode,0)

if __name__ == '__main__': unittest.main()
