"""Blocked views preserve every element and IEEE payload in both directions."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'scripts/correctness'),str(ROOT/'scripts/correctness/lib_v2')]
import recipe as R
import e2e_v2 as E

class DivModViewExecution(unittest.TestCase):
    def test_odd_block_dimensions_and_inverse(self):
        for gather in [False,True]:
            with self.subTest(gather=gather),tempfile.TemporaryDirectory() as folder:
                d=Path(folder);source=d/'source.c'
                args='float input[2][4][3][5],float output[6][20]' if gather else 'float input[6][20],float output[2][4][3][5]'
                assignment='output[r][c]=input[r/3][c/5][r%3][c%5]' if gather else 'output[r/3][c/5][r%3][c%5]=input[r][c]'
                source.write_text('void odd_blocks('+args+'){for(int r=0;r<6;++r)for(int c=0;c<20;++c)'+assignment+';}\n')
                raised=R.raise_c(R.PRODUCTION,str(source),'odd_blocks',str(d/'raised'),[],60)
                self.assertTrue(all(raised[k].ok for k in ['cgeist','raise','debuf']))
                text=subprocess.check_output([R.POLYGEIST_OPT,'--normalize-linalg-matching-views=max-operand-rank=1 fuse-initializers=true',raised['paths']['debuf']],text=True)
                self.assertNotIn('floordiv',text)
                self.assertNotIn(' mod ',text)
                normalized=d/'normalized.mlir';normalized.write_text(text)
                log=d/'lower.log';log.write_text('')
                llvm=E.lower(normalized,d,'blocked','odd_blocks',log)
                self.assertIsNotNone(llvm,log.read_text()[-3000:])
                (d/'wrapper.c').write_text(subprocess.check_output([sys.executable,str(ROOT/'scripts/correctness/gen_wrapper.py'),str(source),'odd_blocks'],text=True))
                (d/'harness.c').write_text('''#include <stdint.h>
#include <string.h>
void odd_blocks(float*,float*);void original(float*,float*);
int main(){
 uint32_t values[]={0,0x80000000,0x7f800000,0xff800000,0x7fc00019,0x7f800001,1,0x80000001,0x3f800000,0xbf800000};
 float in[128],copy[128],want[128],got[128];
 for(int trial=0;trial<20;++trial){
  for(int i=0;i<128;++i){memcpy(in+i,values+(i+trial)%10,4);want[i]=got[i]=19.0f;}
  memcpy(copy,in,sizeof(in));original(in+4,want+4);odd_blocks(in+4,got+4);
  if(memcmp(want,got,sizeof(want))||memcmp(in,copy,sizeof(in)))return 1;
 }
 return 0;
}
''')
                flags=['-O0','-ffp-contract=off','-fno-fast-math']
                subprocess.run([E.CLANG,*flags,'-Dodd_blocks=original','-c',str(source),'-o',str(d/'original.o')],check=True,capture_output=True,text=True)
                linked=subprocess.run([E.CLANG,*flags,str(d/'original.o'),str(d/'wrapper.c'),str(d/'harness.c'),str(llvm),'-lm',f'-L{E.MLIR_LIBDIR}',f'-Wl,-rpath,{E.MLIR_LIBDIR}','-lmlir_c_runner_utils','-o',str(d/'check')],capture_output=True,text=True)
                self.assertEqual(linked.returncode,0,linked.stderr)
                self.assertEqual(subprocess.run([str(d/'check')]).returncode,0)

if __name__=='__main__':unittest.main()
