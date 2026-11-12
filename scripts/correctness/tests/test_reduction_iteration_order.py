"""Collapsing adjacent reduction axes must retain scalar evaluation order."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'scripts/correctness'), str(ROOT/'scripts/correctness/lib_v2')]
import recipe as R
import e2e_v2 as E


class ReductionIterationOrder(unittest.TestCase):
    def test_initialized_rows_preserve_seed_and_order(self):
        with tempfile.TemporaryDirectory() as folder:
            d = Path(folder)
            source = d/'reference.c'
            source.write_text('''void grouped(float input[2][3][5][7],float output[2][3],float seed) {
  for(int a=0;a<2;++a)for(int b=0;b<3;++b){
    output[a][b]=seed;
    for(int c=0;c<5;++c)for(int e=0;e<7;++e)output[a][b]+=input[a][b][c][e];
  }
}
''')
            raised = R.raise_c(R.PRODUCTION,str(source),'grouped',str(d/'source'),[],60)
            self.assertTrue(all(raised[k].ok for k in ['cgeist','raise','debuf']))
            normalized = subprocess.check_output([R.POLYGEIST_OPT,
                '--normalize-linalg-matching-views=max-operand-rank=2 fuse-initializers=true',
                raised['paths']['debuf']],text=True)
            self.assertIn('tensor<6x35xf32>',normalized)
            self.assertNotIn('scf.for',normalized)
            path=d/'normalized.mlir';path.write_text(normalized)
            log=d/'lower.log';log.write_text('')
            llvm=E.lower(path,d,'rows','grouped',log)
            self.assertIsNotNone(llvm,log.read_text()[-3000:])
            wrapper=subprocess.check_output([sys.executable,str(ROOT/'scripts/correctness/gen_wrapper.py'),str(source),'grouped'],text=True)
            (d/'wrapper.c').write_text(wrapper)
            (d/'harness.c').write_text('''#include <string.h>
#include <math.h>
void grouped(float[2][3][5][7],float[2][3],float);
void original(float[2][3][5][7],float[2][3],float);
int main(){
 const float values[]={1e20f,1.f,-1e20f,3.f,-3.f,0.f,-0.f,INFINITY,-INFINITY,NAN,0x1p-149f,-0x1p-149f};
 float input[2][3][5][7],saved[2][3][5][7],want[2][3],got[2][3];
 for(int trial=0;trial<48;++trial)for(int s=0;s<12;++s){
  for(int i=0;i<210;++i)((float*)input)[i]=values[(i+trial)%((trial<12)?3:12)];
  memcpy(saved,input,sizeof(input));
  original(input,want,values[s]);grouped(input,got,values[s]);
  for(int i=0;i<6;++i){float a=((float*)want)[i],b=((float*)got)[i];
   if(!(isnan(a)&&isnan(b))&&memcmp(&a,&b,sizeof(float)))return 1;
  }
  if(memcmp(input,saved,sizeof(input)))return 2;
 }
 return 0;
}
''')
            flags=['-O0','-ffp-contract=off','-fno-fast-math']
            subprocess.run([E.CLANG,*flags,'-Dgrouped=original','-c',str(source),'-o',str(d/'original.o')],check=True,capture_output=True,text=True)
            subprocess.run([E.CLANG,*flags,str(d/'original.o'),str(d/'wrapper.c'),str(d/'harness.c'),str(llvm),'-lm','-o',str(d/'check')],check=True,capture_output=True,text=True)
            result=subprocess.run([str(d/'check')],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_float_cancellation_seed_and_special_values(self):
        with tempfile.TemporaryDirectory() as folder:
            d = Path(folder)
            source = d/'reference.c'
            source.write_text('''float ordered_fold(float input[3][5],float seed) {
  for(int i=0;i<3;++i) for(int j=0;j<5;++j) seed += input[i][j];
  return seed;
}
''')
            raised = R.raise_c(R.PRODUCTION, str(source), 'ordered_fold', str(d/'source'), [], 60)
            self.assertTrue(all(raised[k].ok for k in ['cgeist','raise','debuf']))
            normalized = subprocess.check_output([R.POLYGEIST_OPT,
                '--normalize-linalg-matching-views=max-operand-rank=1 fuse-initializers=true',
                raised['paths']['debuf']], text=True)
            self.assertIn('tensor<15xf32>', normalized)
            self.assertIn('iterator_types = ["reduction"]', normalized)
            path = d/'normalized.mlir'; path.write_text(normalized)
            log = d/'lower.log'; log.write_text('')
            llvm = E.lower(path,d,'order','ordered_fold',log)
            self.assertIsNotNone(llvm,log.read_text()[-3000:])
            (d/'harness.c').write_text('''#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
float ordered_fold(float[3][5],float);
float ordered_fold_impl(float*,float*,int64_t,int64_t,int64_t,int64_t,int64_t,float);
int main(){
 const float values[]={1e20f,1.f,-1e20f,3.f,-3.f,0.f,-0.f,INFINITY,-INFINITY,NAN,0x1p-149f,-0x1p-149f};
 float input[3][5],saved[3][5];
 for(int trial=0;trial<48;++trial)for(int s=0;s<12;++s){
  for(int i=0;i<15;++i)input[i/5][i%5]=values[(i+trial)%((trial<12)?3:12)];
  memcpy(saved,input,sizeof(input));
  float want=ordered_fold(input,values[s]);
  float got=ordered_fold_impl(&input[0][0],&input[0][0],0,3,5,5,1,values[s]);
  if(!(isnan(want)&&isnan(got))&&memcmp(&want,&got,sizeof(float))){
   printf("order mismatch trial%d seed%d want%a got%a\\n",trial,s,want,got);return 1;
  }
  if(memcmp(input,saved,sizeof(input)))return 2;
 }
 return 0;
}
''')
            subprocess.run([E.CLANG,'-O0','-ffp-contract=off','-fno-fast-math',
                str(source),str(d/'harness.c'),str(llvm),'-lm','-o',str(d/'check')],
                check=True,capture_output=True,text=True)
            result = subprocess.run([str(d/'check')],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)


if __name__ == '__main__':
    unittest.main()
