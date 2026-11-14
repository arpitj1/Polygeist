"""Source dependency expansion must expose computations, not helper identities."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'scripts/correctness/lib_v2')]
import recipe as R
import reference_execution as E


class FrontendSourceSupport(unittest.TestCase):
    def config(self, d, body, required=('helper',)):
        support = d/'support.cpp'
        support.write_text(body)
        name = str(support.relative_to(ROOT))
        config = d/'support.json'
        config.write_text(json.dumps(dict(sources=[name],include_dirs=[],
            required_inline=list(required),inputs={name:hashlib.sha256(support.read_bytes()).hexdigest()})))
        return str(config)

    def test_inline_constant_table_and_execute(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'build') as tmp:
            d=Path(tmp)
            config=self.config(d, '''extern "C" float helper(float x) {
 static const float a[4]={float(1.0/3.0), -2.25f, 0.125f};
 float r=0; for(unsigned char i=0;i<4;++i) r=r*x+a[i];
 return x<0?-r:r;
}''')
            source=d/'input.c'
            source.write_text('''extern float helper(float);
void sample(float x[8],float y[8]){for(int i=0;i<8;++i)y[i]=helper(x[i]);}
''')
            before=source.read_bytes()
            stages=R.raise_c(R.PRODUCTION,str(source),'sample',str(d/'ir'),[],90,frontend_support=config)
            for stage in ('cgeist','raise','debuf'):
                self.assertTrue(stages[stage].ok,stages[stage].stderr)
            text=Path(stages['paths']['affine']).read_text()
            self.assertNotIn('func.call @helper',text)
            self.assertNotIn('@init',text)
            self.assertNotIn('scf.while',text)
            self.assertEqual(source.read_bytes(),before)
            log=d/'lower.log';log.write_text('')
            ll=E.lower(Path(stages['paths']['debuf']),d,'sample','sample',log)
            self.assertIsNotNone(ll,log.read_text()[-2000:])
            (d/'wrapper.c').write_text(subprocess.check_output([sys.executable,
                str(ROOT/'scripts/correctness/gen_wrapper.py'),str(source),'sample'],text=True))
            (d/'check.c').write_text('''#include <string.h>
void sample(float*,float*);void original(float*,float*);
int main(void){float x[8]={-4,-1,-0.0f,0,0.25f,1,2,10},a[8],b[8];
 for(int k=0;k<3;k++){sample(x,a);original(x,b);if(memcmp(a,b,sizeof(a)))return 1;}return 0;}
''')
            p=subprocess.run([E.CLANG,'-std=c++17','-O0','-ffp-contract=off',
                '-Dsample=original','-c',str(d/'ir.frontend.cpp'),'-o',str(d/'ref.o')],capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr)
            p=subprocess.run([E.CLANG,'-O0',str(d/'check.c'),str(d/'wrapper.c'),str(d/'ref.o'),
                str(ll),'-lm','-lstdc++',f'-L{E.MLIR_LIBDIR}',f'-Wl,-rpath,{E.MLIR_LIBDIR}',
                '-lmlir_c_runner_utils','-o',str(d/'check')],capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr)
            self.assertEqual(subprocess.run([str(d/'check')]).returncode,0)

    def test_prefix_loop_preserves_allocation_tail(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'build') as tmp:
            d=Path(tmp)
            config=self.config(d,'''extern "C" float helper(float x) {
 if(x==0)return 1;
 float y=1;while(x>1){y=y+x;x=x/2;}return y+x;
}''')
            source=d/'input.c'
            source.write_text('extern float helper(float); void sample(int n,float x[n],float y[n]){for(int i=0;i<n;i++)y[i]=helper(x[i]);}')
            stages=R.raise_c(R.PRODUCTION,str(source),'sample',str(d/'ir'),[],90,frontend_support=config)
            for stage in ('cgeist','raise','debuf'):
                self.assertTrue(stages[stage].ok,stages[stage].stderr)
            log=d/'lower.log';log.write_text('')
            ll=E.lower(Path(stages['paths']['debuf']),d,'sample','sample',log)
            self.assertIsNotNone(ll,log.read_text()[-2000:])
            (d/'wrapper.c').write_text(subprocess.check_output([sys.executable,
                str(ROOT/'scripts/correctness/gen_wrapper.py'),str(source),'sample'],text=True))
            (d/'check.c').write_text('''#include <string.h>
void sample(int,float*,float*);void original(int,float*,float*);
int main(void){float x[16],a[16],b[16];
 for(int n=1;n<=16;n+=3){for(int i=0;i<16;i++){x[i]=i-2;a[i]=b[i]=-777;}
 sample(n,x,a);original(n,x,b);if(memcmp(a,b,sizeof(a)))return 1;}
 return 0;}
''')
            p=subprocess.run([E.CLANG,'-std=c++17','-O0','-ffp-contract=off',
                '-Dsample=original','-c',str(d/'ir.frontend.cpp'),'-o',str(d/'ref.o')],capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr)
            p=subprocess.run([E.CLANG,'-O0',str(d/'check.c'),str(d/'wrapper.c'),str(d/'ref.o'),
                str(ll),'-lm','-lstdc++',f'-L{E.MLIR_LIBDIR}',f'-Wl,-rpath,{E.MLIR_LIBDIR}',
                '-lmlir_c_runner_utils','-o',str(d/'check')],capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr)
            self.assertEqual(subprocess.run([str(d/'check')]).returncode,0)

    def test_missing_body_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'build') as tmp:
            d=Path(tmp)
            config=self.config(d,'extern "C" float helper(float);')
            source=d/'input.c';source.write_text('extern float helper(float); float sample(float x){return helper(x);}')
            stages=R.raise_c(R.PRODUCTION,str(source),'sample',str(d/'ir'),[],60,frontend_support=config)
            self.assertFalse(stages['cgeist'].ok)
            self.assertIn('not inlined',stages['cgeist'].stderr)
            self.assertNotIn('raise',stages)

    def test_dynamic_initialization_is_retained(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'build') as tmp:
            d=Path(tmp);source=d/'input.cpp'
            source.write_text('extern "C" float sample(float x){static const float a[2]={x,1};return a[0]+a[1];}')
            result=R.run(R.PRODUCTION.cgeist_cmd(str(source),'sample',str(d/'ir.mlir'),['-O3']),60)
            self.assertTrue(result.ok,result.stderr)
            self.assertIn('@init', (d/'ir.mlir').read_text())

    def test_boolean_table_uses_storage_width(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'build') as tmp:
            d=Path(tmp);source=d/'input.cpp'
            source.write_text('extern "C" bool sample(int i){static const bool a[2]={true,false};return a[i];}')
            result=R.run(R.PRODUCTION.cgeist_cmd(str(source),'sample',str(d/'ir.mlir'),['-O3']),60)
            self.assertTrue(result.ok,result.stderr)
            text=(d/'ir.mlir').read_text()
            self.assertIn('dense<[1, 0]>',text)
            self.assertNotIn('@init',text)

    def test_recursive_implementation_is_not_an_opaque_replacement(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'build') as tmp:
            d=Path(tmp)
            config=self.config(d,'''float inner(float x){return x<0?-inner(x):x*x;}
extern "C" float helper(float x){return inner(x);}
''')
            source=d/'input.c';source.write_text('extern float helper(float); float sample(float x){return helper(x);}')
            stages=R.raise_c(R.PRODUCTION,str(source),'sample',str(d/'ir'),[],60,frontend_support=config)
            self.assertFalse(stages['cgeist'].ok)
            self.assertIn('inner',stages['cgeist'].stderr)

    def test_builtin_constants_are_values(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'build') as tmp:
            d=Path(tmp);source=d/'input.c'
            source.write_text('float sample(float x){return x<0?__builtin_huge_valf():__builtin_nanf("0x19");}')
            result=R.run(R.PRODUCTION.cgeist_cmd(str(source),'sample',str(d/'ir.mlir'),['-O3']),60)
            self.assertTrue(result.ok,result.stderr)
            text=(d/'ir.mlir').read_text()
            self.assertNotIn('call',text)
            self.assertIn('0x7F800000',text)
            self.assertIn('0x7FC00019',text)

    def test_unsigned_wrap_is_not_converted_to_a_finite_loop(self):
        template='''module {
func.func @counter() -> i8 {
 %zero = arith.constant 0 : i8
 %one = arith.constant 1 : i8
 %bound = arith.constant BOUND : i32
 %r = scf.while (%iv = %zero) : (i8) -> i8 {
   %wide = arith.extui %iv : i8 to i32
   %keep = arith.cmpi sle, %wide, %bound : i32
   scf.condition(%keep) %iv : i8
 } do {
 ^bb0(%iv: i8):
   %next = arith.addi %iv, %one : i8
   scf.yield %next : i8
 }
 return %r : i8
}
}'''
        for bound, remains in ((7,False),(255,True)):
            p=subprocess.run([R.POLYGEIST_OPT,'--canonicalize-scf-for'],
                input=template.replace('BOUND',str(bound)),capture_output=True,text=True,timeout=30)
            self.assertEqual(p.returncode,0,p.stderr)
            self.assertEqual('scf.while' in p.stdout,remains)

    def test_early_returns_and_coupled_recurrence_execution(self):
        bodies = [
            '''#include <cmath>
extern "C" float helper(float x) {
 if(std::isinf(x))return 0;
 if(std::abs(x)<0.5f)return 1+x*x;
 return std::sin(x)/x;
}''',
            '''extern "C" float helper(float x) {
 float p=0,q=0,a=1;
 for(unsigned char i=1;i<5;i++){p=q;q=a;a=x*q-p+i;}
 return a-p;
}''',
            '''extern "C" float helper(float x) {
 if(x==0)return -2;
 if(x==1)return 3;
 if(x<0 || x>1)return 4;
 return x*x;
}''',
            '''extern "C" float helper(float x) {
 if(x<0){if(-x<0)return 7;return -x;}
 if(x>=0)return x;
 return 11;
}''',
            '''float inner(float x) {return x<0?-inner(-x):x*x;}
extern "C" float helper(float x) {return inner(x);}''',
            '''#include <cmath>
extern "C" float helper(float x) {
 if(!std::isfinite(x) || x < -3 || x > 3)return x;
 switch((int)x){case -2:return x*x; case 0:return -x;
 case 1:return x+2;default:return x/3;}
}''']
        self.execute_bodies(bodies)

    def test_structured_bindings_execute_once_and_preserve_references(self):
        self.execute_bodies([
            '''#include <tuple>
extern "C" float helper(float x) {
 auto [value, count] = std::make_tuple(x, 3);
 return value + count;
}''',
            '''struct Pair {float value;float offset;};
extern "C" float helper(float x) {
 Pair p{x,3};auto &[a,b]=p;a+=b;return p.value;
}''',
            '''#include <tuple>
struct Record {float value;int *calls;};
namespace std {
 template<> struct tuple_size<Record> : integral_constant<size_t,2> {};
 template<size_t I> struct tuple_element<I,Record> {using type=float;};
}
template<std::size_t I> float get(const Record &r){++*r.calls;return r.value+I;}
extern "C" float helper(float x) {
 int calls=0;Record r{x,&calls};auto [value,unused]=r;
 return value+calls;
}''',
            '''#include <tuple>
extern "C" float helper(float x) {
 float offset=3;auto [a,b]=std::tie(x,offset);a+=1;return x+b;
}'''], frontend_llvm=True)

    def execute_bodies(self, bodies, frontend_llvm=False):
        for body in bodies:
            with self.subTest(body=body), tempfile.TemporaryDirectory(dir=ROOT/'build') as tmp:
                d=Path(tmp);config=Path(self.config(d,body))
                data=json.loads(config.read_text());data['unroll_threshold']=16
                config.write_text(json.dumps(data))
                source=d/'input.c';source.write_text('extern float helper(float); float sample(float x){return helper(x);}')
                stages=R.raise_c(R.PRODUCTION,str(source),'sample',str(d/'ir'),[],60,frontend_support=str(config))
                for stage in ('cgeist','raise','debuf'):
                    self.assertTrue(stages[stage].ok,stages[stage].stderr)
                self.assertNotIn('llvm.mlir.undef',Path(stages['paths']['debuf']).read_text())
                log=d/'lower.log';log.write_text('')
                if frontend_llvm:
                    # Exercise C++ aggregate layout through the frontend's
                    # normal C ABI lowering. Import execution's descriptor
                    # pipeline currently requires scalarized aggregates.
                    ll=d/'sample_kernel.ll'
                    p=subprocess.run([R.CGEIST,str(d/'ir.frontend.cpp'),
                        '--resource-dir=/usr/lib/clang/14','--function=sample_impl',
                        '-Dsample=sample_impl','-Dhelper=frontend_helper',
                        '-std=c++17','-O3','-S','--emit-llvm',
                        '-o',str(ll)],capture_output=True,text=True)
                    self.assertEqual(p.returncode,0,p.stderr)
                else:
                    ll=E.lower(Path(stages['paths']['debuf']),d,'sample','sample',log)
                self.assertIsNotNone(ll,log.read_text()[-2000:])
                (d/'check.c').write_text('''#include <math.h>
#include <stdint.h>
#include <string.h>
float sample_impl(float);float original(float);
int main(void){uint32_t bits[]={0,0x80000000,1,0x80000001,0x3e800000,0xbe800000,0x3f000000,0xbf000000,0x3f800000,0xbf800000,0x40000000,0x7f800000,0xff800000,0x7fc00019};
for(int i=0;i<sizeof(bits)/4;i++){float x;memcpy(&x,bits+i,4);float a=original(x),b=sample_impl(x);
 if(memcmp(&a,&b,4)&&!(isnan(a)&&isnan(b)))return 1;}return 0;}
''')
                flags=['-O0','-ffp-contract=off','-fno-fast-math']
                p=subprocess.run([E.CLANG,*flags,'-std=c++17','-Dsample=original',
                    '-c',str(d/'ir.frontend.cpp'),'-o',str(d/'ref.o')],capture_output=True,text=True)
                self.assertEqual(p.returncode,0,p.stderr)
                p=subprocess.run([E.CLANG,*flags,str(d/'check.c'),str(d/'ref.o'),str(ll),
                    '-lm','-lstdc++',f'-L{E.MLIR_LIBDIR}',f'-Wl,-rpath,{E.MLIR_LIBDIR}',
                    '-lmlir_c_runner_utils','-o',str(d/'check')],capture_output=True,text=True)
                self.assertEqual(p.returncode,0,p.stderr)
                self.assertEqual(subprocess.run([str(d/'check')]).returncode,0)


if __name__=='__main__': unittest.main()
