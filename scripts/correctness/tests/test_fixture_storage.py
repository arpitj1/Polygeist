"""Execute fixture allocation/guard checks with host CUDA stubs and typed callees."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import aten_fixture_batch as B


class FixtureStorage(unittest.TestCase):
    def execute(self, element, kind, dtype=None, corrupt=False, metadata=None):
        cfg={'dims':{},'args':[('x',kind,'19',False,''),
             ('y','iptr' if kind=='bptr' else kind,'19',True,'')]}
        if dtype:cfg['dtype']=dtype
        if metadata is not None:cfg['integer_array_values']={'x':metadata}
        output='int32_t' if kind=='bptr' else element
        carrier='double' if dtype else 'float'
        code=B.harness('probe',cfg)
        code+=f'''
void native_reference({element} *x,{output} *y){{for(int i=0;i<19;++i)y[i]=x[i];}}
void probe({element} *x,{output} *y){{native_reference(x,y);CORRUPT}}
void native_fixture({carrier} **p,const {carrier} *s,const size_t *n,int count){{
 (void)s;(void)n;(void)count;native_reference(({element}*)p[0],({output}*)p[1]);}}
'''.replace('CORRUPT','((unsigned char*)y)[19*sizeof(*y)]^=1;' if corrupt else '')
        if metadata is not None:
            check='static const int want[]={'+','.join(map(str,metadata))+'};'
            check+=f'for(int i=0;i<19;++i)if(x[i]!=want[i%{len(metadata)}])exit(41);'
            code=code.replace('){for(int i=0;i<19;++i)y[i]=x[i];}','){'+check+'for(int i=0;i<19;++i)y[i]=x[i];}')
        header='''#include <stdlib.h>
#include <string.h>
typedef int cudaError_t;
enum {cudaSuccess=0,cudaMemcpyHostToDevice=1,cudaMemcpyDeviceToHost=2};
static int cudaMalloc(void **p,size_t n){*p=malloc(n);return !*p;}
static int cudaMemcpy(void *d,const void *s,size_t n,int mode){(void)mode;memcpy(d,s,n);return 0;}
static int cudaDeviceSynchronize(void){return 0;}
static const char *cudaGetErrorString(int e){(void)e;return "stub";}
'''
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);(d/'cuda_runtime_api.h').write_text(header);(d/'check.c').write_text(code)
            subprocess.run(['gcc','-std=c11','-D_POSIX_C_SOURCE=200809L','-O2',
                '-fsanitize=undefined','-I'+str(d),str(d/'check.c'),'-lm','-o',str(d/'check')],
                check=True,capture_output=True)
            for mode in ['normal','edge','zeros']:
                result=subprocess.run([str(d/'check'),mode],text=True,capture_output=True)
                self.assertEqual(result.returncode,3 if corrupt else 0,result.stdout+result.stderr)
                self.assertIn('count=19',result.stdout)
                if corrupt:self.assertIn('guardbad=1',result.stdout)

    def test_mixed_int8_int32_arrays_have_correct_byte_sizes(self):
        self.execute('int8_t','bptr')

    def test_existing_float_double_and_int32_storage(self):
        for element,kind,dtype in [('float','ptr',None),('double','ptr','double'),('int32_t','iptr',None)]:
            with self.subTest(element=element):self.execute(element,kind,dtype)

    def test_one_byte_output_overrun_is_detected(self):
        self.execute('int8_t','bptr',corrupt=True)

    def test_index_metadata_is_valid_in_all_input_modes(self):
        self.execute('int32_t','iptr',metadata=[-1,0,1,64,32,63])


if __name__=='__main__':unittest.main()
