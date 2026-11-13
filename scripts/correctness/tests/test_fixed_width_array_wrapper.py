"""A generated descriptor bridge preserves fixed-width integer element sizes."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gen_wrapper as W


class FixedWidthArrays(unittest.TestCase):
    def test_byte_matrix_and_counter_output_execute_through_descriptor(self):
        source='void test(const uint8_t input[2][3], uint32_t output[2]) { }'
        bridge=W.gen_wrapper('test',W.parse_signature(source,'test'))
        program=bridge+'''
void test_impl(unsigned char *ab,unsigned char *a,int64_t off,
 int64_t rows,int64_t cols,int64_t rs,int64_t cs,
 uint32_t *bb,uint32_t *b,int64_t bo,int64_t n,int64_t bs) {
 if(ab!=a || bb!=b || off || bo || rows!=2 || cols!=3 || n!=2 || rs!=3 || cs!=1 || bs!=1)
  b[0]=0;
 else {b[0]=a[0]+a[3];b[1]=a[2]+a[5];}
}
int main(void) {unsigned char a[]={128,0,255,129,0,254};uint32_t b[2]={0};
 test(a,b);return b[0]!=257 || b[1]!=509;}
'''
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'test.c').write_text(program)
            subprocess.run(['cc','-std=c11','-Werror',str(root/'test.c'),'-o',str(root/'test')],check=True,capture_output=True)
            self.assertEqual(subprocess.run([str(root/'test')]).returncode,0)

    def test_same_integer_types_for_pointer_and_array_forms(self):
        for ctype in ['uint8_t','int8_t','uint16_t','int16_t','uint32_t','int32_t']:
            array=W.parse_signature('void f(int n, '+ctype+' x[n]) {}','f')
            pointer=W.parse_signature('void f(int n, '+ctype+' *x) {}','f')
            self.assertEqual(array,pointer)
            bridge=W.gen_wrapper('f',array)
            if '16' in ctype:
                self.assertIn(ctype+' *x',bridge)


if __name__=='__main__':unittest.main()
