"""Unsigned scalar ABI contracts preserve all 32 bits and their bounds."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lib_v2'))
import abi_v2 as A
import gen_v2_runtime as G
import typed_wrapper as T


class UnsignedWrapperScalar(unittest.TestCase):
    def contract(self, minimum=1, maximum=4294967295, wrapper_type='unsigned int'):
        args = [A.SpecArg('value', 'int', ctype='unsigned int')]
        cfg = dict(schema=1, kind='typed_c_wrapper', entry='wrapper',
                   arguments=['value'], arrays={},
                   bounds={'value': {'min': minimum, 'max': maximum}})
        target = dict(library='Vendor', call='operation', backends=['pva'],
                      runtime_adapter=cfg)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'runtime').mkdir()
            (root/'runtime/polygeist_cublas_rt.h').write_text(
                '/* POLYGEIST_TYPED_VENDOR_ENTRY Vendor operation wrapper */\n'
                f'void wrapper({wrapper_type} value);\n')
            with patch.object(A, 'REPO', root), patch.object(A, 'CLANG', 'gcc'):
                params, _, checked = T.contract(A, 'reference', target, 'void', args)
        return A.CallAbi('uint32', 'Vendor', 'operation', params, [],
                         runtime_adapter=checked)

    def test_invalid_bounds_and_signedness_fail_closed(self):
        for kwargs in ({'minimum': -1}, {'maximum': 4294967296},
                       {'wrapper_type': 'int'}):
            with self.subTest(kwargs=kwargs), self.assertRaises(A.AbiError):
                self.contract(**kwargs)

    def test_generated_call_preserves_unsigned_boundary_values(self):
        ca = self.contract()
        self.assertIn('uint32_t value', G.proto(ca))
        source = '''#include <stdint.h>
#include <stdlib.h>
static uint32_t received;
static void wrapper(uint32_t value){received=value;}
static void polygeist_v2_fatal(const char*a,const char*b){(void)a;(void)b;exit(73);}
static double polygeist_v2_now_ms(void){return 0;}
static void polygeist_v2_done(const char*a,const char*b,double t){(void)a;(void)b;(void)t;}
''' + T.generate(ca, G.proto) + '''
int main(int argc,char**argv){
 (void)argv;uint32_t values[]={1,2147483647u,2147483648u,4294967295u};
 if(argc>1){polygeist_v2_uint32(0);return 99;}
 for(unsigned i=0;i<4;++i){polygeist_v2_uint32(values[i]);if(received!=values[i])return 1;}
 return 0;
}
'''
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'check.c').write_text(source)
            compiled=subprocess.run(['cc','-Wall','-Wextra','-Werror',
                str(root/'check.c'),'-o',str(root/'check')],capture_output=True,text=True)
            self.assertEqual(compiled.returncode,0,compiled.stderr)
            self.assertEqual(subprocess.run([str(root/'check')]).returncode,0)
            self.assertEqual(subprocess.run([str(root/'check'),'zero']).returncode,73)


if __name__ == '__main__':
    unittest.main()
