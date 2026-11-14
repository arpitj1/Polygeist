"""Discover actual C definitions with nested expressions in array extents."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts/correctness/lib_v2'))
import abi_v2 as A

class VlaSignature(unittest.TestCase):
    def parse(self, source):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'raised').mkdir()
            (root/'raised/check.c').write_text(source)
            A.spec_c_prototype.cache_clear()
            try:
                with patch.object(A,'SPEC_DIR',root):
                    return A.spec_c_prototype('check')
            finally:
                A.spec_c_prototype.cache_clear()

    def test_casts_and_parenthesized_extents(self):
        ret,args=self.parse('void check(int n,int offset,float x[n],float y[(long)n+(offset)]) {y[offset]=x[0];}')
        self.assertEqual(ret,'void')
        self.assertEqual([(a.name,a.kind,a.rank) for a in args],
            [('n','int',0),('offset','int',0),('x','array',1),('y','array',1)])

    def test_declaration_before_definition_and_parentheses_in_string(self):
        ret,args=self.parse('''void check(int n,float x[n][sizeof(")")]);
void check(int n,float x[n][sizeof(")")]) { x[0][0]=0; }''')
        self.assertEqual(ret,'void')
        self.assertEqual(args[1].rank,2)
        self.assertEqual(args[1].ctype,'float')

    def test_declaration_alone_is_not_a_definition(self):
        with self.assertRaisesRegex(A.AbiError,'no definition'):
            self.parse('void check(int n,float x[(long)n]);')

    def test_function_pointer_parameter_remains_unsupported(self):
        with self.assertRaisesRegex(A.AbiError,'cannot classify'):
            self.parse('void check(int (*callback)(int)) {}')

if __name__=='__main__':unittest.main()
