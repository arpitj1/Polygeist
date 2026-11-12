"""Region execution retains scalar loop carries and their declared storage type."""
from pathlib import Path
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from region_match.mlir_parse import parse_module
from region_match.symexec import execute
from region_match.equations import build_system, Reject
from region_match.matcher import Binding, compare_systems


class RegionScalarCarry(unittest.TestCase):
    def setUp(self):
        mode=patch.dict(os.environ,POLYGEIST_NUMERIC_MODEL='relaxed')
        mode.start();self.addCleanup(mode.stop)

    def program(self,ty,seed):
        source='''module {
 func.func @fold(%x: memref<8xTYPE>, %out: memref<1xTYPE>) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %end = arith.constant 8 : index
  %seed = arith.constant SEED : TYPE
  %result = scf.for %i = %zero to %end step %one iter_args(%acc = %seed) -> (TYPE) {
   %value = memref.load %x[%i] : memref<8xTYPE>
   %next = arith.ADD %acc, %value : TYPE
   scf.yield %next : TYPE
  }
  memref.store %result, %out[%zero] : memref<1xTYPE>
  return
 }
}'''.replace('TYPE',ty).replace('SEED',seed).replace('ADD','addi' if ty.startswith('i') else 'addf')
        functions,aliases=parse_module(source);p=execute(functions[0],aliases)
        self.assertEqual([b.elem for b in p.buffers if b.kind=='scratch'],[ty])
        return p

    def test_float_carry_is_proved_without_changing_seed(self):
        def system(seed):
            p=self.program('f32',seed)
            return build_system(p,set(range(len(p.top_ops))))
        a=system('3.0');b=system('3.0');wrong=system('4.0')
        binding=Binding({k:(k,k if k in a.outputs else None,tuple(range(v.rank)))
                         for k,v in a.arrays.items()}, {}, {})
        self.assertTrue(compare_systems(a,b,binding).ok)
        self.assertFalse(compare_systems(a,wrong,binding).ok)

    def test_integer_carry_does_not_use_a_real_arithmetic_reduction_summary(self):
        p=self.program('i32','3')
        with self.assertRaisesRegex(Reject,'accumulation is not'):
            build_system(p,set(range(len(p.top_ops))))

    def test_float_carry_keeps_declared_precision(self):
        self.program('f32','0.0')
        self.program('f64','0.0')


if __name__=='__main__':unittest.main()
