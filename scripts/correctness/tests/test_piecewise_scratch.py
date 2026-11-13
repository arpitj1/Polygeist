"""Small initialized scratch tables retain coverage and exact read values."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from region_match.equations import FSel,Reject
from region_match.mlir_parse import parse_module
from region_match.symexec import execute
from test_ordered_overwrite import system
from test_bounded_integer_normalization import evaluate


SOURCE='''module {
 func.func @coefficient_table(%out: memref<3xi32>) {
  %one = arith.constant 1 : i32
  %two = arith.constant 2 : i32
  %table = memref.alloca() : memref<3xi32>
  affine.store %one, %table[0] : memref<3xi32>
  affine.store %two, %table[1] : memref<3xi32>
  affine.store %one, %table[2] : memref<3xi32>
  affine.for %i = 0 to 3 {
   %v = affine.load %table[%i] : memref<3xi32>
   affine.store %v, %out[%i] : memref<3xi32>
  }
  return
 }
}'''


def selected(f,index):
    if isinstance(f,FSel):
        assert f.cond[0]=='aff'
        good=all((c.aff.const+sum(a*index for _,a in c.aff.terms))==0 if c.kind=='eq'
                 else (c.aff.const+sum(a*index for _,a in c.aff.terms))>=0 for c in f.cond[1])
        return selected(f.a if good else f.b,index)
    return evaluate(f,{})


class PiecewiseScratch(unittest.TestCase):
    def test_every_initialized_element_preserves_value(self):
        equations=system(SOURCE).equations
        self.assertEqual(len(equations),1)
        self.assertEqual([selected(equations[0].formula,i) for i in range(3)],[1,2,1])

    def test_uninitialized_element_does_not_get_a_default(self):
        missing=SOURCE.replace('  affine.store %two, %table[1] : memref<3xi32>\n','')
        with self.assertRaises(Reject):system(missing)

    def test_output_layout_factoring_uses_generic_iteration_bounds(self):
        text='''#m = affine_map<(d0,d1)->(0,d0*8+d1+9)>
#i = affine_map<(d0,d1)->(d0,d1)>
module {
 func.func @layout(%input: memref<4x8xi8>, %output: memref<4x8xi8>) {
  %two = arith.constant 2 : index
  %six = arith.constant 6 : index
  %in = bufferization.to_tensor %input : memref<4x8xi8>
  %out = bufferization.to_tensor %output : memref<4x8xi8>
  %iv = polygeist.submap(%in, %two, %six) {map = #m} : (tensor<4x8xi8>, index, index) -> tensor<2x6xi8>
  %ov = polygeist.submap(%out, %two, %six) {map = #m} : (tensor<4x8xi8>, index, index) -> tensor<2x6xi8>
  %result = linalg.generic {indexing_maps = [#i, #i], iterator_types = ["parallel", "parallel"]} ins(%iv : tensor<2x6xi8>) outs(%ov : tensor<2x6xi8>) {
  ^bb0(%v: i8, %old: i8):
   linalg.yield %v : i8
  } -> tensor<2x6xi8>
  return
 }
}'''
        functions,aliases=parse_module(text)
        program=execute(functions[0],aliases)
        statement=next(s for s in program.stmts if s.origin=='linalg.generic')
        self.assertEqual([a.const for a in statement.idx],[1,1])
        self.assertEqual([list(a.terms) for a in statement.idx],[[('g1_0',1)],[('g1_1',1)]])


if __name__=='__main__':unittest.main()
