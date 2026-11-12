"""Exact scalar-DAG inversion with typed overflow and ambiguity rejection."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'scripts/correctness'),str(ROOT/'scripts/correctness/lib_v2')]
import match_v2 as M
import mlir_lite as ML


def matcher(body):
    m=M.Matcher.__new__(M.Matcher)
    m.L=ML.Module('''module {
func.func @shape(%n:i32,%r:i32) -> index {
 %ni=arith.index_cast %n : i32 to index
 %ri=arith.index_cast %r : i32 to index
 %two=arith.constant 2 : index
 BODY
 return %out : index
}
}'''.replace('BODY',body))
    m.A=ML.Module('module {}')
    m.spec=SimpleNamespace(params={'%n':'i32','%r':'i32'},yield_op=m.L.ops[-1])
    m.literal_allowed=lambda *args: True
    return m


class IntegerParameterInference(unittest.TestCase):
    def test_remainder_inference_retains_typed_expression(self):
        for body in ['%twice=arith.muli %ni,%two : index\n %out=arith.addi %twice,%ri : index',
                     '%out=affine.apply affine_map<(d0)[s0] -> (2*d0+s0)>(%ni)[%ri]']:
            m=matcher(body);env=M.Env();env.vals['%n']=('int',5,'i32')
            result=m.infer_integer_parameter('%out',11,env)
            self.assertIsNotNone(result)
            self.assertEqual(result.vals['%r'],('int',1,'i32'))
            self.assertEqual(m.bound_integer('%out',result),11)
            self.assertNotIn('%r',env.vals)

    def test_ambiguous_nonlinear_and_fractional_candidates_reject(self):
        for body,target in [('%twice=arith.muli %ni,%two : index\n %out=arith.addi %twice,%ri : index',11),
                            ('%out=arith.muli %ni,%ri : index',12),
                            ('%out=arith.muli %ni,%two : index',11)]:
            self.assertIsNone(matcher(body).infer_integer_parameter('%out',target,M.Env()))

    def test_algebraic_cancellation_cannot_hide_intermediate_overflow(self):
        m=matcher('''%max=arith.constant 2147483647 : index
 %large=arith.addi %ni,%max : index
 %out=arith.subi %large,%max : index''')
        self.assertIsNone(m.infer_integer_parameter('%out',1,M.Env()))


if __name__=='__main__':unittest.main()
