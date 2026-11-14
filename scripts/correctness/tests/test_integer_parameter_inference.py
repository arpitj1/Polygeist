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


    def test_unsigned_index_cast_inference_is_nonnegative_and_exact(self):
        m=matcher('%out=arith.index_castui %n : i32 to index')
        result=m.infer_integer_parameter('%out',31,M.Env())
        self.assertIsNotNone(result)
        self.assertEqual(result.vals['%n'],('int',31,'i32'))
        self.assertIsNone(m.infer_integer_parameter('%out',-1,M.Env()))
        self.assertIsNone(m.infer_integer_parameter('%out',2147483648,M.Env()))
        env=M.Env();env.vals['%n']=('int',-1,'i32')
        self.assertIsNone(m.bound_integer('%out',env))


    def test_captured_casted_integer_literal_uses_same_typed_solver(self):
        m=matcher('%out=arith.index_castui %n : i32 to index')
        results=list(m.match_cap('%out',('Lit',64),None,M.Env()))
        self.assertEqual(len(results),1)
        self.assertEqual(results[0].vals['%n'],('int',64,'i32'))
        for value in [-1,2147483648,float('inf'),float('nan')]:
            self.assertEqual(list(m.match_cap('%out',('Lit',value),None,M.Env())),[])

    def test_captured_expression_retains_existing_binding_and_overflow_checks(self):
        m=matcher('%out=arith.addi %ni,%two : index')
        env=M.Env();env.vals['%n']=('int',5,'i32')
        self.assertEqual(len(list(m.match_cap('%out',('Lit',7),None,env))),1)
        self.assertEqual(list(m.match_cap('%out',('Lit',8),None,env)),[])
        m=matcher('%large=arith.addi %ni,%two : index\n %out=arith.subi %large,%two : index')
        self.assertEqual(list(m.match_cap('%out',('Lit',2147483647),None,M.Env())),[])


if __name__=='__main__':unittest.main()
