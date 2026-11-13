"""Range-checked normalization agrees with integer bitvector execution."""
from pathlib import Path
import random
import re
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from region_match.affine import Aff
from region_match.equations import FRead,FConst,FBin,FUn,FFold,FAcc
from region_match.integer_normalize import normalize


def evaluate(f, values, env=None, acc=None):
    env=env or {}
    if isinstance(f,FConst):return int(f.v)
    if isinstance(f,FRead):
        index=f.idx[0].const+sum(c*env[v] for v,c in f.idx[0].terms)
        return values[f.arr][index]
    if isinstance(f,FAcc):return acc
    if isinstance(f,FFold):
        out=evaluate(f.default,values,env,acc)
        for i in range(f.lo.const,f.hi.const):
            out=evaluate(f.body,values,dict(env,**{f.var:i}),out)
        return out
    if isinstance(f,FUn):
        value=evaluate(f.a,values,env,acc)
        if f.op=='integer.value':return value
        if f.op.startswith('integer.floor_div:'):return value//int(f.op.split(':')[1])
        if f.op.startswith('unsigned-read:i'):return value & (2**int(f.op.split(':i')[1])-1)
        cast=re.fullmatch(r'arith\.(extui|trunci):i(\d+) to i(\d+)',f.op)
        if cast:return value & (2**int(cast[2] if cast[1]=='extui' else cast[3])-1)
        raise AssertionError(f.op)
    if isinstance(f,FBin):
        a,b=evaluate(f.a,values,env,acc),evaluate(f.b,values,env,acc)
        if f.op=='add':return a+b
        if f.op=='mul':return a*b
        op,bits=f.op.split(':i');bits=int(bits);mask=2**bits-1
        a &= mask;b &= mask
        if op=='arith.addi':return (a+b)&mask
        if op=='arith.muli':return (a*b)&mask
        if op=='arith.subi':return (a-b)&mask
        if op=='arith.shli':return (a<<b)&mask
        if op=='arith.shrui':return a>>b
        if op=='arith.shrsi':return ((a-2**bits if a>=2**(bits-1) else a)>>b)&mask
        if op=='arith.divui':return a//b
        if op=='arith.divsi':return int((a-2**bits if a>=2**(bits-1) else a)/(b-2**bits if b>=2**(bits-1) else b))&mask
    raise AssertionError(type(f))


def byte(name='x',index=None):
    return FUn('arith.extui:i8 to i32',FRead(name,'in',[index or Aff.cst(0)]))


class BoundedIntegerNormalization(unittest.TestCase):
    def test_fixed_point_casts_and_rounding_exhaustive_two_bytes(self):
        x,y=byte(),byte('y')
        shifted=FBin('arith.shli:i32',x,FConst(8))
        raw=FUn('arith.trunci:i32 to i16',shifted)
        quarter=FBin('arith.shrsi:i32',FUn('arith.extui:i16 to i32',raw),FConst(2))
        f=FBin('arith.shrsi:i32',FBin('arith.addi:i32',quarter,FBin('arith.muli:i32',y,FConst(128))),FConst(8))
        n=normalize(f,[])
        self.assertNotEqual(f,n)
        for a in range(256):
            for b in range(256):
                values={'x':[a],'y':[b]}
                self.assertEqual(evaluate(f,values),evaluate(n,values))

    def test_overflow_signed_shift_and_lossy_narrowing_stay_typed(self):
        x=FRead('x','in',[Aff.cst(0)])
        full=FUn('unsigned-read:i32',x)
        expressions=[FBin('arith.addi:i32',full,FConst(1)),
                     FBin('arith.shrsi:i32',full,FConst(1)),
                     FUn('arith.trunci:i32 to i8',full),
                     FBin('arith.shli:i32',byte(),FConst(32))]
        for f in expressions:
            with self.subTest(expression=f):self.assertEqual(normalize(f,[]),f)

    def test_positive_uint32_logical_shift_preserves_high_bit(self):
        f=FBin('arith.shrui:i32',FUn('unsigned-read:i32',FRead('x','in',[Aff.cst(0)])),FConst(4))
        n=normalize(f,[])
        rng=random.Random(0)
        for value in [0,2**31-1,2**31,2**32-1]+[rng.getrandbits(32) for _ in range(500)]:
            self.assertEqual(evaluate(f,{'x':[value]}),evaluate(n,{'x':[value]}))

    def test_ordered_recurrence_is_not_reassociated(self):
        body=FBin('arith.addi:i32',FBin('arith.muli:i32',FAcc(),FConst(3)),byte(index=Aff.var('r')))
        f=FFold('r',Aff.cst(0),Aff.cst(3),[],body,FConst(0))
        n=normalize(f,[])
        self.assertNotEqual(f,n)
        for values in [[1,2,3],[255,0,1],[0,255,0]]:
            self.assertEqual(evaluate(f,{'x':values}),evaluate(n,{'x':values}))
        self.assertEqual(evaluate(n,{'x':[1,2,3]}),18)

    def test_rounding_constants_remain_distinct(self):
        f=lambda bias: FBin('arith.shrsi:i32',FBin('arith.addi:i32',byte(),FConst(bias)),FConst(4))
        self.assertNotEqual(normalize(f(8),[]),normalize(f(7),[]))
        self.assertEqual(normalize(FConst(1.0),[]),FConst(1.0))

    def test_bounded_reciprocal_matches_division_for_every_possible_sum(self):
        total=FConst(4)
        for i in range(9):total=FBin('arith.addi:i32',total,byte(index=Aff.cst(i)))
        division=FBin('arith.divsi:i32',total,FConst(9))
        magic=lambda k:FBin('arith.shrsi:i32',FBin('arith.muli:i32',total,FConst(k)),FConst(23))
        normalized=normalize(division,[])
        self.assertEqual(normalize(magic(932068),[]),normalized)
        self.assertNotEqual(normalize(magic(932067),[]),normalized)
        for value in range(9*255+1):
            values={'x':[min(255,max(0,value-i*255)) for i in range(9)]}
            self.assertEqual(evaluate(division,values),evaluate(magic(932068),values))
            self.assertEqual(evaluate(normalized,values),evaluate(division,values))

    def test_division_rejects_zero_negative_and_signed_high_bit(self):
        full=FUn('unsigned-read:i32',FRead('x','in',[Aff.cst(0)]))
        for f in [FBin('arith.divsi:i32',full,FConst(9)),
                  FBin('arith.divui:i32',byte(),FConst(0)),
                  FBin('arith.divsi:i32',byte(),FConst(-9))]:
            self.assertEqual(normalize(f,[]),f)


if __name__=='__main__':unittest.main()
