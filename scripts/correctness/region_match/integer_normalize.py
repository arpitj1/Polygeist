"""Exact normalization of bounded integer expressions before EqSat encoding.

Only unsigned, nonoverflowing 8/16/32-bit operations are normalized. Opaque
unsigned reads retain their storage width, right shifts retain floor rounding,
and ordered finite recurrences expand in their original execution order.
Unknown ranges, signed extensions, wraparound and large recurrences stay typed.
No application names or library identities participate.
"""
from dataclasses import dataclass
from functools import reduce
from math import gcd
import re

from .affine import Aff
from .equations import (FRead, FConst, FIndex, FBin, FUn, FFold, FSel,
                        fsubst, replace_acc, _simplify_sel)


@dataclass
class Form:
    constant: int
    terms: dict
    lower: int
    upper: int

    def expression(self):
        terms = [FConst(self.constant)] if self.constant or not self.terms else []
        for atom, coefficient in sorted(self.terms.items(), key=lambda p: repr(p[0])):
            terms.append(atom if coefficient == 1 else FBin('mul', FConst(coefficient), atom))
        return reduce(lambda a, b: FBin('add', a, b), terms)

    def scale(self, k):
        lo, hi = self.lower*k, self.upper*k
        return Form(self.constant*k, {a:c*k for a,c in self.terms.items() if c*k}, min(lo,hi),max(lo,hi))

    def plus(self, other):
        terms = dict(self.terms)
        for a,c in other.terms.items(): terms[a]=terms.get(a,0)+c
        return Form(self.constant+other.constant,{a:c for a,c in terms.items() if c},
                    self.lower+other.lower,self.upper+other.upper)


def bounded(f, budget):
    budget[0] -= 1
    if budget[0] < 0: return None
    if isinstance(f,FConst) and f.v.is_integer() and abs(f.v) <= 2**32:
        n=int(f.v);return Form(n,{},n,n)
    if isinstance(f,FIndex) and f.value.is_const():
        n=f.value.const;return Form(n,{},n,n) if abs(n)<=2**32 else None
    if isinstance(f,FUn):
        cast=re.fullmatch(r'arith\.(extui|trunci):i(8|16|32) to i(8|16|32)',f.op)
        unsigned=re.fullmatch(r'unsigned-read:i(8|16|32)',f.op)
        if isinstance(f.a,FRead) and (unsigned or cast and cast[1]=='extui'):
            bits=int(unsigned[1] if unsigned else cast[2])
            if cast and int(cast[3]) <= bits: return None
            atom=FUn('unsigned-read:i'+str(bits),f.a)
            return Form(0,{atom:1},0,2**bits-1)
        if not cast: return None
        a=bounded(f.a,budget)
        bits=int(cast[2] if cast[1]=='extui' else cast[3])
        if a and 0<=a.lower<=a.upper<2**bits: return a
        return None
    if not isinstance(f,FBin): return None
    operation=re.fullmatch(r'arith\.(addi|subi|muli|shli|shrsi|shrui|divsi|divui):i(8|16|32)',f.op)
    if not operation: return None
    a,b=bounded(f.a,budget),bounded(f.b,budget)
    if a is None or b is None: return None
    op,bits=operation[1],int(operation[2]);mask=2**bits-1
    if not (0<=a.lower<=a.upper<=mask and 0<=b.lower<=b.upper<=mask): return None
    if op=='addi': result=a.plus(b)
    elif op=='subi': result=a.plus(b.scale(-1))
    elif op=='muli':
        if not a.terms: result=b.scale(a.constant)
        elif not b.terms: result=a.scale(b.constant)
        else: return None
    elif op in ('divsi','divui'):
        if b.terms or b.constant<=0: return None
        if op=='divsi' and max(a.upper,b.upper)>=2**(bits-1): return None
        result=floor_divide(a,b.constant)
    else:
        if b.terms or not 0<=b.constant<bits: return None
        divisor=2**b.constant
        if op=='shli': result=a.scale(divisor)
        else:
            if op=='shrsi' and a.upper>=2**(bits-1): return None
            result=floor_divide(a,divisor)
    return result if 0<=result.lower<=result.upper<=mask else None


def floor_divide(a,divisor):
    """Canonicalize exact integer division, including bounded reciprocals.

    A constant multiply followed by a shift can implement integer division.
    A candidate reciprocal is accepted only after checking EVERY integer in
    the proven input interval (at most 4096), not sample inputs. Wider or
    unknown intervals keep their original multiply and denominator.
    """
    common=reduce(gcd,[divisor,abs(a.constant),*[abs(c) for c in a.terms.values()]])
    inner=Form(a.constant//common,{x:c//common for x,c in a.terms.items()},
               a.lower//common,a.upper//common)
    divisor//=common
    if divisor==1:return inner
    factor=reduce(gcd,[abs(inner.constant),*[abs(c) for c in inner.terms.values()]])
    if factor>1:
        candidate=(divisor+factor//2)//factor
        lo=(inner.lower+factor-1)//factor;hi=inner.upper//factor
        if (candidate>0 and 0<=hi-lo<4096 and all(
                factor*x//divisor==x//candidate for x in range(lo,hi+1))):
            inner=Form(inner.constant//factor,{x:c//factor for x,c in inner.terms.items()},lo,hi)
            divisor=candidate
    if divisor==1:return inner
    atom=FUn('integer.floor_div:'+str(divisor),inner.expression())
    return Form(0,{atom:1},inner.lower//divisor,inner.upper//divisor)


def expand(f, ctx, budget):
    budget[0]-=1
    if budget[0]<0: raise ValueError('finite recurrence expansion budget')
    if isinstance(f,FFold) and f.lo.is_const() and f.hi.is_const() and 0<=f.hi.const-f.lo.const<=16:
        value=expand(f.default,ctx,budget)
        for i in range(f.lo.const,f.hi.const):
            substitution={f.var:Aff.cst(i)}
            body=replace_acc(fsubst(f.body,substitution),value)
            guards=tuple(c.subst(substitution) for c in f.guard)
            body=_simplify_sel(FSel(('aff',guards),body,value),ctx)
            value=expand(body,ctx,budget)
        return value
    if isinstance(f,FBin):return FBin(f.op,expand(f.a,ctx,budget),expand(f.b,ctx,budget))
    if isinstance(f,FUn):return FUn(f.op,expand(f.a,ctx,budget))
    if isinstance(f,FSel):
        simplified=_simplify_sel(f,ctx)
        if not isinstance(simplified,FSel): return expand(simplified,ctx,budget)
        return FSel(f.cond,expand(f.a,ctx,budget),expand(f.b,ctx,budget))
    return f


def normalize(f,ctx):
    if not isinstance(f, (FUn, FBin, FFold)):
        return f
    try: expanded=expand(f,ctx,[4096])
    except ValueError: return f
    result=bounded(expanded,[4096])
    return FUn('integer.value',result.expression()) if result else f
