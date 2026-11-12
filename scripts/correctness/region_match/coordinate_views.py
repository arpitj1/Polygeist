"""Use a dominating contiguous view as the coordinates of a flat operand.

This is exact physical-address factoring, not an arithmetic rewrite or a
library recognizer. The original buffer ID/name remains the memory identity.
The emitter receives the existing SSA view and must keep its definition outside
the replacement window. Unknown layouts, nonzero offsets, tensor writes, and
accesses whose remainder cannot be bounded remain in their original coordinates.
"""
import re

from .affine import Aff, bounds_of, deps, ge, implies, lt
from .symexec import CondF, EBin, ERead, ESel, EUn, MemVal, Stmt


def _reads(value):
    if isinstance(value, ERead):
        yield value
    elif isinstance(value, EBin):
        yield from _reads(value.a)
        yield from _reads(value.b)
    elif isinstance(value, EUn):
        yield from _reads(value.a)
    elif isinstance(value, ESel):
        if isinstance(value.cond, CondF):
            yield from _reads(value.cond.a)
            yield from _reads(value.cond.b)
        yield from _reads(value.a)
        yield from _reads(value.b)


def _factor(address, width, domain):
    """Return q,r only after proving address=q*width+r and 0<=r<width."""
    quotient = Aff({v: c // width for v, c in address.terms if c % width == 0},
                   address.const // width)
    remainder = address - quotient.scale(width)
    if implies(domain, ge(remainder, 0)) and implies(domain, lt(remainder, width)):
        return (quotient, remainder)
    return None


def _apply_coordinates(program, buffer, coordinates, sizes, metadata):
    memo = {}
    def remap(expression):
        if id(expression) in memo:
            return memo[id(expression)]
        result = expression
        if isinstance(expression, ERead) and expression.buf is buffer:
            result = ERead(buffer, coordinates[('read', id(expression))], expression.time,
                           expression.guard, expression.vars)
        elif isinstance(expression, EBin):
            result = EBin(expression.op, remap(expression.a), remap(expression.b))
        elif isinstance(expression, EUn):
            result = EUn(expression.op, remap(expression.a))
        elif isinstance(expression, ESel):
            condition = expression.cond
            if isinstance(condition, CondF):
                condition = CondF(condition.kind, remap(condition.a), remap(condition.b))
            result = ESel(condition, remap(expression.a), remap(expression.b))
        memo[id(expression)] = result
        return result
    updated = {}
    for statement in program.stmts:
        indices = (list(coordinates[('write', statement.id)])
                   if statement.buf is buffer else statement.idx)
        updated[statement.id] = Stmt(
            statement.id, statement.vars, statement.domain, statement.guard,
            statement.time, statement.buf, indices, remap(statement.value),
            statement.par_dims, statement.top, statement.origin)
    program.stmts = [updated[statement.id] for statement in program.stmts]
    program.stored = {id(memo[key]): (updated[stored[0].id],) + stored[1:]
                      for key, stored in program.stored.items() if key in memo}
    buffer.rank = 2
    buffer.shape = list(sizes)
    program.region_views[buffer.name] = metadata


def normalize_coordinate_views(program):
    """Normalize fully proved existing views; preserve original sync identity.

    program.region_views[base] contains:
      ssa, type, base_type: existing view and its/original MLIR types;
      defined_at: top-level operation which must dominate the replacement;
      inner_extent: positive static row-major width;
      sizes: two affine extents, used for the logical array shape.
    """
    for top, op in enumerate(program.top_ops):
        if op.name != 'memref.reinterpret_cast' or len(op.results) != 1:
            continue
        ssa = op.results[0]
        value = program.env.get(ssa)
        if not isinstance(value, MemVal) or value.view is None:
            continue
        buffer, view = value.buf, value.view
        if (buffer.kind != 'arg' or buffer.rank != 1 or buffer.priv or
                buffer.elem not in ('f16', 'f32', 'f64', 'bf16') or
                buffer.name in program.region_views or
                program.tensor_writes.get(buffer.id) or
                'strided<' in buffer.arg_type or 'affine_map' in buffer.arg_type or
                len(view.index) != 1 or len(view.sizes) != 2 or
                view.sizes[1] is None or not view.sizes[1].is_const()):
            continue
        width = view.sizes[1].const
        if (width <= 1 or view.sizes[0] is None or
                view.index[0] != Aff.var('@v0', width) + Aff.var('@v1')):
            continue
        match = re.search(r'\s+to\s+(memref<.*>)\s*$', op.line)
        if match is None or 'strided<' in match[1] or 'affine_map' in match[1]:
            continue
        accesses = []
        for statement in program.stmts:
            if statement.buf is buffer:
                accesses.append((statement, None, statement.idx,
                                 statement.domain + statement.guard))
            for read in _reads(statement.value):
                if read.buf is buffer:
                    accesses.append((statement, read, read.idx,
                                     statement.domain + statement.guard + list(read.guard)))
        if not accesses:
            continue
        coordinates = {}
        safe = True
        for statement, read, indices, domain in accesses:
            if statement.top <= top or len(indices) != 1:
                safe = False
                break
            factored = _factor(indices[0], width, domain)
            if (factored is None or not implies(domain, ge(factored[0], 0)) or
                    not implies(domain, lt(factored[0], view.sizes[0]))):
                safe = False
                break
            key = ('read', id(read)) if read is not None else ('write', statement.id)
            previous = coordinates.get(key)
            if previous is not None and previous != factored:
                safe = False
                break
            coordinates[key] = factored
        if not safe:
            continue
        _apply_coordinates(program, buffer, coordinates, view.sizes, dict(
            ssa=ssa, type=match[1], base_type=buffer.arg_type,
            defined_at=top, inner_extent=width, sizes=list(view.sizes)))
    _normalize_complete_block_inputs(program)
    return program


def _complete_block_read(read, statement, width, factored):
    """Every reached outer coordinate reads all residues, not a partial tail."""
    quotient, remainder = factored
    if quotient.is_const() or len(remainder.terms) != 1 or remainder.terms[0][1] != 1:
        return False
    variable = remainder.terms[0][0]
    if (variable not in read.vars or variable not in statement.vars or
            variable in deps(quotient.vars()) or
            any(variable in deps(c.vars()) for c in statement.guard + list(read.guard))):
        return False
    try:
        lows, highs, rest = bounds_of(variable, statement.domain)
    except Exception:
        return False
    if (not lows or not highs or any(not v.is_const() for v in lows + highs) or
            any(variable in deps(c.vars()) for c in rest)):
        return False
    return (max(v.const for v in lows) == -remainder.const and
            min(v.const for v in highs) == width - 1 - remainder.const)


def _normalize_complete_block_inputs(program):
    """Factor readonly full-block accesses using their original valid-access contract.

    A complete read of residues [0,K) implies 0 <= outer and
    (outer+1)*K <= physical_capacity whenever the original program is defined.
    Hence the original accesses fit a contiguous view with
    floor(physical_capacity/K) rows even when an unused tail is present.
    No range is assumed for an indirect index or for an incomplete block.
    """
    for buffer in program.buffers:
        if (buffer.kind != 'arg' or buffer.rank != 1 or buffer.priv or
                buffer.elem not in ('f16', 'f32', 'f64', 'bf16') or
                buffer.name in program.region_views or
                program.tensor_writes.get(buffer.id) or
                'strided<' in buffer.arg_type or 'affine_map' in buffer.arg_type or
                any(s.buf is buffer for s in program.stmts)):
            continue
        accesses = [(s, r) for s in program.stmts for r in _reads(s.value)
                    if r.buf is buffer]
        if not accesses or any(len(r.idx) != 1 for _, r in accesses):
            continue
        candidates = set()
        for statement, read in accesses:
            candidates.update(abs(c) for _, c in read.idx[0].terms if abs(c) > 1)
            for variable in read.vars:
                try:
                    lows, highs, _ = bounds_of(variable, statement.domain)
                except Exception:
                    continue
                if lows and highs and all(v.is_const() for v in lows + highs):
                    extent = min(v.const for v in highs) - max(v.const for v in lows) + 1
                    if extent > 1:
                        candidates.add(extent)
        for width in sorted(candidates):
            coordinates = {}
            valid = True
            for statement, read in accesses:
                domain = statement.domain + statement.guard + list(read.guard)
                factored = _factor(read.idx[0], width, domain)
                if factored is None or not _complete_block_read(read, statement, width, factored):
                    valid = False
                    break
                coordinates[('read', id(read))] = factored
            if not valid:
                continue
            outer_parameter = 'pviewdim' + buffer.name + '_0'
            outer = Aff.var(outer_parameter)
            program.params[outer_parameter] = f'viewdim({buffer.name},0)'
            sizes = [outer, Aff.cst(width)]
            _apply_coordinates(program, buffer, coordinates, sizes, dict(
                generated=True, ssa=None, type=f'memref<?x{width}x{buffer.elem}>',
                base_type=buffer.arg_type, defined_at=-1, inner_extent=width,
                sizes=sizes, outer_dimension='floor(source_capacity / inner_extent)'))
            break
