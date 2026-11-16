#!/usr/bin/env python3
"""lower_v2.py -- lower `kernel.launch @netlib_*` to lib_v2 runtime calls.

MLIR in, MLIR out.  Every `kernel.launch @netlib_X(...)` becomes
`func.call @polygeist_v2_<target.call>(...)` (see runtime/polygeist_v2_rt.h).
The lowering is driven entirely by the ABI table of abi_v2.py (manifests,
spec C prototypes, spec intents, vendor headers); it contains no per-spec or
per-routine code.

Stages
  1. prepare   drop every kernel.defn and synthesise one stub per launched
               spec/signature whose body yields the destination operands (the
               spec's written arrays, in argument order).  kernel.launch's
               BufferizableOpInterface reads destinations from such a yield, so
               One-Shot Bufferize can decide in-place vs copy soundly.  Mark
               to_tensor `restrict writable` (production pre-ABI convention).
  2. bufferize production view cleanup (--lower-polygeist-submap
               --canonicalize-polygeist, loop fallback for leftovers) followed
               by One-Shot Bufferize (allow-unknown-ops,
               bufferize-function-boundaries), as polygeist_build.sh does
               before ABI lowering.  Skipped when the input has no tensors.
  3. rewrite   each (now memref) launch -> func.call:
                 fixed enum args  -> i32 constants (vendor enum values)
                 fixed numbers    -> f64 / i32 constants
                 int / scalar     -> i32 / f64 (cast as needed)
                 arrays           -> !llvm.ptr to element 0 (aligned pointer +
                                     offset * elem size) and i64 byte extent of
                                     the view footprint
                 layout params    -> from memref strides, never from extents:
                                     rank 2: ld = stride[0] (asserting a unit
                                     inner stride); rank 1: inc = stride[0] x
                                     the spec's increment.
               Stub defns are removed.

Usage: lower_v2.py IN.mlir -o OUT.mlir [--no-bufferize] [--report R.json]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from c_types import MLIR_ELEMENT_TYPES
import abi_v2 as A  # noqa: E402
import recipe as R  # noqa: E402

LAUNCH_RE = re.compile(
    r"^(?P<ind>\s*)(?:(?P<res>%[\w$.\-]+)(?::(?P<nres>\d+))?\s*=\s*)?"
    r"kernel\.launch\s+@(?P<sym>[\w$.\-]+)\((?P<ops>[^)]*)\)\s*"
    r"(?P<attrs>\{.*?\})?\s*:\s*\((?P<tys>.*)\)\s*->\s*(?P<rty>.*?)\s*$")

ELEM_BYTES = {"f64": 8, "f32": 4, "i8": 1, "i16": 2, "i32": 4, "i64": 8}


class LowerError(Exception):
    pass


def split_top(s: str) -> list[str]:
    return A._split_top(s)


def parse_results(rty: str) -> list[str]:
    rty = rty.strip()
    if rty in ("()", ""):
        return []
    if rty.startswith("(") and rty.endswith(")"):
        return split_top(rty[1:-1])
    return [rty]


def memref_info(ty: str) -> tuple[int, str]:
    m = re.fullmatch(r"memref<((?:[?\d]+x)*)([a-z]\w*)(?:,.*)?>", ty.strip())
    if not m:
        raise LowerError(f"not a ranked memref type: {ty}")
    rank = len([d for d in m.group(1).split("x") if d])
    return rank, m.group(2)


def find_block(text: str, start: int) -> int:
    """index just past the `}` closing the first `{` at/after start."""
    i = text.index("{", start)
    depth = 0
    for k in range(i, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return k + 1
    raise LowerError("unbalanced braces")


def strip_defns(text: str) -> str:
    out, pos = [], 0
    for m in re.finditer(r"^\s*kernel\.defn\b", text, flags=re.M):
        if m.start() < pos:
            continue
        end = find_block(text, m.end())
        out.append(text[pos:m.start()])
        pos = end
        if pos < len(text) and text[pos] == "\n":
            pos += 1
    out.append(text[pos:])
    return "".join(out)


def insert_after_module_open(text: str, lines: str) -> str:
    m = re.search(r"^module\b[^\n]*\{\s*$", text, flags=re.M)
    if not m:
        raise LowerError("no top-level `module {` line")
    return text[:m.end()] + "\n" + lines + text[m.end():]


# ---------------------------------------------------------------------------
# Stage 1: stub defns
# ---------------------------------------------------------------------------

def mark_tensor_roots(text: str, specs: dict, names: dict) -> str:
    """Retain read-only roots when every use is a view or a known read.

    Library read/write roles come from the validated ABI. Unknown consumers
    retain the historical writable treatment; no computation is recognized.
    """
    import mlir_lite as ML
    module = ML.Module(text)
    uses = {}
    for op in module.ops:
        for index, value in enumerate(op.operands):
            definition = module.def_of(value, op)
            if definition:
                uses.setdefault((definition[0].idx, definition[1]), []).append((op, index))
    views = {'polygeist.submap', 'tensor.cast', 'tensor.collapse_shape',
             'tensor.expand_shape', 'tensor.extract_slice'}

    def readonly(root):
        pending, seen = [(root.idx, 0)], set()
        while pending:
            key = pending.pop()
            if key in seen:
                continue
            seen.add(key)
            for op, index in uses.get(key, []):
                if op.name in views and index == 0:
                    pending.extend((op.idx, k) for k in range(len(op.results)))
                elif op.name in ('tensor.dim', 'tensor.extract') and index == 0:
                    continue
                elif op.name == 'linalg.generic' and index < len(op.ins):
                    continue
                elif op.name == 'kernel.launch':
                    symbol = re.search(r'@([\w.$-]+)\(', op.header)
                    spec = specs.get(names.get(symbol[1], symbol[1])) if symbol else None
                    if (spec is None or index >= len(spec.args) or
                            spec.args[index].intent != 'in'):
                        return False
                else:
                    return False
        return True

    for op in reversed(module.ops):
        if op.name != 'bufferization.to_tensor' or not op.results:
            continue
        # Preserve explicit contracts; infer only the unannotated C boundary.
        if re.search(r'\b(restrict|writable)\b', op.header):
            continue
        replacement = re.sub(r'(bufferization\.to_tensor\s+%[\w.$-]+)\s*:',
                             r'\1 restrict'+('' if readonly(op) else ' writable')+' :', op.text)
        text = text[:op.start] + replacement + text[op.end:]
    return text


def reuse_elementwise_destinations(text: str, specs: dict, names: dict) -> str:
    """Choose an existing tensor for a validated overwrite-only destination.

    Only reuse a private library result with no other users. External memrefs
    may be read through pointers that tensor bufferization cannot track.
    One-shot bufferization still checks the selected destination normally.
    """
    import mlir_lite as ML
    module = ML.Module(text)
    views = {'tensor.cast', 'tensor.collapse_shape', 'tensor.expand_shape'}
    uses = {}
    for user in module.ops:
        for value in user.operands:
            if (definition := module.def_of(value, user)):
                uses.setdefault((definition[0].idx, definition[1]), set()).add(user.idx)

    def root(value, user, through_submap=False, sole_use=False):
        seen = set()
        while (definition := module.def_of(value, user)):
            op, result = definition
            if op.idx in seen:
                return None
            seen.add(op.idx)
            if sole_use and uses.get((op.idx, result)) != {user.idx}:
                return None
            if op.name not in views and not (through_submap and op.name == 'polygeist.submap'):
                return op
            if not op.operands:
                return None
            value, user = op.operands[0], op
        return None

    for op in reversed(module.ops):
        if op.name != 'kernel.launch':
            continue
        match = LAUNCH_RE.match(op.text)
        if not match or 'polygeist.elementwise_exact_alias' not in (match['attrs'] or ''):
            continue
        spec = specs.get(names.get(match['sym'], match['sym']))
        if not spec:
            continue
        outputs = [i for i, arg in enumerate(spec.args) if arg.kind == 'array' and arg.intent != 'in']
        if len(outputs) != 1 or spec.args[outputs[0]].intent != 'out':
            continue
        operands, types = split_top(match['ops']), split_top(match['tys'])
        dest = outputs[0]
        if len(operands) != len(spec.args) or len(types) != len(operands):
            continue
        allocation = root(operands[dest], op)
        if allocation is None or allocation.name != 'tensor.empty':
            continue
        for i, arg in enumerate(spec.args):
            if arg.kind != 'array' or arg.intent != 'in' or types[i] != types[dest]:
                continue
            source = root(operands[i], op, through_submap=True, sole_use=True)
            if source is None or source.name != 'kernel.launch':
                continue
            producer = LAUNCH_RE.match(source.text)
            producer_spec = specs.get(names.get(producer['sym'], producer['sym'])) if producer else None
            if not producer_spec:
                continue
            producer_outputs = [j for j, a in enumerate(producer_spec.args)
                                if a.kind == 'array' and a.intent != 'in']
            producer_operands = split_top(producer['ops'])
            if len(producer_outputs) != 1 or producer_outputs[0] >= len(producer_operands):
                continue
            # tensor.empty may be shared by several overwrite-only results;
            # its contents do not carry a computed value. The produced result
            # itself (including its views) must still have only this user.
            storage = root(producer_operands[producer_outputs[0]], source)
            if storage is None or storage.name != 'tensor.empty':
                continue
            operands[dest] = operands[i]
            replacement = op.text[:match.start('ops')] + ', '.join(operands) + op.text[match.end('ops'):]
            text = text[:op.start] + replacement + text[op.end:]
            break
    return text


def prepare(text: str, specs: dict, calls: dict | None = None) -> tuple[str, dict]:
    """-> (text, defn name -> spec)"""
    text = strip_defns(text)
    sigs: dict[tuple, str] = {}
    lines = text.split("\n")
    # Keep application symbols (including declarations and referenced external
    # callees) separate from the temporary kernel definitions used for ABI
    # preparation. A library spec can legitimately share an application's name.
    existing = "\n".join(ln for ln in lines if not LAUNCH_RE.match(ln))
    reserved = set()
    for quoted, plain in re.findall(r'@(?:"((?:[^"\\]|\\.)*)"|([\w$.\-]+))', existing):
        reserved.add(re.sub(r'\\([0-9A-Fa-f]{2})',
                            lambda m: chr(int(m[1], 16)), quoted) if quoted else plain)
    for i, ln in enumerate(lines):
        m = LAUNCH_RE.match(ln)
        if not m:
            continue
        sym = m.group("sym")
        if sym not in specs:
            raise LowerError(f"{sym}: no ABI (manifest/spec/header binding failed)")
        tys = tuple(split_top(m.group("tys")))
        rtys = tuple(parse_results(m.group("rty")))
        key = (sym, tys, rtys)
        if key not in sigs:
            n = sum(1 for k in sigs if k[0] == sym)
            name = sym if n == 0 else f"{sym}__v2sig{n}"
            while name in reserved:
                n += 1
                name = f"{sym}__v2sig{n}"
            sigs[key] = name
            reserved.add(name)
        name = sigs[key]
        # Never trust a capability supplied in input IR. It is derived from
        # the validated reference/backend contract, not the application name.
        capability = "polygeist.elementwise_exact_alias"
        attrs = [a for a in split_top((m.group("attrs") or "{}")[1:-1])
                 if a.strip().split("=", 1)[0].strip() != capability]
        call = (calls or {}).get(specs[sym].call)
        cfg = call.runtime_adapter if call else None
        if (cfg and cfg.get("alias_policy") == "exact_or_disjoint" and
                (cfg.get("kind") == "pointwise_reference" or
                 cfg.get("elementwise_exact_alias") is True)):
            attrs.append(capability)
        attr_text = " {" + ", ".join(attrs) + "}" if attrs else ""
        prefix = ln[:m.start("attrs")] if m.group("attrs") else ln[:m.start("tys")].rsplit(":", 1)[0]
        lines[i] = (prefix.rstrip() + attr_text + " : (" + m.group("tys") +
                    ") -> " + m.group("rty"))
        if name != sym:
            lines[i] = lines[i].replace(f"@{sym}(", f"@{name}(", 1)
    stubs, names = [], {}
    for (sym, tys, rtys), name in sigs.items():
        sa: A.SpecAbi = specs[sym]
        written = [k for k, a in enumerate(sa.args) if a.kind == "array" and a.intent in ("out", "inout")]
        args = ", ".join(f"%a{k}: {t}" for k, t in enumerate(tys))
        ret = f" -> ({', '.join(rtys)})" if len(rtys) > 1 else (f" -> {rtys[0]}" if rtys else "")
        if sa.ret != "void":
            # Read-only tensor inputs become memrefs while the ordinary scalar
            # SSA result survives bufferization. Mutating return-value APIs need
            # explicit tensor results as well and remain unsupported here.
            expected = {"float": ("f32",), "double": ("f64",)}.get(sa.ret)
            if written or rtys != expected:
                raise LowerError(f"{sym}: unsupported scalar-return destination contract")
            body = ""
            if rtys:
                body = (f"    %r = arith.constant 0.0 : {rtys[0]}\n"
                        f"    kernel.yield %r : {rtys[0]}")
            else:
                body = "    kernel.yield"
            stubs.append(f"  kernel.defn @{name}({args}){ret} {{\n{body}\n  }}\n")
            names[name] = sym
            continue
        if len(rtys) > len(written):
            raise LowerError(f"{sym}: {len(rtys)} results but only {len(written)} written arrays")
        dests = written[:len(rtys)] if rtys else []
        for k, rt in zip(dests, rtys):
            if tys[k] != rt:
                raise LowerError(f"{sym}: result type {rt} != destination operand type {tys[k]}")
        y = (f"    kernel.yield {', '.join(f'%a{k}' for k in dests)} : {', '.join(rtys)}"
             if rtys else "    kernel.yield")
        stubs.append(f"  kernel.defn @{name}({args}){ret} {{\n{y}\n  }}\n")
        names[name] = sym
    text = "\n".join(lines)
    if stubs:
        text = insert_after_module_open(text, "".join(stubs))
    text = mark_tensor_roots(text, specs, names)
    text = reuse_elementwise_destinations(text, specs, names)
    return text, names


# ---------------------------------------------------------------------------
# Stage 2: bufferize (production pre-ABI flow)
# ---------------------------------------------------------------------------

def run_opt(args: list[str], text: str, log: list) -> str:
    with tempfile.TemporaryDirectory() as d:
        inp, out = Path(d) / "in.mlir", Path(d) / "out.mlir"
        inp.write_text(text)
        cmd = [R.POLYGEIST_OPT, *args, str(inp), "-o", str(out)]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        log.append({"cmd": " ".join(args), "rc": p.returncode, "stderr": p.stderr[-3000:]})
        if p.returncode:
            raise LowerError(f"polygeist-opt {' '.join(args)} failed:\n{p.stderr[-3000:]}")
        return out.read_text()


def _views_left(text: str) -> bool:
    return bool(re.search(r"polygeist\.submap(Inverse)?\(.*tensor<", text))


def _cleanup_views(text: str, log: list, affine_copy_views=False) -> str:
    submaps = '--lower-polygeist-submap' + ('=affine-copy-writeback=true' if affine_copy_views else '')
    text = run_opt([submaps, "--canonicalize-polygeist"], text, log)
    if "polygeist.submap" in text:
        text = run_opt(["--convert-linalg-to-loops", "--canonicalize",
                        "--lower-polygeist-submap", "--canonicalize"], text, log)
    return text


def legalize(text: str, log: list) -> str:
    """Agent 1's test-only tensor-view legaliser (legalize_tensor_submaps.py)."""
    with tempfile.TemporaryDirectory() as d:
        inp, out = Path(d) / "in.mlir", Path(d) / "out.mlir"
        inp.write_text(text)
        p = subprocess.run([sys.executable, str(HERE / "legalize_tensor_submaps.py"), str(inp),
                            "-o", str(out)], capture_output=True, text=True, timeout=600)
        log.append({"cmd": "legalize_tensor_submaps.py", "rc": p.returncode,
                    "stderr": p.stderr[-3000:]})
        if p.returncode:
            raise LowerError(f"legalize_tensor_submaps.py failed:\n{p.stderr[-2000:]}")
        return out.read_text()


def bufferize(text: str, log: list, allow_legaliser: bool = True,
              affine_copy_views: bool = False) -> str:
    if "tensor<" not in text:
        return text
    src = text
    # production flow first (polygeist_build.sh ABI cleanup)
    text = _cleanup_views(src, log, affine_copy_views)
    if _views_left(text):
        if not allow_legaliser:
            raise LowerError("tensor-form polygeist.submap/submapInverse survive the production "
                             "view lowering (dynamic-size reduction write-back)")
        log.append({"note": "production view lowering left tensor submaps; "
                            "falling back to legalize_tensor_submaps.py"})
        text = _cleanup_views(legalize(src, log), log, affine_copy_views)
        if _views_left(text):
            raise LowerError("tensor-form submaps survive even the legaliser")
    # affine.for has no BufferizableOpInterface: tensor iter_args would stay
    # behind to_memref/to_tensor casts.  --lower-affine (which the later
    # mlir-opt pipeline runs anyway) turns them into scf.for first.
    text = run_opt(["--lower-affine", "--empty-tensor-to-alloc-tensor",
                    "--one-shot-bufferize=allow-unknown-ops bufferize-function-boundaries",
                    "--canonicalize", "--cse"], text, log)
    return text


# ---------------------------------------------------------------------------
# Stage 3: launch -> call
# ---------------------------------------------------------------------------

class Emitter:
    def __init__(self, ind: str, tag: str):
        self.ind, self.tag, self.n, self.lines = ind, tag, 0, []
        self.consts: dict[tuple, str] = {}

    def fresh(self, hint: str) -> str:
        self.n += 1
        return f"%pv2_{self.tag}_{hint}{self.n}"

    def op(self, rhs: str, hint: str = "v") -> str:
        v = self.fresh(hint)
        self.lines.append(f"{self.ind}{v} = {rhs}")
        return v

    def const(self, val, ty: str) -> str:
        key = (str(val), ty)
        if key not in self.consts:
            lit = repr(float(val)) if ty.startswith("f") else str(int(val))
            if ty.startswith("f") and "e" not in lit and "." not in lit:
                lit += ".0"
            self.consts[key] = self.op(f"arith.constant {lit} : {ty}", "c")
        return self.consts[key]

    def raw(self, line: str):
        self.lines.append(f"{self.ind}{line}")


def cast_int(e: Emitter, v: str, ty: str) -> str:
    if ty == "i32":
        return v
    if ty == "index":
        return e.op(f"arith.index_cast {v} : index to i32", "i")
    if ty == "i64":
        return e.op(f"arith.trunci {v} : i64 to i32", "i")
    if re.fullmatch(r"i(8|16)", ty):
        return e.op(f"arith.extsi {v} : {ty} to i32", "i")
    raise LowerError(f"cannot pass {ty} as int")


def cast_scalar(e: Emitter, v: str, ty: str, target: str) -> str:
    if ty == target:
        return v
    raise LowerError(f"cannot pass {ty} as {target} library scalar")


class ArrayVal:
    def __init__(self, e: Emitter, v: str, ty: str, spec_elem: str):
        rank, elem = memref_info(ty)
        if elem != spec_elem:
            raise LowerError(f"array {v}: element {elem} != library element {spec_elem}")
        self.rank, self.v, self.ty = rank, v, ty
        nb = ELEM_BYTES[elem]
        idx = ", ".join(["index"] * (1 + 2 * rank))
        md = e.fresh("md")
        e.raw(f"{md}:{2 + 2 * rank} = memref.extract_strided_metadata {v} : {ty} -> memref<{elem}>, {idx}")
        self.off = f"{md}#1"
        self.sizes = [f"{md}#{2 + d}" for d in range(rank)]
        self.strides = [f"{md}#{2 + rank + d}" for d in range(rank)]
        al = e.op(f"memref.extract_aligned_pointer_as_index {v} : {ty} -> index", "al")
        c0, c1, cb = e.const(0, "index"), e.const(1, "index"), e.const(nb, "index")
        offb = e.op(f"arith.muli {self.off}, {cb} : index", "ob")
        addr = e.op(f"arith.addi {al}, {offb} : index", "ad")
        a64 = e.op(f"arith.index_cast {addr} : index to i64", "a")
        self.ptr = e.op(f"llvm.inttoptr {a64} : i64 to !llvm.ptr", "p")
        # footprint: (sum (size_d - 1) * stride_d + 1) elements, 0 if empty
        acc, empty = c1, None
        for s, st in zip(self.sizes, self.strides):
            sm1 = e.op(f"arith.subi {s}, {c1} : index", "s")
            t = e.op(f"arith.muli {sm1}, {st} : index", "t")
            acc = e.op(f"arith.addi {acc}, {t} : index", "f")
            z = e.op(f"arith.cmpi eq, {s}, {c0} : index", "z")
            empty = z if empty is None else e.op(f"arith.ori {empty}, {z} : i1", "z")
        if empty is not None:
            acc = e.op(f"arith.select {empty}, {c0}, {acc} : index", "f")
        nbytes = e.op(f"arith.muli {acc}, {cb} : index", "nb")
        self.bytes = e.op(f"arith.index_cast {nbytes} : index to i64", "nb")
        self._layout = None

    def layout(self, e: Emitter, spec_inc: str | None, what: str) -> str:
        """Leading dimension (rank 2) or increment (rank 1) from strides."""
        c1 = e.const(1, "index")
        if self.rank == 2:
            # BLAS needs a unit inner stride; a single column needs none.
            okstride = e.op(f"arith.cmpi eq, {self.strides[1]}, {c1} : index", "ok")
            onecol = e.op(f"arith.cmpi ule, {self.sizes[1]}, {c1} : index", "ok")
            ok = e.op(f"arith.ori {okstride}, {onecol} : i1", "ok")
            e.raw(f'cf.assert {ok}, "lib_v2 {what}: column-major operand needs unit inner stride"')
            # with one row the stride is irrelevant; BLAS still wants ld >= rows
            onerow = e.op(f"arith.cmpi ule, {self.sizes[0]}, {c1} : index", "r")
            mx = e.op(f"arith.maxui {self.sizes[1]}, {c1} : index", "r")
            ld = e.op(f"arith.select {onerow}, {mx}, {self.strides[0]} : index", "ld")
            return e.op(f"arith.index_cast {ld} : index to i32", "ld")
        if self.rank == 1:
            one = e.op(f"arith.cmpi ule, {self.sizes[0]}, {c1} : index", "r")
            st = e.op(f"arith.select {one}, {c1}, {self.strides[0]} : index", "st")
            st32 = e.op(f"arith.index_cast {st} : index to i32", "st")
            if spec_inc is None:
                return st32
            return e.op(f"arith.muli {st32}, {spec_inc} : i32", "inc")
        if self.rank > 2:
            # A typed multidimensional C array is contiguous. Verify every
            # non-unit dimension instead of pretending it has a BLAS ld.
            expected = c1
            for size, stride in reversed(list(zip(self.sizes, self.strides))):
                equal = e.op(f"arith.cmpi eq, {stride}, {expected} : index", "dense")
                unit = e.op(f"arith.cmpi ule, {size}, {c1} : index", "unit")
                valid = e.op(f"arith.ori {equal}, {unit} : i1", "dense")
                e.raw(f'cf.assert {valid}, "typed array requires contiguous strides"')
                expected = e.op(f"arith.muli {expected}, {size} : index", "extent")
            return e.const(1, 'i32')
        raise LowerError(f"{what}: rank-{self.rank} operand has no BLAS layout")



def matrix_needs_layout_adapter(ty: str) -> bool:
    """Only potentially non-unit matrix views need a physical layout copy."""
    if memref_info(ty)[0] != 2:
        return False
    shape = ty[len('memref<'):].split(',', 1)[0].split('x')
    if shape[1] in ('0', '1'):
        return False
    layout = re.search(r'strided<\[([^]]+)\]', ty)
    if layout:
        return split_top(layout[1])[1].strip() != '1'
    return ',' in ty  # nontrivial/unknown layout; helper has a no-copy fast path


def pack_matrix(e: Emitter, av: ArrayVal, written: bool) -> str:
    original = [av.ptr, av.bytes]
    shape = [e.op(f"arith.index_cast {v} : index to i64", "dim")
             for v in av.sizes + av.strides]
    element = e.const(ELEM_BYTES[memref_info(av.ty)[1]], 'i32')
    arguments = original + shape + [element]
    types = '!llvm.ptr, i64, i64, i64, i64, i64, i32'
    av.ptr = e.op(f"func.call @polygeist_v2_matrix_pack({', '.join(arguments)}) : ({types}) -> !llvm.ptr", 'packed')
    c1 = e.const(1, 'index')
    nonunit = e.op(f"arith.cmpi ne, {av.strides[1]}, {c1} : index", 'nonunit')
    multiple = e.op(f"arith.cmpi ugt, {av.sizes[1]}, {c1} : index", 'multiple')
    copied = e.op(f"arith.andi {nonunit}, {multiple} : i1", 'copied')
    row = e.op(f"arith.select {copied}, {av.sizes[1]}, {av.strides[0]} : index", 'stride')
    count = e.op(f"arith.muli {shape[0]}, {shape[1]} : i64", 'count')
    size = e.const(ELEM_BYTES[memref_info(av.ty)[1]], 'i64')
    dense = e.op(f"arith.muli {count}, {size} : i64", 'dense')
    av.bytes = e.op(f"arith.select {copied}, {dense}, {av.bytes} : i64", 'bytes')
    av.strides = [row, c1]
    write = e.const(int(written), 'i32')
    return (f"func.call @polygeist_v2_matrix_release({', '.join([av.ptr] + arguments + [write])}) : "
            f"(!llvm.ptr, {types}, i32) -> ()")


def vector_needs_layout_adapter(ty: str) -> bool:
    if memref_info(ty)[0] != 1:
        return False
    layout = re.search(r'strided<\[([^]]+)\]', ty)
    return layout[1].strip() != '1' if layout else ',' in ty


def pack_vector(e: Emitter, av: ArrayVal, written: bool) -> str:
    """Use the existing device-aware layout copy for a contiguous vector API."""
    n = e.op(f"arith.index_cast {av.sizes[0]} : index to i64", 'dim')
    stride = e.op(f"arith.index_cast {av.strides[0]} : index to i64", 'stride')
    width = ELEM_BYTES[memref_info(av.ty)[1]]
    arguments = [av.ptr, av.bytes, e.const(1, 'i64'), n,
                 e.const(0, 'i64'), stride, e.const(width, 'i32')]
    types = '!llvm.ptr, i64, i64, i64, i64, i64, i32'
    av.ptr = e.op(f"func.call @polygeist_v2_matrix_pack({', '.join(arguments)}) : ({types}) -> !llvm.ptr", 'packed')
    av.bytes = e.op(f"arith.muli {n}, {e.const(width, 'i64')} : i64", 'dense')
    av.strides = [e.const(1, 'index')]
    write = e.const(int(written), 'i32')
    return (f"func.call @polygeist_v2_matrix_release({', '.join([av.ptr] + arguments + [write])}) : "
            f"(!llvm.ptr, {types}, i32) -> ()")


def lower_launch(m: re.Match, sa: A.SpecAbi, ca: A.CallAbi, tag: str) -> tuple[list[str], str]:
    ind = m.group("ind")
    ops = split_top(m.group("ops"))
    tys = split_top(m.group("tys"))
    rtys = parse_results(m.group("rty"))
    if len(ops) != len(sa.args) or len(tys) != len(sa.args):
        raise LowerError(f"{sa.spec}: launch has {len(ops)} operands, spec has {len(sa.args)}")
    e = Emitter(ind, tag)
    arrays: dict[int, ArrayVal] = {}
    releases = []
    elem_of = {p.name: p.elem for p in ca.params if p.tkind == "array"}

    def array(i: int, pname: str) -> ArrayVal:
        if i not in arrays:
            if not tys[i].startswith("memref<"):
                raise LowerError(f"{sa.spec}: array operand {sa.args[i].name} is {tys[i]} (not bufferized)")
            spec_elem = MLIR_ELEMENT_TYPES[elem_of[pname]]
            arrays[i] = ArrayVal(e, ops[i], tys[i], spec_elem)
            if ca.descriptor_adapter or ca.runtime_adapter:
                param = next(p for p in ca.params if p.name == pname)
                if arrays[i].rank != param.rank:
                    raise LowerError(f"{sa.spec}: descriptor array rank mismatch")
        return arrays[i]

    def int_value(b) -> str:
        if b[0] in ("arg", "alias"):
            return cast_int(e, ops[b[1]], tys[b[1]])
        if b[0] == "const_num":
            return e.const(int(b[1]), "i32")
        raise LowerError(f"{sa.spec}: cannot bind int from {b}")

    # Exact shape contracts are needed for higher-rank C arrays: a capacity
    # check alone cannot prove the inner strides required by a VLA prototype.
    if ca.runtime_adapter and ca.runtime_adapter.get('kind') in ('typed_c_wrapper', 'tensor_pointwise_reference'):
        for p in ca.params:
            if p.tkind != 'array' or p.rank <= 2:
                continue
            av = array(sa.binding[p.name][1], p.name)
            for actual, dimension in zip(av.sizes, ca.runtime_adapter['arrays'][p.name]):
                if type(dimension) is int:
                    expected = e.const(dimension, 'index')
                elif isinstance(dimension, dict) and 'product' in dimension:
                    expected = e.const(1, 'index')
                    for factor in dimension['product']:
                        if type(factor) is int:
                            term = e.const(factor, 'index')
                        else:
                            value = int_value(sa.binding[factor])
                            term = e.op(f'arith.index_cast {value} : i32 to index', 'extent')
                        expected = e.op(f'arith.muli {expected}, {term} : index', 'extent')
                elif isinstance(dimension, dict) and 'terms' in dimension:
                    from typed_wrapper import dimension_terms
                    terms, offset = dimension_terms(dimension)
                    expected = e.const(offset, 'index')
                    for name, coefficient in sorted(terms.items()):
                        value = int_value(sa.binding[name])
                        term = e.op(f'arith.index_cast {value} : i32 to index', 'extent')
                        if coefficient != 1:
                            scale = e.const(coefficient, 'index')
                            term = e.op(f'arith.muli {term}, {scale} : index', 'extent')
                        expected = e.op(f'arith.addi {expected}, {term} : index', 'extent')
                else:
                    name = dimension['dimension'] if isinstance(dimension, dict) else dimension
                    value = int_value(sa.binding[name])
                    expected = e.op(f'arith.index_cast {value} : i32 to index', 'extent')
                    if isinstance(dimension, dict):
                        offset = e.const(dimension['offset'], 'index')
                        expected = e.op(f'arith.addi {expected}, {offset} : index', 'extent')
                equal = e.op(f'arith.cmpi eq, {actual}, {expected} : index', 'shape')
                e.raw(f'cf.assert {equal}, "typed array extent differs from declared shape"')

    # Ordinary BLAS matrices and contiguous vector-only CUDA adapters need
    # packing. Pointwise graphs already accept strides directly. Preserve
    # original alias checks before any packing can hide an overlap.
    packed_indices = set()
    if not ca.descriptor_adapter and not ca.runtime_adapter:
        for p in ca.params:
            if p.tkind == 'int' and p.layout_of:
                i = sa.binding[p.layout_of][1]
                if matrix_needs_layout_adapter(tys[i]):
                    packed_indices.add(i)
    elif (ca.runtime_adapter and ca.backends == ['cuda'] and
          ca.runtime_adapter.get('kind') != 'pointwise_reference'):
        for p in ca.params:
            if p.tkind == 'array':
                i = sa.binding[p.name][1]
                if vector_needs_layout_adapter(tys[i]):
                    packed_indices.add(i)
    if packed_indices:
        # Resolve every original descriptor before allocating any copy, so
        # packing cannot hide original operand overlap from the ABI guards.
        for p in ca.params:
            if p.tkind == 'array':
                array(sa.binding[p.name][1], p.name)
        for i in sorted(packed_indices):
            av = arrays[i]
            c1 = e.const(1, 'index')
            axis = av.rank - 1
            nonunit = e.op(f"arith.cmpi ne, {av.strides[axis]}, {c1} : index", 'nonunit')
            multiple = e.op(f"arith.cmpi ugt, {av.sizes[axis]}, {c1} : index", 'multiple')
            needed = e.op(f"arith.andi {nonunit}, {multiple} : i1", 'pack_needed')
            flag = e.op(f"arith.extui {needed} : i1 to i32", 'pack_flag')
            for j, other in arrays.items():
                if i != j:
                    e.raw(f"func.call @polygeist_v2_matrix_require_disjoint({av.ptr}, {av.bytes}, "
                          f"{other.ptr}, {other.bytes}, {flag}) : (!llvm.ptr, i64, !llvm.ptr, i64, i32) -> ()")
        for i in sorted(packed_indices):
            pack = pack_vector if arrays[i].rank == 1 else pack_matrix
            releases.append(pack(e, arrays[i], sa.args[i].intent in ('out', 'inout')))
    args, atys = [], []
    for p in ca.params:
        b = sa.binding[p.name]
        if p.tkind == "enum":
            if b[0] != "const_enum":
                raise LowerError(f"{sa.spec}: enum {p.name} must be a fixed constant")
            args.append(e.const(b[1], "i32"))
            atys.append("i32")
        elif p.tkind == "scalar":
            scalar_ty = {"float": "f32", "double": "f64"}[p.elem]
            if b[0] in ("arg", "alias"):
                args.append(cast_scalar(e, ops[b[1]], tys[b[1]], scalar_ty))
            elif b[0] == "const_num":
                args.append(e.const(float(b[1]), scalar_ty))
            else:
                raise LowerError(f"{sa.spec}: scalar {p.name} unbound")
            atys.append(scalar_ty)
        elif p.tkind == "array":
            if b[0] not in ("arg", "alias"):
                raise LowerError(f"{sa.spec}: array {p.name} unbound")
            av = array(b[1], p.name)
            args += [av.ptr, av.bytes]
            atys += ["!llvm.ptr", "i64"]
        elif p.tkind == "int" and p.layout_of:
            ab = sa.binding[p.layout_of]
            av = array(ab[1], p.layout_of)
            spec_inc = None
            if av.rank == 1:
                # the spec's own increment (strided variants / fixed 1) scales
                # the memref stride; for rank 2 the operand is superseded
                spec_inc = int_value(b)
            args.append(av.layout(e, spec_inc, f"{sa.spec}.{p.name}"))
            atys.append("i32")
        elif p.tkind == "int":
            args.append(int_value(b))
            atys.append("i32")
        else:
            raise LowerError(f"{sa.spec}: unsupported ABI kind {p.tkind}")
    callee = f"polygeist_v2_{ca.call}"
    res = m.group("res")
    ret = ca.returns
    ret_ty = MLIR_ELEMENT_TYPES.get(ret)
    # bufferized launches have no results; the only SSA result a library call
    # can produce is the scalar of a value-returning spec
    if rtys and not (ret and rtys == [ret_ty]):
        raise LowerError(f"{sa.spec}: unlowerable launch results {rtys} (not bufferized?)")
    call = f"func.call @{callee}({', '.join(args)}) : ({', '.join(atys)})"
    if ret and rtys and res:
        e.raw(f"{res} = {call} -> {ret_ty}")
    elif ret:
        e.raw(f"{e.fresh('r')} = {call} -> {ret_ty}")
    else:
        e.raw(f"{call} -> ()")
    for release in reversed(releases):
        e.raw(release)
    decl = f"  func.func private @{callee}({', '.join(atys)})" + (f" -> {ret_ty}" if ret else "")
    return e.lines, decl


def rewrite(text: str, names: dict, specs: dict, calls: dict, report: list,
            backend: str = "cpu") -> str:
    out, decls = [], {}
    for n, ln in enumerate(text.split("\n")):
        m = LAUNCH_RE.match(ln)
        if not m:
            out.append(ln)
            continue
        name = m.group("sym")
        spec = names.get(name, name)
        if spec not in specs:
            raise LowerError(f"launch of {name}: not a lib_v2 spec with an ABI")
        sa = specs[spec]
        ca = calls[sa.call]
        lines, decl = lower_launch(m, sa, ca, str(n))
        if backend == "cuda":
            # The CUDA ABI accepts resident device pointers. This is not a
            # graph-capture promise: a wrapper may allocate or synchronize.
            decl += " attributes {polygeist.gpu_runtime}"
        prev = decls.setdefault(ca.call, decl)
        if prev != decl:
            raise LowerError(f"inconsistent declarations for {ca.call}")
        out.extend(lines)
        report.append({"spec": spec, "call": f"polygeist_v2_{ca.call}",
                       "cpu_entry": ca.fortran["entry"] if ca.fortran else None,
                       "gpu_entry": ca.header_symbol,
                       "operand_types": split_top(m.group("tys"))})
    text = strip_defns("\n".join(out))
    if 'func.call @polygeist_v2_matrix_pack(' in text:
        decls['matrix_disjoint'] = ('  func.func private @polygeist_v2_matrix_require_disjoint('
            '!llvm.ptr, i64, !llvm.ptr, i64, i32)')
        decls['matrix_pack'] = ('  func.func private @polygeist_v2_matrix_pack('
            '!llvm.ptr, i64, i64, i64, i64, i64, i32) -> !llvm.ptr')
        decls['matrix_release'] = ('  func.func private @polygeist_v2_matrix_release('
            '!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i32, i32)')
    if decls:
        text = insert_after_module_open(text, "\n".join(decls.values()) + "\n")
    if "kernel.launch" in text:
        raise LowerError("kernel.launch left after lowering")
    return text


COPY_RE = re.compile(r'^(?P<ind>\s*)memref\.copy (?P<src>%[\w$.\-]+), (?P<dst>%[\w$.\-]+) : (?P<sty>memref<.*>) to (?P<dty>memref<.*>)\s*$')

COPY_DECL = ('  func.func private @polygeist_v2_memref_copy(i64, i64, !llvm.ptr, '
             '!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) attributes '
             '{polygeist.gpu_runtime, polygeist.gpu_data_operands = array<i64: 5, 6>}\n')


def emit_descriptor_copy(e, element, sizes, source_strides, destination_strides,
                         source, destination):
    rank = len(sizes)
    pointers = []
    for values in (sizes, source_strides, destination_strides):
        ty = f'memref<{max(rank, 1)}xi64>'
        storage = e.op(f'memref.alloca() : {ty}', 'meta')
        for i, value in enumerate(values):
            at = e.const(i, 'index')
            val = e.op(f'arith.index_cast {value} : index to i64', 'dim')
            e.raw(f'memref.store {val}, {storage}[{at}] : {ty}')
        address = e.op(f'memref.extract_aligned_pointer_as_index {storage} : {ty} -> index', 'address')
        integer = e.op(f'arith.index_cast {address} : index to i64', 'addr')
        pointers.append(e.op(f'llvm.inttoptr {integer} : i64 to !llvm.ptr', 'ptr'))
    args = [e.const(rank, 'i64'), e.const(ELEM_BYTES[element], 'i64'),
            *pointers, source, destination]
    types = ['i64', 'i64'] + ['!llvm.ptr'] * 5
    e.raw(f"func.call @polygeist_v2_memref_copy({', '.join(args)}) : ({', '.join(types)}) -> ()")


def affine_memory_copies(text: str) -> str:
    """Lower pure affine memory movement to descriptors after bufferization.

    Computation bodies, reductions, symbolic/nonlinear maps and unproved loop
    extents stay unchanged. The copy runtime preserves holes and handles alias
    overlap; this only composes addresses, without selecting library kernels.
    """
    import mlir_lite as ML
    module = ML.Module(text)
    changed = False
    for op in reversed(module.ops):
        if (op.name != 'linalg.generic' or op.results or len(op.ins) != 1 or
                len(op.outs) != 1 or not op.iters or set(op.iters) != {'parallel'} or
                len(op.maps) != 2):
            continue
        body = [line for line in op.body if line != '}']
        if len(body) != 2:
            continue
        args = re.match(r'\^bb\w*\((%[\w.$-]+)\s*:', body[0])
        if not args or not re.fullmatch(r'linalg.yield '+re.escape(args[1])+r'\s*:\s*\w+', body[1]):
            continue
        tys = op.in_types + op.out_types
        if any(not ty.startswith('memref<') for ty in tys):
            continue
        rank = len(op.iters)
        maps = []
        for mapping in op.maps:
            nd, ns, _ = ML.parse_affine_map(mapping)
            if nd != rank or ns:
                break
            try:
                poly = ML.map_apply(mapping, ML.identity_dims(rank), [])
            except ML.AffineParseError:
                break
            if any(any(key and (len(key) != 1 or key[0][0] != 'd') for key in p) for p in poly):
                break
            maps.append(poly)
        if len(maps) != 2:
            continue
        bounds = {}
        for operand, mapping in enumerate(maps):
            for dim, polynomial in enumerate(mapping):
                iterator = ML.pure_dim(polynomial)
                if iterator is not None:
                    bounds.setdefault(iterator, (operand, dim))
        if set(bounds) != set(range(rank)):
            continue
        element = memref_info(tys[0])[1]
        if element not in ELEM_BYTES or memref_info(tys[1])[1] != element:
            continue
        ind = re.match(r'\s*', op.header)[0]
        e = Emitter(ind + '  ', 'affinecopy'+str(op.idx))
        arrays = [ArrayVal(e, value, ty, element)
                  for value, ty in zip(op.ins+op.outs, tys)]
        sizes = [arrays[bounds[i][0]].sizes[bounds[i][1]] for i in range(rank)]
        pointers, strides = [], []
        for array, mapping in zip(arrays, maps):
            def dot(coefficients):
                result = e.const(0, 'index')
                for coefficient, stride in zip(coefficients, array.strides):
                    if coefficient:
                        c = e.const(coefficient, 'index')
                        term = e.op(f'arith.muli {c}, {stride} : index', 'stride')
                        result = e.op(f'arith.addi {result}, {term} : index', 'sum')
                return result
            strides.append([dot([p.get((("d", i),), 0) for p in mapping])
                            for i in range(rank)])
            offset = dot([p.get((), 0) for p in mapping])
            scale = e.const(ELEM_BYTES[element], 'index')
            offset = e.op(f'arith.muli {offset}, {scale} : index', 'bytes')
            offset = e.op(f'arith.index_cast {offset} : index to i64', 'offset')
            pointers.append(e.op(f'llvm.getelementptr {array.ptr}[{offset}] : '
                                '(!llvm.ptr, i64) -> !llvm.ptr, i8', 'ptr'))
        emit_descriptor_copy(e, element, sizes, *strides, *pointers)
        e.raw('memref.alloca_scope.return')
        replacement = '\n'.join([ind+'memref.alloca_scope {', *e.lines, ind+'}'])+'\n'
        text = text[:op.start] + replacement + text[op.end:]
        changed = True
    if changed and 'func.func private @polygeist_v2_memref_copy(' not in text:
        text = insert_after_module_open(text, COPY_DECL)
    return text


def scalar_memory_copies(text: str) -> str:
    """Stage scalar host accesses through copies when the buffer may be on CUDA.

    This is address-space plumbing after matching, including scalar returns
    written to application buffers. It does not recognize arithmetic. Proven
    stack storage stays directly accessible, so repeated lowering is stable.
    """
    import mlir_lite as ML
    module = ML.Module(text)
    views = {'memref.cast', 'memref.subview', 'memref.reinterpret_cast',
             'memref.collapse_shape', 'memref.expand_shape', 'memref.view'}

    def stack(value, use):
        definition = module.def_of(value, use)
        if not definition:
            return False
        root = definition[0]
        if root.name == 'memref.alloca':
            return True
        return bool(root.name in views and root.operands and stack(root.operands[0], root))

    for op in reversed(module.ops):
        if op.name not in ('memref.load', 'memref.store'):
            continue
        load = op.name == 'memref.load'
        match = re.fullmatch(
            r'(?P<ind>\s*)(?:(?P<result>%[\w$.\-]+) = memref\.load |'
            r'memref\.store (?P<value>%[\w$.\-]+), )(?P<buffer>%[\w$.\-]+)'
            r'\[(?P<indices>[^]]*)\] : (?P<type>memref<.*>)\s*', op.text)
        if not match or stack(match['buffer'], op):
            continue
        rank, elem = memref_info(match['type'])
        if elem not in ELEM_BYTES:
            raise LowerError('unsupported CUDA scalar memory element: ' + elem)
        indices = split_top(match['indices'])
        if len(indices) != rank:
            raise LowerError('scalar access rank mismatch')
        e = Emitter(match['ind'] + '  ', 'scalar' + str(op.idx))
        ty = f'memref<{elem}>'
        slot = e.op(f'memref.alloca() : {ty}', 'slot')
        if rank:
            viewty = f'memref<{elem}, strided<[], offset: ?>>'
            ones = ', '.join(['1'] * rank)
            view = e.op(f'memref.subview {match["buffer"]}[{match["indices"]}] '
                        f'[{ones}] [{ones}] : {match["type"]} to {viewty}', 'view')
        else:
            view, viewty = match['buffer'], match['type']
        if load:
            e.raw(f'memref.copy {view}, {slot} : {viewty} to {ty}')
            value = e.op(f'memref.load {slot}[] : {ty}', 'value')
            e.raw(f'memref.alloca_scope.return {value} : {elem}')
            first = f'{match["result"]} = memref.alloca_scope -> ({elem}) {{'
        else:
            e.raw(f'memref.store {match["value"]}, {slot}[] : {ty}')
            e.raw(f'memref.copy {slot}, {view} : {ty} to {viewty}')
            e.raw('memref.alloca_scope.return')
            first = 'memref.alloca_scope {'
        replacement = '\n'.join([match['ind'] + first, *e.lines, match['ind'] + '}']) + '\n'
        text = text[:op.start] + replacement + text[op.end:]
    return text


def rewrite_memory_copies(text: str, backend: str) -> str:
    """CUDA operands may be device pointers; residual memory movement must
    preserve that address space. CPU/PVA emission is unchanged."""
    if backend != 'cuda':
        return text
    text = scalar_memory_copies(affine_memory_copies(text))
    lines, used = [], False
    for n, line in enumerate(text.split('\n')):
        m = COPY_RE.match(line)
        if not m:
            lines.append(line)
            continue
        if m['src'] == m['dst'] and m['sty'] == m['dty']:
            continue  # Exact SSA/type identity is a no-op, including empty views.
        rank, elem = memref_info(m['sty'])
        if memref_info(m['dty']) != (rank, elem):
            raise LowerError('memory copy rank/element mismatch')
        e = Emitter(m['ind'], 'copy' + str(n))
        src = ArrayVal(e, m['src'], m['sty'], elem)
        dst = ArrayVal(e, m['dst'], m['dty'], elem)
        for a, b in zip(src.sizes, dst.sizes):
            ok = e.op(f'arith.cmpi eq, {a}, {b} : index', 'shape')
            e.raw(f'cf.assert {ok}, "lib_v2 memory copy: shape mismatch"')
        prefix, e.lines = e.lines, []
        e.ind += '  '
        emit_descriptor_copy(e, elem, src.sizes, src.strides, dst.strides, src.ptr, dst.ptr)
        lines.extend(prefix)
        lines.append(m['ind'] + 'memref.alloca_scope {')
        lines.extend(e.lines)
        lines.append(m['ind'] + '  memref.alloca_scope.return')
        lines.append(m['ind'] + '}')
        used = True
    output = '\n'.join(lines)
    if used and 'func.func private @polygeist_v2_memref_copy(' not in output:
        output = insert_after_module_open(output, COPY_DECL)
    return output

def drop_overwritten_materializations(text: str, specs: dict, names: dict,
                                      calls: dict) -> tuple[str, int]:
    """Remove dead copies into private buffers before complete overwrites.

    The imported reference validates a write-only [0,n) output.
    Static allocation cardinality and full-buffer aliases prove that the call
    overwrites every byte; any intervening read or control flow blocks removal.
    This is memory-effect cleanup after selection, not library recognition.
    """
    import math
    import mlir_lite as ML
    import runtime_adapter
    module = ML.Module(text)
    uses = {}
    for op in module.ops:
        for index, value in enumerate(op.operands):
            definition = module.def_of(value, op)
            if definition:
                uses.setdefault((definition[0].idx, definition[1]), []).append((op, index))
    views = {'memref.cast', 'memref.collapse_shape', 'memref.expand_shape'}
    removed = set()
    for launch in module.ops:
        m = LAUNCH_RE.match(launch.header) if launch.name == 'kernel.launch' else None
        if not m:
            continue
        spec = specs.get(names.get(m['sym'], m['sym']))
        call = calls.get(spec.call) if spec else None
        cfg = call.runtime_adapter if call else None
        if not call:
            continue
        operands = split_top(m['ops'])
        if cfg and cfg.get('kind') == 'pointwise_reference':
            extent_name,output_name=cfg['extent_arg'],cfg['graph']['output']
        else:
            # Derive the same effect for any validated rank-one reference,
            # including ordinary vendor APIs without a graph adapter. The
            # imported body, never the API or application name, establishes
            # the complete write range and absence of reads of old output.
            effect=None
            for arg in spec.args:
                if arg.kind!='int':continue
                try:
                    f,g,gb,inputs,output=runtime_adapter.reference_region(A,spec.spec,spec.args,arg.name)
                except (A.AbiError,ValueError,KeyError,IndexError):
                    continue
                allowed={'polygeist.submap','polygeist.submapInverse',
                         'linalg.generic','linalg.yield','kernel.yield'}
                if any(o.name not in allowed and not o.name.startswith(('arith.','math.')) for o in f.ops):
                    continue
                writes=[o for o in f.ops if o.name=='polygeist.submapInverse']
                returns=[o for o in f.ops if o.name=='kernel.yield']
                if (len(writes)!=1 or len(returns)!=1 or len(g.results)!=1 or
                        len(writes[0].results)!=1 or
                        writes[0].operands[:2]!=['%'+output,g.results[0]] or
                        returns[0].operands!=writes[0].results or
                        'map=affine_map<(d0)->(d0)>' not in re.sub(r'\s+','',writes[0].text)):
                    continue
                # The inverse view must write the very extent used by the
                # region's output view; extra or different coordinates fail.
                outview=next((o for o in f.ops if g.outs[0] in o.results),None)
                if outview is None or writes[0].operands[2:]!=outview.operands[1:]:
                    continue
                effect=(arg.name,output);break
            if effect is None:continue
            extent_name,output_name=effect
        extent_index = next(i for i, a in enumerate(spec.args) if a.name == extent_name)
        output_index = next(i for i, a in enumerate(spec.args) if a.name == output_name)
        # The validated reference writes [0,n). Require that this is exactly
        # the entire allocation, not merely a prefix of a larger destination.
        extent = module.def_of(operands[extent_index], launch)
        cm = re.search(r'arith.constant (\d+) : i32', extent[0].header) if extent else None
        if not cm or int(cm[1]) <= 0:
            continue
        definition = module.def_of(operands[output_index], launch)
        while definition and definition[0].name in views:
            view = definition[0]
            definition = module.def_of(view.operands[0], view)
        if (not definition or definition[0].name != 'memref.alloc' or
                definition[0].block != launch.block):
            continue
        allocation = definition[0]
        shape = re.search(r': memref<((?:\d+x)+)f32>\s*$', allocation.header)
        if not shape or math.prod(map(int, shape[1].rstrip('x').split('x'))) != int(cm[1]):
            continue
        # A fresh private allocation cannot have hidden aliases. Follow only
        # full-buffer shape views; all other uses before the call are a barrier.
        pending, seen, copies, safe = [(allocation.idx, 0)], set(), [], True
        while pending and safe:
            key = pending.pop()
            if key in seen:
                continue
            seen.add(key)
            for op, index in uses.get(key, []):
                if op.start > launch.start:
                    continue
                if op.block != launch.block:
                    safe = False
                    break
                if op.idx == launch.idx:
                    if index != output_index:
                        safe = False
                        break
                elif op.name in views and index == 0:
                    pending.extend((op.idx, i) for i in range(len(op.results)))
                elif op.name == 'memref.dim' and index == 0:
                    continue
                elif (op.name == 'linalg.generic' and not op.results and
                      len(op.ins) == 1 and len(op.outs) == 1 and index == 1 and
                      op.iters and set(op.iters) == {'parallel'}):
                    body = [line for line in op.body if line != '}']
                    arg = re.match(r'\^bb\w*\((%[\w.$-]+)\s*:', body[0]) if len(body) == 2 else None
                    if not arg or not re.fullmatch(r'linalg.yield ' + re.escape(arg[1]) + r'\s*:\s*f32', body[1]):
                        safe = False
                        break
                    copies.append(op)
                elif op.name == 'memref.copy' and index == 1:
                    copies.append(op)
                else:
                    safe = False
                    break
        if safe:
            removed.update(op.idx for op in copies)
    for op in reversed(module.ops):
        if op.idx in removed:
            text = text[:op.start] + text[op.end:]
    return text, len(removed)


def lower(text: str, do_bufferize: bool = True, report: dict | None = None,
          allow_legaliser: bool = True, backend: str = "cpu",
          deallocate: bool = False) -> str:
    report = report if report is not None else {}
    report.setdefault("launches", [])
    report.setdefault("opt", [])
    used = sorted({m.group("sym") for m in map(LAUNCH_RE.match, text.split("\n")) if m})
    if not used:
        report["note"] = "no kernel.launch"
        text = strip_defns(text)
        if do_bufferize:
            text = bufferize(re.sub(r"bufferization\.to_tensor (%[^ ]*) :",
                                    r"bufferization.to_tensor \1 restrict writable :", text),
                             report["opt"], allow_legaliser)
        if deallocate:
            text = run_opt(["--lower-affine", "--buffer-deallocation"], text, report["opt"])
        return text
    calls, specs, errors = A.build_table()
    bad = [s for s in used if s not in specs]
    if bad:
        raise LowerError("no ABI for " + ", ".join(f"{s} ({errors.get(s, 'not a lib_v2 spec')})" for s in bad))
    text, names = prepare(text, specs, calls)
    if do_bufferize:
        text = bufferize(text, report["opt"], allow_legaliser,
                         affine_copy_views=(backend == 'cuda'))
    text, removed = drop_overwritten_materializations(text, specs, names, calls)
    report['eliminated_output_initializations'] = removed
    # Resolve lifetimes before rewriting launches to raw-pointer calls. After
    # that rewrite memref liveness cannot see the call's use of the buffer.
    if deallocate:
        text = run_opt(["--lower-affine", "--buffer-deallocation"], text, report["opt"])
    return rewrite(text, names, specs, calls, report["launches"], backend)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--memory-copies-only", action="store_true",
                    help="lower residual copies after higher-level GPU transforms")
    ap.add_argument("--no-bufferize", action="store_true",
                    help="input launches are already memref-form")
    ap.add_argument("--deallocate", action="store_true",
                    help="resolve buffer lifetimes before emitting raw-pointer calls")
    ap.add_argument("--no-legaliser", action="store_true",
                    help="fail instead of falling back to legalize_tensor_submaps.py "
                         "when production view lowering leaves tensor submaps")
    ap.add_argument("--report")
    ap.add_argument("--backend", choices=("cpu", "cuda", "pva"), default="cpu",
                    help="runtime ABI backend; CUDA declarations accept device pointers")
    a = ap.parse_args()
    report: dict = {}
    try:
        if a.memory_copies_only:
            out = rewrite_memory_copies(Path(a.input).read_text(), a.backend)
        else:
            out = lower(Path(a.input).read_text(), not a.no_bufferize, report,
                        allow_legaliser=not a.no_legaliser, backend=a.backend,
                        deallocate=a.deallocate)
    except LowerError as ex:
        print(f"lower_v2: error: {ex}", file=sys.stderr)
        if a.report:
            report["error"] = str(ex)
            Path(a.report).write_text(json.dumps(report, indent=1))
        return 1
    Path(a.output).write_text(out)
    if a.report:
        Path(a.report).write_text(json.dumps(report, indent=1))
    calls = [l["call"] for l in report.get("launches", [])]
    leg = any("legalize" in str(o.get("cmd", "")) for o in report.get("opt", []))
    print(f"lower_v2: {len(calls)} launch(es) -> {', '.join(calls) or '(none)'}"
          + ("  [legaliser fallback used]" if leg else ""), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
