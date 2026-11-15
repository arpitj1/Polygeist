"""Exact whole-region equivalence for imported structured computations.

The equation interpreter need not understand a loop recurrence to prove that
two instances of the same typed IR region are identical. Specialize declared
integer dimensions, canonicalize both sides, then compare the complete generic
MLIR serialization (including nested blocks, predicates, types and attributes).
No scalar implementation call or unverified reference is accepted here.
"""
from functools import lru_cache
import hashlib
import itertools
import json
from pathlib import Path
import re
import subprocess

from lib_v2 import mlir_lite as ML
from lib_v2 import recipe as R
from lib_v2.spec_provenance import file_hash, validated_stages
from .literal_policy import literal_ok


# Standard scalar math is already part of the IR's mathematical vocabulary.
# Application-specific helper declarations are deliberately excluded.
_MATH = set(('acos acosh asin asinh atan atanh atan2 cbrt ceil copysign cos cosh '
             'erf erfc exp exp2 expm1 fabs floor fma fmax fmin fmod frexp '
             'hypot ldexp lgamma log log10 log1p log2 nextafter pow remainder '
             'round sin sinh sqrt tan tanh tgamma trunc').split())
_MATH |= {n + 'f' for n in tuple(_MATH)}


def _module(text, function, args, substitutions=None, constants=()):
    header, body = function.text.split('\n', 1)
    # This initial exact-region implementation handles memory-output functions.
    if '->' in header or any(t.startswith('tensor<') for _, t in args):
        return None
    if re.search(r'\b(?:llvm.mlir.addressof|kernel.launch)\b', body):
        return None
    calls = set(re.findall(r'\bfunc.call @([\w.$-]+)', body))
    if calls - _MATH:
        return None
    # Carry immutable coefficient tables into the identity proof. Keep their
    # symbol identities as well as their full types, payloads and attributes;
    # never identify two distinct globals just because their contents agree.
    symbol = r'@(?:"(?:[^"\\]|\\.)*"|[\w.$-]+)'
    globals_used = sorted(set(re.findall(r'\bmemref.get_global\s+(' + symbol + ')', body)))
    # Table contents are values; an application's private table address need
    # not equal a runtime library's table address. Accept direct reads only.
    tables = {v for op in function.ops if op.name == 'memref.get_global'
              for v in op.results}
    for op in function.ops:
        uses = (set(ML.SSA_RE.findall(op.header)) - set(op.results)) & tables
        if uses and op.name not in ('memref.load', 'affine.load', 'memref.dim'):
            return None
    declarations = []
    for name in globals_used:
        declaration = re.search(r'^\s*(memref.global "private" constant\s+' +
                                re.escape(name) + r'\s*:[^\n]+)$', text, re.M)
        if not declaration or '= dense<' not in declaration[1]:
            return None
        declarations.append(declaration[1])
    for name in sorted(calls):
        found = re.search(r'^\s*(func.func private @' + re.escape(name) +
                          r'\([^\n]+)$', text, re.M)
        if not found or '{' in found[1]:
            return None
        declarations.append(found[1])
    if substitutions:
        body = ML.SSA_RE.sub(lambda m: substitutions.get(m[0], m[0]), body)
    body = ML.inline_aliases(body, ML.parse_aliases(text))
    signature = ', '.join(n + ': ' + t for n, t in args)
    return ('module {\n' + '\n'.join(declarations) + '\nfunc.func @_region(' +
            signature + ') {\n' + '\n'.join(constants) + '\n' + body + '\n}\n')


@lru_cache(maxsize=256)
def _canonical(module, compiler_hash):
    if module is None:
        return None
    proc = subprocess.run([R.POLYGEIST_OPT, '--canonicalize-polygeist', '--cse',
                           '--mlir-print-op-generic', '--mlir-print-local-scope'],
                          input=module, text=True, capture_output=True, timeout=30)
    if proc.returncode:
        return None
    # Canonicalization can hoist the same independent constants in different
    # orders on the two sides. Sort only contiguous constant runs; parsing and
    # reprinting then gives canonical SSA numbering without textual renaming
    # of values across block scopes. No effectful operation is reordered.
    lines = proc.stdout.splitlines(keepends=True)
    i = 0
    while i < len(lines):
        j = i
        while j < len(lines) and re.fullmatch(
                r'\s*%[\w.]+ = "arith.constant"\(\) .*\n?', lines[j]):
            j += 1
        if j > i:
            lines[i:j] = sorted(lines[i:j], key=lambda s:s.split(' = ',1)[1])
            i = j
        else:
            i += 1
    printed = subprocess.run([R.POLYGEIST_OPT, '--mlir-print-op-generic',
                              '--mlir-print-local-scope'], input=''.join(lines),
                             text=True, capture_output=True, timeout=30)
    return printed.stdout.strip() if printed.returncode == 0 else None


def _catalog(spec_dir, allowed):
    root = Path(spec_dir)
    imported = json.loads((root/'import_manifest.json').read_text())['specifications']
    for path in sorted((root/'manifest').glob('*.json')):
        for entry in json.loads(path.read_text()).get('specs', []):
            name = entry['name']
            if name not in allowed:
                continue
            permitted = validated_stages(imported.get(name, {}))
            for stage, suffix in [('debufferized', '_debuf'), ('linalg', '_linalg'),
                                  ('cgeist-affine', '')]:
                source = root/'raised'/(name+suffix+'.mlir')
                if stage not in permitted or not source.exists():
                    continue
                text = source.read_text()
                # Existing Linalg body matching covers straight-line computations.
                if not re.search(r'\b(?:scf.while|scf.for|affine.for|scf.index_switch)\b', text):
                    continue
                for function in ML.Module(text).funcs:
                    if function.name == name and function.ops:
                        yield entry, text, function


def match_regions(text, spec_dir, allowed_specs):
    """Propose integer-dimension bindings; exact region equality decides."""
    from .matcher import RegionMatch
    if not allowed_specs or not re.search(r'\b(?:scf.while|scf.for|affine.for|scf.index_switch)\b', text):
        return []
    compiler_hash = file_hash(Path(R.POLYGEIST_OPT))
    catalog = list(_catalog(spec_dir, allowed_specs))
    matches = []
    for app in ML.Module(text).funcs:
        if not app.ops or not app.args or any(not (t.startswith('memref<') or t in ('i32','i64','index','f32','f64')) for _, t in app.args):
            continue
        actual = _canonical(_module(text, app, app.args), compiler_hash)
        if actual is None:
            continue
        # These are proposals only. A dimension is accepted only if substituting
        # it makes every operation in the whole reference body identical.
        bounds = sorted({int(v) for v in re.findall(r'\bto (\d+)\b', app.text)} |
                        {int(v) for v in re.findall(r'arith.constant (\d+) : (?:index|i32|i64)\b', app.text)})
        bounds = [v for v in bounds if 0 < v <= 2147483647]
        if len(bounds) > 32:
            continue
        for entry, source, ref in catalog:
            if len(app.args) > 6 or len(entry.get('args',[])) != len(ref.args):
                continue
            extra = len(ref.args) - len(app.args)
            if extra not in (0, 1):
                continue
            # Bind all runtime operands by exact type, allowing one extra
            # integer extent to be specialized. Scalar values remain runtime
            # values; the complete region comparison proves their use.
            omitted = [None] if not extra else [i for i,(_,t) in enumerate(ref.args)
                                                if t in ('i32','i64','index')]
            found = False
            proposals = []
            for bound_index in omitted:
                runtime = [(i,n,t) for i,(n,t) in enumerate(ref.args) if i != bound_index]
                for permutation in itertools.permutations(range(len(app.args))):
                    if any(t != app.args[j][1] for (_,_,t),j in zip(runtime,permutation)):
                        continue
                    replacements = {n:app.args[j][0] for (_,n,_),j in zip(runtime,permutation)}
                    proposals.append((bound_index,replacements))
            for bound_index,replacements in proposals:
                for value in bounds if bound_index is not None else [None]:
                    constants = []
                    if bound_index is not None:
                        n,t = ref.args[bound_index]
                        if not literal_ok(entry, entry['args'][bound_index], value, 'int')[0]:
                            continue
                        replacements[n] = '%region_bound'
                        constants = [f'%region_bound = arith.constant {value} : {t}']
                    candidate = _canonical(_module(source, ref, app.args, replacements, constants), compiler_hash)
                    if candidate is None or candidate != actual:
                        continue
                    bindings, operands, types, lines = {}, [], [], []
                    for i,(n,t) in enumerate(ref.args):
                        if i != bound_index:
                            operand = replacements[n]
                            bindings[entry['args'][i]] = operand
                        else:
                            operand = '%region_bound'
                            lines.append(f'    {operand} = arith.constant {value} : {t}')
                            bindings[entry['args'][i]] = f'literal {value}'
                        operands.append(operand);types.append(t)
                    lines.append(f"    kernel.launch @{entry['name']}({', '.join(operands)}) : ({', '.join(types)}) -> ()")
                    first, last = min(o.start for o in app.ops), max(o.end for o in app.ops)
                    # Keep the function terminator and replace every preceding op.
                    end = next(o.start for o in app.ops if o.parent is None and o.name in ('return','func.return'))
                    matches.append(RegionMatch((first,end),entry['name'],'\n'.join(lines)+'\n',bindings,
                        'exact structured-region identity after integer binding; canonical IR sha256='+
                        hashlib.sha256(actual.encode()).hexdigest()))
                    found = True
                    break
                if found:
                    break
            if found:
                break
    return matches
