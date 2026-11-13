"""Typed C-wrapper ABI binding; no computation recognition or formulas."""
import functools
import hashlib
import re
import subprocess

from c_types import canonical_type


def dimension_terms(dimension):
    """A checked affine size expression; never evaluate manifest text as code."""
    if type(dimension) is int:
        return {}, dimension
    if isinstance(dimension, str):
        return {dimension: 1}, 0
    if isinstance(dimension, dict) and set(dimension) == {'dimension', 'offset'}:
        terms, offset = {dimension['dimension']: 1}, dimension['offset']
    elif isinstance(dimension, dict) and set(dimension) == {'terms', 'offset'}:
        terms, offset = dimension['terms'], dimension['offset']
    else:
        raise ValueError('invalid affine extent')
    if (not isinstance(terms, dict) or not terms or len(terms) > 16 or
            type(offset) is not int or not -2147483647 <= offset <= 2147483647 or
            any(not isinstance(name, str) or type(coefficient) is not int or
                coefficient == 0 or not -2147483647 <= coefficient <= 2147483647
                for name, coefficient in terms.items())):
        raise ValueError('invalid affine extent terms')
    return terms, offset


@functools.lru_cache(maxsize=8)
def _preprocess_header(path, digest, compiler):
    result = subprocess.run([compiler, '-E', '-P', '-x', 'c', path],
                            capture_output=True, text=True)
    if result.returncode:
        raise ValueError('cannot preprocess runtime declarations: ' + result.stderr[-300:])
    return result.stdout


def _parameter(parameter, text):
    pointer = parameter.count('*')
    if pointer > 1 or '[' in parameter or '(' in parameter:
        raise ValueError('unsupported wrapper parameter: ' + parameter)
    if pointer:
        base, name = parameter.split('*')
        if name.strip() and not re.fullmatch(r'\s*\w+\s*', name):
            raise ValueError('unsupported pointer qualifier: ' + parameter)
        return canonical_type(base, text), True, bool(re.search(r'\bconst\b', base))
    try:
        return canonical_type(parameter, text), False, False
    except ValueError:
        named = re.fullmatch(r'(.+?)\s+\w+', parameter.strip())
        if named:
            return canonical_type(named[1], text), False, False
        raise


def contract(A, spec, target, ret, args):
    cfg = target['runtime_adapter']
    allowed = {'schema', 'kind', 'entry', 'arguments', 'arrays', 'bounds', 'alias_policy', 'linkage',
               'configuration'}
    if (set(cfg) - allowed or cfg.get('schema') != 1
            or cfg.get('kind') != 'typed_c_wrapper' or ret not in ('void', 'float', 'double', 'int')):
        raise A.AbiError('unsupported typed C-wrapper schema')
    if cfg.get('linkage', 'required') not in ('required', 'optional'):
        raise A.AbiError('unsupported typed C-wrapper linkage')
    backend = target.get('backends')
    if backend not in (['cuda'], ['pva']):
        raise A.AbiError('typed C-wrapper requires one supported backend')
    if cfg.get('alias_policy', 'disjoint') != 'disjoint':
        raise A.AbiError('typed C-wrapper requires disjoint outputs')
    entry = cfg.get('entry', '')
    family, call = target['library'], target['call']
    if not all(isinstance(v, str) and re.fullmatch(r'[A-Za-z_]\w*', v)
               for v in (entry, family, call)):
        raise A.AbiError('invalid typed C-wrapper identity')
    header = A.REPO / 'runtime/polygeist_cublas_rt.h'
    raw = header.read_text()
    marker = 'POLYGEIST_TYPED_VENDOR_ENTRY ' + family + ' ' + call + ' ' + entry
    if not re.search(r'/\*\s*' + re.escape(marker) + r'\s*\*/', raw):
        raise A.AbiError('runtime entry lacks the selected vendor binding')
    try:
        text = _preprocess_header(str(header), hashlib.sha256(raw.encode()).hexdigest(), A.CLANG)
        declaration = re.search(r'\b(void|float|double|int)\s+' + re.escape(entry) + r'\s*\(([^;{}]*)\)\s*;', text)
        if not declaration or declaration[1] != ret:
            raise ValueError('runtime/reference return types disagree')
        prototype = [_parameter(p.strip(), text) for p in A._split_top(declaration[2])]
    except ValueError as error:
        raise A.AbiError(str(error)) from error
    ordered = cfg.get('arguments')
    references = {arg.name: arg for arg in args}
    if (not isinstance(ordered, list) or len(ordered) != len(prototype)
            or any(not isinstance(v, (str, dict)) for v in ordered)
            or {v for v in ordered if isinstance(v, str)} != set(references)):
        raise A.AbiError('runtime contract must bind every reference and wrapper parameter')
    configured = {}
    try:
        if 'configuration' in cfg:
            from scalar_configuration import resolve
            configured = resolve(cfg['configuration'], A.manifest_entry(spec)[1].get('defines', {}))
    except (ValueError, SyntaxError) as error:
        raise A.AbiError(str(error)) from error
    resolved = []
    for name, (element, pointer, const) in zip(ordered, prototype):
        if isinstance(name, dict):
            from scalar_configuration import literal
            try:
                if pointer:
                    raise ValueError('configuration cannot synthesize a pointer')
                resolved.append({'literal': literal(name, configured, element)})
            except ValueError as error:
                raise A.AbiError(str(error)) from error
            continue
        resolved.append(name)
        arg = references[name]
        if (canonical_type(arg.ctype) != element or (arg.kind == 'array') != pointer
                or (pointer and const and arg.intent in ('out', 'inout'))):
            raise A.AbiError('runtime/reference parameter types or intents disagree: ' + name)
        if not pointer and not ((arg.kind == 'int' and element == 'int') or
                                (arg.kind == 'scalar' and element in ('float', 'double'))):
            raise A.AbiError('wrapper scalar requires exact int32, FP32, or FP64 storage')
    params, binding, validated = geometry(A, target, args, cfg, resolved)
    if configured:
        validated['resolved_configuration'] = configured
    return params, binding, validated


def geometry(A, target, args, cfg, ordered):
    """Validate bounded array geometry after an adapter validates its ABI."""
    arrays = {a.name: a for a in args if a.kind == 'array'}
    dimensions = {a.name for a in args if a.kind == 'int'}
    shapes = cfg.get('arrays', {})
    bounds = cfg.get('bounds', {})
    if not isinstance(shapes, dict) or set(shapes) != set(arrays):
        raise A.AbiError('every array needs an explicit extent contract')
    if not isinstance(bounds, dict) or set(bounds) != dimensions:
        raise A.AbiError('every integer argument needs explicit bounds')
    for name, bound in bounds.items():
        if (not isinstance(bound, dict) or not {'min', 'max'} <= set(bound)
                or set(bound) - {'min', 'max', 'multiple_of'}
                or any(type(v) is not int for v in bound.values())
                or not -2147483648 <= bound['min'] <= bound['max'] <= 2147483647
                or not 1 <= bound.get('multiple_of', 1) <= 2147483647):
            raise A.AbiError('invalid integer bounds: ' + name)
    for name, shape in shapes.items():
        arg = arrays[name]
        def valid_dimension(d):
            if isinstance(d, dict) and set(d) == {'product'}:
                factors = d['product']
                if not isinstance(factors, list) or not 2 <= len(factors) <= 8:
                    return False
                maximum = 1
                for factor in factors:
                    if type(factor) is int:
                        low = high = factor
                    elif isinstance(factor, str) and factor in bounds:
                        low, high = bounds[factor]['min'], bounds[factor]['max']
                    else:
                        return False
                    if low <= 0 or high > 2147483647 // maximum:
                        return False
                    maximum *= high
                return True
            try:
                terms, low = dimension_terms(d)
            except (ValueError, TypeError):
                return False
            high = low
            for name, coefficient in sorted(terms.items()):
                if name not in dimensions:
                    return False
                limits = [coefficient * bounds[name][k] for k in ('min', 'max')]
                low += min(limits)
                high += max(limits)
                if low < -(1 << 63) or high >= (1 << 63):
                    return False
            return 0 < low <= high <= 2147483647
        if (not isinstance(shape, list) or not 1 <= arg.rank <= 16
                or len(shape) != arg.rank or any(not valid_dimension(d) for d in shape)):
            raise A.AbiError('array extents require positive fixed sizes or bounded dimensions')
        if arg.vla_ld is not None and shape[-1] != arg.vla_ld:
            raise A.AbiError('array extent disagrees with the reference leading dimension')
    params, binding = [], {}
    for i, arg in enumerate(args):
        if arg.kind == 'array':
            param = A.AbiParam(arg.name, 'array', elem=canonical_type(arg.ctype), rank=arg.rank,
                               layout=arg.name + '_stride', written=arg.intent in ('out', 'inout'),
                               const=arg.intent == 'in')
        elif arg.kind == 'int':
            param = A.AbiParam(arg.name, 'int')
        else:
            param = A.AbiParam(arg.name, 'scalar', elem=arg.ctype)
        params.append(param)
        binding[arg.name] = ('arg', i)
        if arg.kind == 'array':
            params.append(A.AbiParam(param.layout, 'int', layout_of=arg.name))
            binding[param.layout] = ('const_num', 1)
    return params, binding, dict(cfg, ordered_arguments=ordered,
                                 memory_domain='cuda_device' if target['backends'] == ['cuda'] else 'host')


def extent_expression(dimension):
    if isinstance(dimension, dict) and set(dimension) == {'product'}:
        return '(' + ' * '.join(extent_expression(d) for d in dimension['product']) + ')'
    if isinstance(dimension, dict) and 'terms' in dimension:
        terms, offset = dimension_terms(dimension)
        return '(' + ' + '.join([f'((int64_t){name} * ({coefficient}))'
                                  for name, coefficient in sorted(terms.items())]
                                 + [str(offset)]) + ')'
    if isinstance(dimension, dict):
        return f'((int64_t){dimension["dimension"]} + ({dimension["offset"]}))'
    return f'(int64_t){dimension}'


def generate(ca, prototype, emit_call=None):
    """Mechanically marshal an already proved reference to its typed C entry."""
    cfg = ca.runtime_adapter
    arrays = [p for p in ca.params if p.tkind == 'array']
    cuda = cfg['memory_domain'] == 'cuda_device'
    declaration = ''
    if cfg.get('linkage', 'required') == 'optional':
        declaration = f'extern __typeof__({cfg["entry"]}) {cfg["entry"]} __attribute__((weak));\n'
    body = [declaration + prototype(ca) + ' {\n']
    if declaration:
        body.append(f'  if (!{cfg["entry"]})\n'
                    f'    polygeist_v2_fatal("{ca.call}", "optional vendor entry is not linked");\n')
    for name, bound in cfg['bounds'].items():
        body.append(f'  if ({name} < {bound["min"]} || {name} > {bound["max"]})\n'
                    f'    polygeist_v2_fatal("{ca.call}", "integer outside wrapper bounds");\n')
        if 'multiple_of' in bound:
            body.append(f'  if ({name} % {bound["multiple_of"]} != 0)\n'
                        f'    polygeist_v2_fatal("{ca.call}", "integer violates wrapper divisibility");\n')
    for p in arrays:
        dims = cfg['arrays'][p.name]
        body.append(f'  int64_t pv2_bytes_{p.name} = sizeof({p.elem});\n')
        for dim in dims:
            expr = extent_expression(dim)
            body.append(f'  if ({expr} <= 0 || pv2_bytes_{p.name} > INT64_MAX / {expr})\n'
                        f'    polygeist_v2_fatal("{ca.call}", "array extent overflow");\n'
                        f'  pv2_bytes_{p.name} *= {expr};\n')
        # Higher-rank typed arrays use the contiguous-layout flag proved by
        # ABI lowering. BLAS-compatible rank-two arrays retain their ld.
        stride = extent_expression(dims[-1]) if p.rank == 2 else '1'
        body.append(f'  if ({p.layout} != {stride} || !{p.name} || {p.name}_bytes < pv2_bytes_{p.name})\n'
                    f'    polygeist_v2_fatal("{ca.call}", "array layout or capacity mismatch");\n')
    for i, p in enumerate(arrays):
        for q in arrays[i + 1:]:
            if p.written or q.written:
                body.append(f'  if (polygeist_v2_overlaps({p.name}, {p.name}_bytes, {q.name}, {q.name}_bytes))\n'
                            f'    polygeist_v2_fatal("{ca.call}", "unsupported output overlap");\n')
    if cuda:
        body += ['  polygeist_v2_map_t pv2_map;\n',
                 f'  polygeist_v2_map_init(&pv2_map, "{ca.call}", "{cfg["entry"]}");\n']
        for p in arrays:
            body.append(f'  int pv2_k_{p.name} = polygeist_v2_map_add(&pv2_map, {p.name}, {p.name}_bytes, {int(p.written)});\n')
        body += ['  polygeist_v2_map_to_device(&pv2_map);\n',
                 '  polygeist_require_gpu_libraries_begin();\n',
                 '  polygeist_cublas_pipeline_begin();\n']
    else:
        body.append('  double pv2_t0 = polygeist_v2_now_ms();\n')
    by_name = {p.name: p for p in arrays}
    arguments = []
    for name in cfg['ordered_arguments']:
        if isinstance(name, dict):
            arguments.append(name['literal'])
            continue
        p = by_name.get(name)
        if cuda and p:
            element = ('const ' if not p.written else '') + p.elem
            arguments.append(f'({element} *)polygeist_v2_map_dev(&pv2_map, pv2_k_{name})')
        else:
            arguments.append(name)
    result = f'{ca.returns} pv2_result = ' if ca.returns else ''
    if emit_call is None:
        body.append(f'  {result}{cfg["entry"]}({", ".join(arguments)});\n')
    else:
        body.append(emit_call(dict(zip(cfg['ordered_arguments'], arguments))))
    if cuda:
        body += ['  polygeist_cublas_pipeline_end();\n',
                 '  polygeist_require_gpu_libraries_end();\n',
                 '  polygeist_v2_map_to_host(&pv2_map);\n']
    else:
        body.append(f'  polygeist_v2_done("{ca.call}", "{cfg["entry"]}", pv2_t0);\n')
    if ca.returns:
        body.append('  return pv2_result;\n')
    return ''.join(body) + '}\n\n'
