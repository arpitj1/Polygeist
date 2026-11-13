"""Descriptor emission from an imported parallel reference, never an application.

Projected permutations and broadcasts are composed before contiguous axes are
coalesced. The scalar program is read from the validated public reference body.
"""
import re


def reference(A, spec, args, shapes):
    import kernel_match as K
    import runtime_adapter as R
    suite = A.SPEC_DIR.name.removeprefix('lib_')
    text = (A.SPEC_DIR.parent / f'kernel_library_{suite}.mlir').read_text()
    fs = [f for f in R._parsed_reference_module(text).funcs if f.name == spec]
    if len(fs) != 1:
        raise A.AbiError('tensor adapter needs one imported reference')
    f = fs[0]
    gs = [o for o in f.ops if o.name == 'linalg.generic']
    if len(gs) != 1:
        raise A.AbiError('tensor adapter requires one parallel region')
    g = gs[0]
    rank = len(g.iters)
    if (not 1 <= rank <= 8 or g.iters != ['parallel'] * rank or
            len(g.outs) != 1 or g.parent is not None):
        raise A.AbiError('tensor adapter requires one fully parallel output')
    defs = {r: o for o in f.ops for r in o.results}
    argmap = {a.name: a for a in args}
    outer = {'arith.index_cast', 'arith.constant', 'polygeist.submap',
             'polygeist.submapInverse', 'linalg.generic', 'kernel.yield'}
    if any(o.parent is None and o.name not in outer for o in f.ops):
        raise A.AbiError('tensor reference has computation outside its region')
    dimensions = [None] * rank

    def projection(text, expected):
        m = re.fullmatch(r'affine_map<\((.*?)\)\s*->\s*\((.*?)\)>', text.strip())
        if not m or m[1].replace(' ', '').split(',') != [f'd{i}' for i in range(expected)]:
            raise A.AbiError('tensor view requires a projected permutation')
        fields = [v.strip() for v in m[2].split(',')]
        if any(not re.fullmatch(r'd\d+', v) for v in fields):
            raise A.AbiError('tensor view requires unshifted dimensions')
        axes = [int(v[1:]) for v in fields]
        if len(set(axes)) != len(axes) or any(i >= expected for i in axes):
            raise A.AbiError('tensor view repeats or invents dimensions')
        return axes

    def bound(value):
        o = defs.get(value)
        if o and o.name == 'arith.index_cast' and len(o.operands) == 1:
            a = argmap.get(o.operands[0].lstrip('%'))
            if a and a.kind == 'int' and a.ctype == 'int':
                return a.name
        if o and o.name == 'arith.constant':
            m = re.search(r'arith.constant (\d+) : index', o.text)
            if m:
                return int(m[1])
        raise A.AbiError('tensor extent must preserve a reference integer or constant')

    def view(value, axes):
        visited = set()
        while value in defs:
            if value in visited:
                raise A.AbiError('recursive tensor view')
            visited.add(value)
            o = defs[value]
            if o.name != 'polygeist.submap' or len(o.operands) != len(axes) + 1:
                raise A.AbiError('tensor adapter cannot emit this reference view')
            for axis, extent in zip(axes, o.operands[1:]):
                size = bound(extent)
                if dimensions[axis] is not None and dimensions[axis] != size:
                    raise A.AbiError('inconsistent tensor iteration extent')
                dimensions[axis] = size
            m = re.search(r'map\s*=\s*(affine_map<\([^)]*\)\s*->\s*\([^)]*\)>)', o.text)
            if not m:
                raise A.AbiError('tensor view has no affine map')
            axes = [axes[i] for i in projection(m[1], len(axes))]
            value = o.operands[0]
        name = value.lstrip('%')
        a = argmap.get(name)
        if not a or a.kind != 'array' or a.rank != len(axes) or a.ctype != 'float':
            raise A.AbiError('tensor view must bind an FP32 reference array')
        for axis, size in zip(axes, shapes[name]):
            if dimensions[axis] is not None and dimensions[axis] != size:
                raise A.AbiError('tensor shape does not cover the reference extent')
            dimensions[axis] = size
        return name, axes

    output, output_axes = view(g.outs[0], projection(g.maps[-1], rank))
    if sorted(output_axes) != list(range(rank)) or argmap[output].intent != 'out':
        raise A.AbiError('tensor output must overwrite the full reference argument')
    inputs = [view(v, projection(m, rank)) for v, m in zip(g.ins, g.maps[:-1])]
    names = [n for n, axes in inputs]
    if (not 1 <= len(names) <= 8 or len(set(names)) != len(names) or output in names or
            set(names + [output]) != {a.name for a in args if a.kind == 'array'}):
        raise A.AbiError('tensor adapter needs distinct bound input arrays and one output')
    if any(argmap[n].intent != 'in' for n in names):
        raise A.AbiError('tensor adapter cannot read an output argument')
    # Verify that kernel.yield commits exactly this fully overwritten view.
    yields = [o for o in f.ops if o.name == 'kernel.yield']
    if len(yields) != 1 or len(yields[0].operands) != 1:
        raise A.AbiError('tensor reference needs one returned output')
    result = yields[0].operands[0]
    inverse = defs.get(result)
    if result != g.results[0]:
        if (not inverse or inverse.name != 'polygeist.submapInverse' or
                inverse.operands[:2] != ['%' + output, g.results[0]]):
            raise A.AbiError('tensor reference output is not the emitted region')
        m = re.search(r'map\s*=\s*(affine_map<\([^)]*\)\s*->\s*\([^)]*\)>)', inverse.text)
        if (not m or projection(m[1], rank) != output_axes or
                [bound(v) for v in inverse.operands[2:]] != dimensions):
            raise A.AbiError('tensor inverse view does not cover the output')
    elif output_axes != list(range(rank)):
        raise A.AbiError('tensor output permutation lacks its inverse')
    for name, axes in inputs:
        if axes != [a for a in output_axes if a in axes]:
            raise A.AbiError('tensor adapter needs consistently ordered contiguous arrays')
    # Merge only adjacent axes with identical broadcast behavior in every input.
    groups = []
    for axis in output_axes:
        signature = tuple(axis in axes for name, axes in inputs)
        if groups and signature == groups[-1][1]:
            groups[-1][0].append(axis)
        else:
            groups.append(([axis], signature))
    if len(groups) > 3:
        raise A.AbiError('tensor graph needs more than three coalesced dimensions')
    groups = [([], tuple(False for _ in inputs))] * (3 - len(groups)) + groups
    extents = [[dimensions[a] for a in axes] for axes, signature in groups]
    masks = [[bool(signature[i]) for axes, signature in groups] for i in range(len(inputs))]
    gb = K.parse_generics(g.text, infer_outputs_from_yield=True)[0]
    graph = R.pointwise_graph(A, spec, args, reference=(f, g, gb, names, output), input_slots=8)
    # The currently linked ND runtime emits binary arithmetic with a full-sized
    # first operand. Reject unsupported vendor graphs before accepting this ABI.
    full = {i: all(mask[j] or not extents[j] for j in range(3)) for i, mask in enumerate(masks)}
    full.update({i: False for i in range(8, 16)})
    for i in range(graph['nodes']):
        instruction = (graph['words'][i // 2] >> (32 * (i % 2))) & 0xffffffff
        mode, a, b = instruction >> 24, (instruction >> 16) & 255, (instruction >> 8) & 255
        if (mode not in (1, 2, 3) or not (full.get(a) or full.get(b)) or
                (mode == 3 and not full.get(a))):
            raise A.AbiError('unsupported broadcast tensor scalar program')
        full[16 + i] = True
    return dict(graph=graph, extents=extents, input_masks=masks)


def contract(A, spec, target, ret, args):
    from typed_wrapper import geometry
    cfg = target['runtime_adapter']
    if (set(cfg) != {'schema', 'kind', 'arrays', 'bounds'} or cfg['schema'] != 1 or
            ret != 'void' or target.get('backends') != ['cuda'] or
            (target['library'], target['call']) != ('cuDNN', 'cudnnBackendExecute') or
            any(a.ctype not in ('int', 'float') or a.kind not in ('int', 'scalar', 'array') for a in args)):
        raise A.AbiError('unsupported tensor pointwise reference contract')
    A.resolve_proto('cuDNN', 'cudnnBackendExecute')
    params, bindings, checked = geometry(A, target, args, cfg, [a.name for a in args])
    if any(not isinstance(d, (int, str)) for shape in cfg['arrays'].values() for d in shape):
        raise A.AbiError('tensor graph dimensions must bind exact reference extents')
    descriptor = reference(A, spec, args, cfg['arrays'])
    return params, bindings, dict(checked, **descriptor,
        entry='polygeist_cudnn_tensor_pointwise_f32', vendor_dependencies=['cuDNN'])


def generate(ca, prototype):
    from typed_wrapper import generate as marshal, extent_expression
    cfg = ca.runtime_adapter
    graph = cfg['graph']
    def invoke(arguments):
        product = lambda ds: ' * '.join('(' + extent_expression(d) + ')' for d in ds) or '1'
        lines = ['  PolygeistPointwiseTensorLayout pv2_tensor = {0};\n',
                 f'  pv2_tensor.rank = 3; pv2_tensor.inputs = {len(graph["inputs"])}; pv2_tensor.nodes = {graph["nodes"]};\n']
        for d, dims in enumerate(cfg['extents']):
            lines.append(f'  pv2_tensor.dims[{d}] = {product(dims)};\n')
        lines += ['  int64_t pv2_stride = 1;\n',
                  '  for (int d = 2; d >= 0; --d) { pv2_tensor.output_strides[d] = pv2_stride; pv2_stride *= pv2_tensor.dims[d]; }\n']
        for i, mask in enumerate(cfg['input_masks']):
            for d, present in enumerate(mask):
                size = f'pv2_tensor.dims[{d}]' if present else '1'
                lines.append(f'  pv2_tensor.input_dims[{i}][{d}] = {size};\n')
            lines += ['  pv2_stride = 1;\n',
                      f'  for (int d = 2; d >= 0; --d) {{ pv2_tensor.input_strides[{i}][d] = pv2_stride; pv2_stride *= pv2_tensor.input_dims[{i}][d]; }}\n']
        for i, word in enumerate(graph['words']):
            lines.append(f'  pv2_tensor.words[{i}] = UINT64_C({word % (1 << 64)});\n')
        scalars = [str(v) + 'f' for v in graph['scalars']] + ['0.0f'] * (8 - len(graph['scalars']))
        for name, i in graph['scalar_arguments'].items():
            scalars[i] = name
        lines += ['  const float *pv2_tensor_inputs[] = {' + ', '.join(arguments[n] for n in graph['inputs']) + '};\n',
                  '  float pv2_tensor_scalars[8] = {' + ', '.join(scalars) + '};\n',
                  f'  polygeist_cudnn_tensor_pointwise_f32(&pv2_tensor, pv2_tensor_inputs, pv2_tensor_scalars, {arguments[graph["output"]]});\n']
        return ''.join(lines)
    return marshal(ca, prototype, emit_call=invoke)
