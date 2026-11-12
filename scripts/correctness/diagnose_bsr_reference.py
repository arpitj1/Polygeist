#!/usr/bin/env python3
"""Record raising/equation blockers for the imported public BSR reference.

This diagnostic does not add recognizers or modify either input. Its purpose is
separating compiler/executor gaps from library-reference coverage. GPU correctness
and performance require the normal fixture campaign after a complete proof.
"""
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(HERE / 'lib_v2')]
import recipe
from region_match.matcher import _load_system, enumerate_bindings, compare_systems
from region_match.mlir_parse import parse_module
from region_match.symexec import execute
from region_match.equations import build_system


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    os.environ['POLYGEIST_NUMERIC_MODEL'] = 'relaxed'
    args.output.mkdir(parents=True, exist_ok=True)
    reference = ROOT / 'generic_solver/lib_v3/ref_c/cusp_bsr_matvec.c'
    wrapper = args.output / 'reference.c'
    wrapper.write_text('#define NETLIB_SPEC cusp_bsr_matvec_beta0_f32\n'
                       '#include "' + str(reference) + '"\n')
    cases = [
        ('reference', wrapper, 'cusp_bsr_matvec_beta0_f32'),
        ('application', ROOT / 'issues/aten_c_kernels/aten_sparse_addmv_bsr_cpu.c',
         'aten_sparse_addmv_bsr_cpu'),
    ]
    result = {'numeric_model': 'relaxed', 'cases': {}}
    systems = {}
    for label, source, function in cases:
        steps = recipe.raise_c(recipe.PRODUCTION, str(source), function,
                               str(args.output / label), [], 120)
        case = {'source': str(source),
                'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'raise': {k: dataclasses.asdict(v) if dataclasses.is_dataclass(v) else v
                          for k, v in steps.items()}, 'equations': {}}
        for stage in ('affine', 'linalg', 'debuf'):
            path = steps['paths'][stage]
            try:
                if label == 'application' and stage == 'linalg':
                    functions, aliases = parse_module(Path(path).read_text())
                    program = execute(next(f for f in functions if f.name == function),
                                      aliases, coordinate_views=True)
                    system = build_system(program, {i for i,k in enumerate(program.top_kinds)
                                                    if k in ('work','writeback')})
                    case['region_views'] = program.region_views
                else:
                    _, system = _load_system(path, function)
                systems[(label,stage)] = system
                case['equations'][stage] = {
                    'ok': True, 'equations': [str(e) for e in system.equations],
                    'array_ranks': {n: b.rank for n, b in system.arrays.items()},
                }
            except Exception as error:
                case['equations'][stage] = {
                    'ok': False, 'error': type(error).__name__ + ': ' + str(error)}
        result['cases'][label] = case
    if ('application','linalg') in systems and ('reference','affine') in systems:
        app, ref = systems[('application','linalg')], systems[('reference','affine')]
        matches = [binding for binding in enumerate_bindings(app,ref,zero_cost_only=True)
                   if compare_systems(app,ref,binding).ok]
        result['equation_comparison'] = {
            'matched':bool(matches), 'bindings':[str(binding) for binding in matches],
            'scope':'Equation diagnostic only; emitted proof, ABI and GPU validation are separate'}
    (args.output / 'diagnostic.json').write_text(json.dumps(result, indent=2, default=str) + '\n')
    print(json.dumps({k: v['equations'] for k, v in result['cases'].items()}, indent=2))


if __name__ == '__main__':
    main()
