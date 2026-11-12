#!/usr/bin/env python3
"""Retest frozen COO kernels with the historical native API/preparation scope.

Only the native reference object and a copied harness are rebuilt. The copied
harness materializes allocating native outputs once for strict validation,
outside all timing intervals. New measurements use five alternating pairs.
"""
import argparse
import csv
import json
from pathlib import Path
import shlex
import shutil
import subprocess

import aten_fixture_batch as fixtures
from aten_frozen_runtime_campaign import (
    check_harness_count, digest, prepare_one, remote_libraries, save, transport,
)

NAMES = ['aten_sparse_addmm_cpu', 'aten_hspmm_cpu', 'aten_sspaddmm_cpu']


def prepare(source, output, name):
    original = json.loads((source / name / 'build.json').read_text())
    assert original['configuration']['timing_samples'] == 5
    prepared = prepare_one(source / name, output, 5)
    dest = output / name
    native = (source / name / 'native.cpp').read_text()
    marker = 'extern "C" void native_fixture('
    native = native.replace(marker,
        'static at::Tensor native_saved_result, native_output;\n' + marker)
    start = native.index('auto matrix=at::sparse_coo_tensor(')
    inner = 'K' if name == 'aten_sspaddmm_cpu' else '64'
    setup = f'''static auto matrix=at::sparse_coo_tensor(
      at::stack({{t[0],t[1]}}).to(at::kLong),t[2],{{R,{inner}}},
      t[2].options().layout(at::kSparse)).coalesce();
    static auto dense=t[3].view({{{inner},C}});
    if(!native_output.defined()) native_output=t[4].view({{R,C}});
'''
    if name == 'aten_sparse_addmm_cpu':
        operation = '''    static auto initial=at::zeros({R,C},t[2].options());
    at::addmm_out(native_output,initial,matrix,dense,0.0,1.0);
'''
    else:
        operation = '    native_saved_result=at::mm(matrix,dense);\n'
    native = native[:start] + setup + operation + '''}
extern "C" void native_fixture_materialize(){
  if(native_saved_result.defined()) native_output.copy_(native_saved_result);
}
'''
    (dest / 'native.cpp').write_text(native)
    harness = (dest / 'harness.c').read_text()
    call = '    invoke_native(device); CU(cudaDeviceSynchronize());'
    assert harness.count(call) == 1
    original_harness = harness
    harness = harness.replace('void native_fixture(float**,const float*,const size_t*,int);',
        'void native_fixture(float**,const float*,const size_t*,int);\n'
        'void native_fixture_materialize(void);')
    harness = harness.replace(call, call + '\n'
        '    native_fixture_materialize(); CU(cudaDeviceSynchronize());')
    (dest / 'harness.c').write_text(harness)
    check_harness_count(dest, 5)
    # The entire timing loop and all correctness comparisons remain unchanged.
    timing_start = '    if(argc>1&&!strcmp(argv[1],"timing")) {'
    correctness_start = '    for(int resident=0;resident<2;resident++) {'
    assert harness[harness.index(timing_start):] == original_harness[original_harness.index(timing_start):]
    assert correctness_start in harness
    native_command = list(original['commands'][1])
    native_command[native_command.index('-c') + 1] = str(dest / 'native.cpp')
    native_command[native_command.index('-o') + 1] = str(dest / 'native.o')
    includes = [x for x in original['commands'][2] if x.startswith('-I')]
    harness_command = ['aarch64-linux-gnu-gcc', *includes, '-O3', '-fno-inline',
        '-fno-inline-functions', '-fsemantic-interposition', '-fno-ipa-cp',
        '-fno-ipa-cp-clone', '-fno-ipa-sra', '-c', str(dest / 'harness.c'),
        '-o', str(dest / 'timing-harness.o')]
    link = []
    for item in original['commands'][-1]:
        p = Path(item)
        if item.endswith('.o'):
            if p.name in ('harness.o', 'timing-harness.o'):
                item = str(dest / 'timing-harness.o')
            elif p.name == 'native.o':
                item = str(dest / 'native.o')
            else:
                item = str(dest / ('work' if p.parent.name == 'work' else 'objects') / p.name)
        link.append(item)
    link[link.index('-o') + 1] = str(dest / 'raised-program')
    commands = [native_command, harness_command, link]
    with (dest / 'attribution-build.log').open('w') as log:
        for command in commands:
            log.write(shlex.join(command) + '\n'); log.flush()
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
    frozen = {path: sha for path, sha in prepared['frozen_object_sha256'].items()
              if Path(path).name not in ('native.o', 'harness.o', 'timing-harness.o')}
    assert all(digest(dest / path) == sha for path, sha in frozen.items())
    built = dict(original, binary_sha256=digest(dest / 'raised-program'),
        commands=original['commands'][:-1] + commands,
        native_attribution=dict(source=str(source / name),
            original_binary_sha256=original['binary_sha256'],
            frozen_compute_object_sha256=frozen,
            native_source_sha256=digest(dest / 'native.cpp'),
            harness_sha256=digest(dest / 'harness.c'),
            scope='Only native baseline and validation-only output materialization changed. '
                'Prebuilt/coalesced COO matrix; original raised inputs and compute objects unchanged; '
                'five alternating pairs with the original input restoration protocol.'))
    save(dest / 'build.json', built)
    shutil.copy2(dest / 'raised-program', dest / 'timing-program')
    prepared.update(timing_binary_sha256=built['binary_sha256'],
        timing_harness_sha256=digest(dest / 'harness.c'), commands=commands,
        adaptation=built['native_attribution']['scope'], native_attribution=True)
    save(dest / 'prepared.json', prepared)
    print('PREPARED', name, built['binary_sha256'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--paper', type=Path, required=True)
    parser.add_argument('--transport', type=Path, required=True)
    parser.add_argument('--remote-library-path', required=True)
    args = parser.parse_args()
    args.source = args.source.resolve(); args.output = args.output.resolve()
    for name in NAMES:
        prepare(args.source, args.output, name)
    board = transport(args.transport)
    first = args.output / NAMES[0] / 'raised-program'
    remote = '/tmp/ll-coo-attribution-' + digest(first)[:12]
    assert board.upload(first, remote) == 0
    assert board.run('chmod +x ' + shlex.quote(remote), timeout=15)[0] == 0
    environment = 'LD_LIBRARY_PATH=' + shlex.quote(args.remote_library_path) + ' '
    before = remote_libraries(board, remote, environment, args.output, 'before')
    from types import SimpleNamespace
    fixtures.run_gpu(NAMES, SimpleNamespace(output=args.output,
        transport=str(args.transport), remote_library_path=args.remote_library_path,
        cudnn_plan_cache=True, run_timeout=180, correctness_kinds=['paper'],
        skip_timing=False, time_validated_normal_inputs=False))
    after = remote_libraries(board, remote, environment, args.output, 'after')
    save(args.output / 'dynamic-library-stability.json', dict(unchanged=before == after))
    assert before == after
    paper = {r['kernel']: r for r in csv.DictReader(args.paper.open())}
    rows = []
    for name in NAMES:
        dest = args.output / name
        result = json.loads((dest / 'gpu-result.json').read_text())
        old = json.loads((args.source / name / 'gpu-result.json').read_text())
        row = dict(kernel=name, isolated_followup=True, fully_validated=result['fully_validated'],
            paper=paper[name], primary_result=str(args.source / name / 'gpu-result.json'),
            baseline_scope=('Actual addmm_out(beta=0, alpha=1) with prebuilt/coalesced COO matrix '
                'and preallocated output.' if name == 'aten_sparse_addmm_cpu' else
                'Actual mm on a prebuilt/coalesced COO matrix; result allocation included; '
                'output copy occurs only for strict validation, outside timing.'),
            measured_samples=5, paper_samples=20 if '20' in paper[name]['timing_protocol'] else 5,
            input_recipe_note=('Recorded raised fixture has all-zero row/column indices, whereas '
                'historical native recipe used distributed random entries. No paper-comparable '
                'speedup is asserted.' if name == 'aten_sspaddmm_cpu' else
                'Both current paths use the exact recorded raised inputs. Historical hspmm native '
                'used different random values/column ordering.' if name == 'aten_hspmm_cpu' else
                'Recorded native Section 4.2 and raised fixture inputs match.'),
            paper_speedup_comparable=name == 'aten_sparse_addmm_cpu')
        if result['fully_validated'] and result.get('timing', {}).get('passed'):
            assert len(result['timing']['samples']) == 10
            current = result['timing']['minimum_first_five_ms']
            previous = old['timing']['minimum_first_five_ms']
            row.update(raised_ms=current['raised'], native_ms=current['native_aten_cuda'],
                current_native_over_raised=current['native_aten_cuda']/current['raised'],
                previous_native_ms=previous['native_aten_cuda'],
                previous_raised_ms=previous['raised'],
                raised_over_paper=current['raised']*1000/float(paper[name]['raised_us']))
        else:
            row['status'] = 'strict_failed_no_timing'
        save(dest / 'paper-comparison.json', row); rows.append(row)
        print('ATTRIBUTION RESULT', json.dumps(row), flush=True)
    save(args.output / 'paper-comparisons.json', rows)


if __name__ == '__main__':
    main()
