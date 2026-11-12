#!/usr/bin/env python3
"""Check recorded ATen fixture shapes and inputs before GPU execution."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import subprocess

import aten_fixture_batch as F
from aten_paper_shapes import load_shapes, materialize


def check(name, record, output, timeout):
    dest = output/name
    dest.mkdir(parents=True, exist_ok=True)
    cfg = F.configuration(name, record)
    source = dest/(name+'.c')
    materialize(F.BENCH.ATEN/(name+'.c'), source, record)
    prefix = F.harness(name, cfg).split('/* Included after')[0]
    prefix = prefix.replace('#include <cuda_runtime_api.h>', '')
    element = 'double' if cfg.get('dtype') == 'double' else 'float'
    body = '''
int main(void) {
  ELEMENT *p[NA];
  for(int a=0;a<NA;a++) {
    p[a]=malloc(sizes[a]*element_bytes[a]);
    if(!p[a])return 2;
    for(size_t i=0;i<sizes[a];i++) {
      double value=paper_value(a,i);
      if(integer_arrays[a]==2)((int8_t*)p[a])[i]=(int8_t)value;
      else if(integer_arrays[a])((int32_t*)p[a])[i]=(int32_t)value;
      else p[a][i]=value;
    }
  }
  for(int s=0;s<NS;s++)parameters[s]=paper_parameters[s];
  invoke_reference(p);
  for(int a=0;a<NA;a++)free(p[a]);
  puts("PASS sanitized reference");return 0;
}
'''.replace('ELEMENT', element)
    harness = dest/'check.c'
    harness.write_text(prefix+body)
    cmd = ['gcc', '-O1', '-g', '-fno-fast-math', '-ffp-contract=off',
           '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
           '-fno-omit-frame-pointer', '-D'+name+'=native_reference',
           str(source), str(harness), '-lm', '-o', str(dest/'check')]
    row = dict(kernel=name, shape_record=record, command=cmd,
               source_sha256=F.sha(source), input_scope='recorded paper recipe')
    with (dest/'compile.log').open('w') as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
    row['compile_rc'] = rc
    if rc == 0:
        with (dest/'run.log').open('w') as log:
            try:
                row['run_rc'] = subprocess.run([str(dest/'check')], stdout=log,
                    stderr=subprocess.STDOUT, timeout=timeout).returncode
            except subprocess.TimeoutExpired:
                row['run_rc'] = 124
    row['passed'] = row.get('run_rc') == 0
    F.save(dest/'result.json', row)
    print(name, 'PASS' if row['passed'] else 'FAIL', row.get('run_rc', rc), flush=True)
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shape-specs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cases', nargs='+')
    p.add_argument('--jobs', type=int, default=2)
    p.add_argument('--timeout', type=int, default=600)
    a = p.parse_args()
    records = load_shapes(a.shape_specs)
    names = a.cases or sorted(records)
    a.output = a.output.resolve()
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
        rows = list(pool.map(lambda name: check(name, records[name], a.output, a.timeout), names))
    F.save(a.output/'results.json', rows)
    print('Sanitized references passed', sum(r['passed'] for r in rows), '/', len(rows))


if __name__ == '__main__':
    main()
