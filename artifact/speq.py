"""Execute the real SpEQ artifact in its pinned LLVM/Python container."""
import json
import os
from pathlib import Path
from common import ARTIFACT, ROOT, command, required, save, setting, sha

def run(output, files, *, graph=False, func=None, cflags=''):
    output.mkdir(parents=True, exist_ok=True)
    source = Path(required('speq_root')).resolve()/'lleq-artifact'
    script = ARTIFACT/'benchmarks'/('graph_sizes/speq_capture.py' if graph else 'eqsat/speq_match.py')
    image = setting('speq_image', 'speq-lleq:extract')
    args = ['docker', 'run', '--rm', '--network', 'none', '--user', f'{os.getuid()}:{os.getgid()}',
            '-w', '/lleq', '-v', f'{source}:/lleq:ro', '-v', f'{script.parent}:/scripts:ro',
            '-v', f'{ROOT}/tools/cgeist/Test/polybench:/pb:ro', '-v', f'{output}:/out',
            image, 'python3', '/scripts/'+script.name, '--out', '/out/results.json']
    if func: args += ['--func', func]
    if cflags: args += ['--cflags', cflags]
    args += files
    command(args, output/'run.log', timeout=int(setting('speq_timeout', 1800)))
    command(['docker', 'image', 'inspect', image], output/'image.json')
    save(output/'inputs.json', dict(driver_sha256=sha(script),
         artifact_hashes={str(p.relative_to(source)):sha(p) for p in source.glob('*.py')}))
    return json.loads((output/'results.json').read_text())
