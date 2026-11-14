"""Compile manifest-declared scalar support for reference execution.

The support uses pinned public definitions, shared by the original C and raised
code. It supplies opaque scalar function implementations, not matching rules.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import threading

import spec_provenance as P

_lock = threading.Lock()
_objects = {}


def objects(manifest, work, compiler):
    result = []
    for config in manifest.get('validation', {}).get('support', []):
        if set(config) != {'source', 'include_dirs', 'inputs'}:
            raise ValueError('invalid reference support schema')
        source = P.input_path(config['source'])
        inputs = {name: P.file_hash(P.input_path(name)) for name in config['inputs']}
        if config['source'] not in inputs:
            raise ValueError('support source must be a recorded input')
        includes = [P.input_path(name) for name in config['include_dirs']]
        digest = hashlib.sha256(json.dumps(dict(config=config, inputs=inputs,
            compiler=P.file_hash(compiler)), sort_keys=True).encode()).hexdigest()
        with _lock:
            if digest not in _objects:
                directory = Path(work) / 'support' / digest
                directory.mkdir(parents=True, exist_ok=True)
                output = directory / 'support.o'
                command = [str(compiler), '-x', 'c++', '-std=c++17', '-O0',
                           '-ffp-contract=off', *['-I' + str(p) for p in includes],
                           '-c', str(source), '-o', str(output)]
                p = subprocess.run(command, capture_output=True, text=True, timeout=120)
                (directory / 'build.log').write_text(p.stdout + p.stderr)
                if p.returncode:
                    raise ValueError('reference support compilation failed: ' + p.stderr[-500:])
                _objects[digest] = output
            result.append(_objects[digest])
    return result
