"""Expose declared source dependencies to the common raising frontend.

This is source linking, not matching: implementation bodies must be inlined
before their caller is eligible for a computation-level proof.
"""
import json
import re
from pathlib import Path

import spec_provenance as P


def load(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    if set(config) - {'sources', 'include_dirs', 'required_inline', 'inputs', 'unroll_threshold'} or not {'sources', 'include_dirs', 'required_inline', 'inputs'} <= set(config):
        raise ValueError('invalid source dependency configuration')
    threshold = config.get('unroll_threshold', 0)
    if not isinstance(threshold, int) or not 0 <= threshold <= 32:
        raise ValueError('source dependency unroll threshold must be between zero and 32')
    for name, expected in config['inputs'].items():
        if P.file_hash(P.input_path(name)) != expected:
            raise ValueError('source dependency changed: ' + name)
    if any(name not in config['inputs'] for name in config['sources']):
        raise ValueError('source dependency is not pinned')
    return config


def calls(text):
    return set(re.findall(r'\b(?:(?:func|llvm)\.)?call\s+@([\w.$-]+)\(', text))


def prepare(source, prefix, path, initial_ir):
    config = load(path)
    required = set(config['required_inline'])
    needed = sorted(calls(initial_ir) & required)
    if not needed:
        return None
    wrapper = Path(str(prefix) + '.frontend.cpp')
    # Preserve C linkage and source bytes. Parse dependency headers before
    # fixture macros (e.g. dimensions) can affect those headers.
    lines = ['#include ' + json.dumps(str(P.input_path(name))) for name in config['sources']]
    lines += ['extern "C" {', '#include ' + json.dumps(str(Path(source).resolve())), '}']
    wrapper.write_text('\n'.join(lines) + '\n')
    flags = ['-x', 'c++', '-std=c++17', '--no-inline=false']
    flags += ['-I' + str(P.input_path(name)) for name in config['include_dirs']]
    return wrapper, flags, dict(configuration=str(Path(path).resolve()),
        configuration_sha256=P.file_hash(path), inputs=config['inputs'],
        required_inline=sorted(required), requested=needed,
        unroll_threshold=config.get('unroll_threshold', 0))


def residual(text, record):
    from mlir_lite import Module
    # Reject remaining calls to implementation bodies as well as the public
    # aliases. In particular, inlining an alias around a recursive template
    # must not turn that recursive computation into a new opaque primitive.
    definitions = {f.name for f in Module(text).funcs if f.ops}
    return sorted(calls(text) & (set(record['required_inline']) | definitions))
