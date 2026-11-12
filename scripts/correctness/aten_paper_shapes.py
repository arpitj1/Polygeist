"""Apply pinned benchmark dimensions to separate copies of ATen fixtures."""
import copy
import json
from pathlib import Path
import re


def dimension_aliases(record):
    """Explicit publication-harness names for existing C dimensions."""
    aliases = record.get('dimension_aliases', {})
    if not isinstance(aliases, dict):
        raise ValueError('Dimension aliases must be a mapping')
    dims = record['dims']
    for alias, source in aliases.items():
        if (alias == source or alias not in dims or source not in dims or
                source in aliases or dims[alias] != dims[source]):
            raise ValueError('Invalid or inconsistent dimension alias: '+alias)
    return aliases


def load_shapes(path):
    rows = json.loads(Path(path).read_text())
    result = {}
    for row in rows:
        name = row['kernel']
        if name in result:
            raise ValueError('duplicate benchmark shape: '+name)
        if not row['dims'] or any(not re.fullmatch(r'[A-Za-z_]\w*', k)
                or type(v) is not int or v <= 0 for k, v in row['dims'].items()):
            raise ValueError('invalid benchmark dimensions: '+name)
        dimension_aliases(row)
        result[name] = row
    return result


def apply_shape(name, cfg, record):
    if record['kernel'] != name:
        raise ValueError('Shape record names a different kernel')
    aliases = dimension_aliases(record)
    if set(record['dims']) - set(cfg['dims']) - set(aliases):
        raise ValueError('Shape record contains unknown C dimensions')
    if any(source not in cfg['dims'] for source in aliases.values()):
        raise ValueError('Dimension alias lacks a C dimension')
    floating = any(a[1] == 'ptr' for a in cfg['args'])
    if floating and record['dtype'] != ('f64' if cfg.get('dtype') == 'double' else 'f32'):
        raise ValueError('Benchmark datatype differs from the C fixture')
    out = copy.deepcopy(cfg)
    out['dims'].update(record['dims'])
    out['dimension_scope'] = 'Pinned paper benchmark dimensions; separate C copy; original fixture unchanged'
    if record.get('correction'):
        out['dimension_scope'] = 'Corrected invalid paper shape: '+record['correction']
    return out


def materialize(original, output, record):
    original, output = Path(original), Path(output)
    if original.resolve() == output.resolve():
        raise ValueError('Never replace the original benchmark source')
    text = original.read_text()
    aliases = dimension_aliases(record)
    for name, value in record['dims'].items():
        if type(value) is not int or value <= 0:
            raise ValueError('invalid dimension '+name)
        # Preserve conditional default guards and every computation statement.
        pattern = rf'(^[ \t]*#[ \t]*define[ \t]+{re.escape(name)}[ \t]+)\d+[ \t]*$'
        if name in aliases:
            if re.search(pattern, text, flags=re.M):
                raise ValueError('Harness alias is also a C dimension: '+name)
            continue
        text, count = re.subn(pattern, lambda m: m[1]+str(value), text, flags=re.M)
        if count != 1:
            raise ValueError('Expected one literal dimension definition: '+name)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)


def input_recipe(name, cfg, record, benchmark):
    """Recover initialization expressions from the existing publication harness."""
    historical = copy.deepcopy(benchmark.CASES.get(name) or benchmark.auto_spec(name))
    historical['dims'].update(record['dims'])
    generated = benchmark.harness_text(name, historical)
    expressions = {}
    scalars = {}
    for arg, kind, value, _, _ in historical['args']:
        if kind in ('scalar', 'dscalar', 'iscalar'):
            scalars[arg] = record.get('scalar_args', {}).get(arg, value)
            continue
        match = re.search(rf'for\(size_t i=0;i<{re.escape(arg)}_n;\+\+i\) '
                          rf'{re.escape(arg)}_ref\[i\]=(.*?);', generated)
        if match is None:
            raise ValueError('Cannot recover publication initializer: '+name+'/'+arg)
        expressions[arg] = match[1]
    cfg['paper_initializers'] = expressions
    cfg['paper_parameters'] = scalars
    return cfg
