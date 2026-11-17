"""Content-validated storage for derived library-call metadata.

Reference C prototypes are preprocessed on every invocation, including cache
hits. This accounts for transitive includes, compiler defaults and environment
without trusting a saved list of header dependencies. Application proofs and
backend eligibility checks are never cached here.
"""
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile


def encode(value):
    if dataclasses.is_dataclass(value):
        return {"kind": type(value).__name__, "fields": {
            f.name: encode(getattr(value, f.name)) for f in dataclasses.fields(value)}}
    if isinstance(value, dict):
        return {"kind": "dict", "items": [[encode(k), encode(v)] for k, v in value.items()]}
    if isinstance(value, (list, tuple)):
        return {"kind": type(value).__name__, "items": [encode(v) for v in value]}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported ABI metadata type: {type(value)}")


def decode(value, classes):
    if not isinstance(value, dict):
        return value
    kind = value["kind"]
    if kind == "dict":
        return {decode(k, classes): decode(v, classes) for k, v in value["items"]}
    if kind in ("tuple", "list"):
        values = [decode(v, classes) for v in value["items"]]
        return tuple(values) if kind == "tuple" else values
    cls = classes[kind]
    return cls(**{k: decode(v, classes) for k, v in value["fields"].items()})


def serialized(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def fingerprint(A, names, prototypes):
    """Hash all non-preprocessor inputs read by the ABI derivation code."""
    paths = set((A.REPO / "scripts/correctness").rglob("*.py"))
    paths.update((A.SPEC_DIR / "manifest").glob("*.json"))
    paths.update((A.SPEC_DIR / "raised").glob("*.mlir"))
    paths.update((A.SPEC_DIR / "raised").glob("*.c"))
    suite = A.SPEC_DIR.name.removeprefix("lib_")
    paths.add(A.SPEC_DIR.parent / f"kernel_library_{suite}.mlir")
    for relative in ("BLAS/SRC", "SRC"):
        paths.update((A.NETLIB / relative).glob("*.f"))
    for headers in A.HEADER_FAMILIES.values():
        paths.update(A.CUDA_HEADERS / h for h in headers)
    paths.update(p for p in (A.REPO / "runtime").iterdir()
                 if p.suffix in (".c", ".h", ".inc", ".cu", ".cpp"))
    h = hashlib.sha256()
    policy = {name: os.environ.get(name) for name in (
        "POLYGEIST_NUMERIC_MODEL", "POLYGEIST_KERNEL_LIBRARY")}
    h.update(serialized(encode((1, sys.version, str(A.SPEC_DIR), str(A.NETLIB),
                               str(A.CUDA_HEADERS), A.HEADER_FAMILIES,
                               policy, names, prototypes))).encode())
    for path in sorted(paths):
        h.update(str(path).encode() + b"\0")
        h.update(hashlib.sha256(path.read_bytes()).digest() if path.is_file() else b"missing")
    return h.hexdigest()


def directory():
    if os.environ.get("POLYGEIST_ABI_NO_CACHE", "0") != "0":
        return None
    default = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "polygeist/abi"
    return Path(os.environ.get("POLYGEIST_ABI_CACHE_DIR", default))


def load(directory, key, classes):
    try:
        record = json.loads((directory / (key + ".json")).read_text())
        payload = record["payload"]
        if record["key"] != key or record["sha256"] != hashlib.sha256(serialized(payload).encode()).hexdigest():
            return None
        table = decode(payload, classes)
        if not isinstance(table, tuple) or len(table) != 3 or not all(isinstance(v, dict) for v in table):
            return None
        return table
    except (OSError, ValueError, KeyError, TypeError, AttributeError, RecursionError):
        return None


def save(directory, key, table):
    temporary = None
    try:
        directory.mkdir(parents=True, exist_ok=True)
        payload = encode(table)
        record = dict(key=key, payload=payload,
                      sha256=hashlib.sha256(serialized(payload).encode()).hexdigest())
        with tempfile.NamedTemporaryFile(mode="w", dir=directory, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(record, stream, separators=(",", ":"))
        temporary.replace(directory / (key + ".json"))
    except OSError:
        pass  # A read-only cache must not prevent compilation.
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def event(status, key, elapsed):
    path = os.environ.get("POLYGEIST_ABI_CACHE_LOG")
    if path:
        with open(path, "a") as stream:
            stream.write(json.dumps(dict(status=status, key=key, seconds=elapsed)) + "\n")
