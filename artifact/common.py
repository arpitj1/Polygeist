"""Shared paths, checked commands and provenance for the paper evaluations."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ARTIFACT = Path(__file__).resolve().parent
ROOT = ARTIFACT.parent
CONFIG = json.loads(Path(os.environ.get("LL_ARTIFACT_CONFIG", ARTIFACT / "config.json")).read_text()) if Path(os.environ.get("LL_ARTIFACT_CONFIG", ARTIFACT / "config.json")).exists() else {}
OUT = Path(os.environ.get("LL_ARTIFACT_OUTPUT", ARTIFACT / "results")).resolve()
INPUTS = ARTIFACT / "inputs"
PYTHON = sys.executable

def setting(name, default=None):
    return os.environ.get("LL_" + name.upper(), CONFIG.get(name, default))

def required(name):
    value = setting(name)
    if not value:
        raise RuntimeError(f"Set {name} in artifact/config.json or LL_{name.upper()}; see artifact/README.md")
    return value

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")

def environment(**extra):
    env = dict(os.environ, POLYGEIST_ROOT=str(ROOT), POLYGEIST_SPEC_SUITE="lib_v3",
               POLYGEIST_NUMERIC_MODEL="relaxed", PYTHONDONTWRITEBYTECODE="1",
               PYTHON=PYTHON, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1")
    env["PATH"] = str(ROOT / "build/bin") + ":" + str(ROOT / "llvm-project/build/bin") + ":" + env.get("PATH", "")
    env["PYTHONPATH"] = str(ARTIFACT) + ":" + str(ROOT / "scripts/correctness") + ":" + str(ROOT / "scripts/correctness/lib_v2") + ":" + env.get("PYTHONPATH", "")
    for key, value in CONFIG.get("environment", {}).items():
        env[key] = str(value)
    env.update({key: str(value) for key, value in extra.items()})
    return env

def command(args, log, env=None, timeout=7200, cwd=ROOT, check=True):
    args = list(map(str, args)); log = Path(log); log.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with log.open("a") as stream:
        stream.write("COMMAND " + shlex.join(args) + "\n"); stream.flush()
        try:
            rc = subprocess.run(args, cwd=cwd, env=env or environment(), stdout=stream,
                                stderr=stream, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            rc = 124
    save(log.with_suffix(log.suffix + ".command.json"), dict(command=args, returncode=rc,
         seconds=time.monotonic()-start, cwd=str(cwd)))
    if check and rc:
        raise RuntimeError(f"Command failed ({rc}); see {log}")
    return rc

def provenance(out):
    save(Path(out) / "provenance.json", dict(
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        uptime=subprocess.check_output(["uptime"], text=True).strip(),
        machine=subprocess.check_output(["uname", "-a"], text=True).strip(),
        tools={str(p.relative_to(ROOT)): sha(p) for p in
               [ROOT / "build/bin/cgeist", ROOT / "build/bin/polygeist-opt"] if p.exists()},
        suite=os.environ.get("LL_ARTIFACT_SUITE"), specification_suite="lib_v3"))

def polybench():
    base = ROOT / "tools/cgeist/Test/polybench"
    return {p.stem: p for p in base.rglob("*.c") if p.stem != "polybench"}
