"""Public SSH transport. Authentication stays in the evaluator's SSH config.

The existing campaign drivers use run(command, timeout) and upload(src, dst).
Set LL_SSH_HOST or ssh_host in config.json. The literal value 'local' runs on
the current machine. No host key checks are disabled.
"""
import os
import importlib.util
from functools import lru_cache
from pathlib import Path
import shlex
import shutil
import subprocess
from common import required, setting

@lru_cache(maxsize=1)
def transport():
    """Optional site adapter with run/upload; default remains standard SSH."""
    path = setting('transport_module')
    if not path:
        return None
    spec = importlib.util.spec_from_file_location('artifact_site_transport', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def run(command, timeout=300):
    if transport() is not None:
        return transport().run(command, timeout=timeout)
    host = required("ssh_host")
    args = ["bash", "-lc", command] if host == "local" else ["ssh", "-o", "BatchMode=yes", host, "bash -lc " + shlex.quote(command)]
    try:
        p = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, timeout=timeout)
        return p.returncode, p.stdout
    except subprocess.TimeoutExpired as e:
        return 124, str(e)

def upload(local_path, remote_path):
    if transport() is not None:
        rc, _ = run("mkdir -p " + shlex.quote(str(Path(remote_path).parent)), 30)
        if rc:
            return rc
        return transport().upload(local_path, remote_path)
    host = required("ssh_host"); local_path = Path(local_path)
    rc, _ = run("mkdir -p " + shlex.quote(str(Path(remote_path).parent)), 30)
    if rc:
        return rc
    if host == "local":
        if local_path.resolve() != Path(remote_path).resolve():
            shutil.copy2(local_path, remote_path)
        return 0
    return subprocess.run(["scp", "-q", "-o", "BatchMode=yes", str(local_path),
                           host + ":" + shlex.quote(str(remote_path))]).returncode

def runtime_env():
    lib = setting("remote_library_path", "/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu")
    return "LD_LIBRARY_PATH=" + shlex.quote(lib) + " OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "
