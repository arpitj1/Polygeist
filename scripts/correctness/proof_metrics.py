"""Optional per-proof deadline and measurements for reproducible EqSat studies."""
import json
import atexit
import os
from pathlib import Path
import select
import subprocess
import sys
import time


class ProofDeadline(TimeoutError):
    pass


_worker = None
_worker_policy = None


def stop_worker():
    global _worker, _worker_policy
    if _worker is not None:
        if _worker.poll() is None:
            _worker.kill()
        _worker.wait()
        _worker.stdin.close()
        _worker.stdout.close()
    _worker = _worker_policy = None


atexit.register(stop_worker)


def graph_call(request, seconds):
    """Use a persistent fresh interpreter: native egglog state is never forked.

    A previous fork-based prototype could hang on expression-declaration locks
    inherited from the parent. JSON carries only source terms and plain results.
    Nonblocking pipe I/O makes the deadline cover input, execution, and output.
    """
    global _worker, _worker_policy
    policy=tuple(os.environ.get(k,'') for k in ('POLYGEIST_NUMERIC_MODEL','EQSAT_ABLATION'))
    started=time.monotonic()
    if _worker is None or _worker.poll() is not None or policy != _worker_policy:
        stop_worker()
        _worker=subprocess.Popen([sys.executable,str(Path(__file__).with_name('proof_worker.py'))],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,bufsize=0)
        _worker_policy=policy
        os.set_blocking(_worker.stdin.fileno(),False)
        os.set_blocking(_worker.stdout.fileno(),False)
    data=(json.dumps(request)+'\n').encode();offset=0;response=bytearray()
    try:
        while offset<len(data):
            remaining=seconds-(time.monotonic()-started)
            if remaining<=0 or not select.select([],[_worker.stdin],[],remaining)[1]:
                raise ProofDeadline('proof input deadline')
            offset+=os.write(_worker.stdin.fileno(),data[offset:])
        while b'\n' not in response:
            remaining=seconds-(time.monotonic()-started)
            if remaining<=0 or not select.select([_worker.stdout],[],[],remaining)[0]:
                raise ProofDeadline('equality-saturation proof deadline')
            chunk=os.read(_worker.stdout.fileno(),65536)
            if not chunk:raise RuntimeError('equality-saturation worker exited without a result')
            response.extend(chunk)
        result=json.loads(response)
        if 'error' in result:raise RuntimeError(result['error'])
        return result
    except BaseException:
        stop_worker()
        raise


def deadline_seconds():
    value = float(os.environ.get('POLYGEIST_SATURATION_TIMEOUT', '0'))
    if value < 0 or value != value or value == float('inf'):
        raise ValueError('POLYGEIST_SATURATION_TIMEOUT must be finite and nonnegative')
    return value


def measured_sizes():
    return os.environ.get('POLYGEIST_MATCHER_COLLECT_SIZES', '0') != '0'


def emit(phase, **fields):
    path = os.environ.get('POLYGEIST_MATCHER_TELEMETRY_JSONL')
    if path:
        record = dict(phase=phase, pid=os.getpid(),
                      eqsat_ablation=os.environ.get('EQSAT_ABLATION', 'full'),
                      numeric_model=os.environ.get('POLYGEIST_NUMERIC_MODEL', 'strict'),
                      **fields)
        with open(path, 'a') as output:
            output.write(json.dumps(record, sort_keys=True)+'\n')
