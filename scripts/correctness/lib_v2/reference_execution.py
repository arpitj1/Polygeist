"""Lower and execute imported references with the compiler validation pipeline."""
from __future__ import annotations
import re
import subprocess
import sys
from pathlib import Path
import recipe as R

HERE = Path(__file__).resolve().parent


REPO = R.REPO


LLVM_BIN = REPO / "llvm-project/build/bin"


CLANG = str(LLVM_BIN / "clang")


MLIR_OPT = str(LLVM_BIN / "mlir-opt")


MLIR_TRANSLATE = str(LLVM_BIN / "mlir-translate")


MLIR_LIBDIR = REPO / "llvm-project/build/lib"


PY = sys.executable


LOWER_TO_LLVM = [
    "--convert-math-to-llvm", "--convert-math-to-libm",
    "--empty-tensor-to-alloc-tensor", "--lower-affine",
    "--one-shot-bufferize=bufferize-function-boundaries",
    "--convert-linalg-to-loops", "--convert-scf-to-cf", "--expand-strided-metadata",
    "--lower-affine", "--convert-arith-to-llvm", "--convert-index-to-llvm",
    "--finalize-memref-to-llvm", "--convert-func-to-llvm",
    "--reconcile-unrealized-casts"]


def sh(cmd, log: Path, timeout=900, stdout=None) -> int:
    with log.open("a") as fh:
        fh.write("$ " + " ".join(map(str, cmd)) + "\n")
        fh.flush()
        try:
            p = subprocess.run([str(c) for c in cmd], stdout=stdout or fh, stderr=fh,
                               timeout=timeout)
            return p.returncode
        except subprocess.TimeoutExpired:
            fh.write("TIMEOUT\n")
            return 124


def lower(inp: Path, d: Path, tag: str, fn: str, log: Path) -> Path | None:
    legal = d / f"{tag}_legal.mlir"
    if sh([PY, HERE / "legalize_tensor_submaps.py", inp, "-o", legal], log):
        return None
    canon = d / f"{tag}_canon.mlir"
    if sh([R.POLYGEIST_OPT, "--lower-polygeist-submap", "--canonicalize-polygeist",
           legal, "-o", canon], log):
        return None
    if "polygeist.submap" in canon.read_text():
        c2 = d / f"{tag}_canon_loops.mlir"
        if sh([R.POLYGEIST_OPT, "--convert-linalg-to-loops", "--canonicalize",
               "--lower-polygeist-submap", "--canonicalize", canon, "-o", c2], log):
            return None
        canon = c2
    t = canon.read_text()
    t = re.sub(r"bufferization\.to_tensor (%[^ ]*) :", r"bufferization.to_tensor \1 restrict :", t)
    canon.write_text(t)
    llvm_mlir = d / f"{tag}_llvm.mlir"
    if sh([MLIR_OPT, *LOWER_TO_LLVM, canon, "-o", llvm_mlir], log):
        return None
    ll = d / f"{tag}_kernel.ll"
    if sh([MLIR_TRANSLATE, "--mlir-to-llvmir", llvm_mlir, "-o", ll], log):
        return None
    ll.write_text(re.sub(rf"@{fn}\b", f"@{fn}_impl", ll.read_text()))
    return ll


