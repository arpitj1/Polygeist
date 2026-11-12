"""Run SpEQ's recognizer (lleq-artifact driver.py logic) on arbitrary C kernels.

Runs inside the speq-lleq container with the artifact mounted at /lleq and
this directory at /scripts:

  docker run --rm -w /lleq -v ARTIFACT:/lleq -v SCRIPTS:/scripts -v SRC:/src \
      speq-lleq python3 /scripts/speq_match.py --out /scripts/out.json \
      [--cflags "..."] [--func NAME] file.c ...

The library (GEMM, GEMV, GEMV sink/perm variant, Histogram), the opt pipeline
and the eqsat schedule (5 transform, 1 expand, 3 transform) are copied verbatim
from driver.py so results are comparable with the paper's Table 3.  Unlike
driver.py, every REV loop nest is reported, not just whether any matched.
"""
import argparse, json, os, re, subprocess, sys, tempfile, time

sys.path.insert(0, "/lleq")
from parseIR import foldFromStr, ToEggAbstract, egraph, Val, expand, transform  # noqa: E402
from egglog import rewrite  # noqa: E402

BIN = "/llvm-lleq/build/bin"
CLANG_ARGS = ["-emit-llvm", "-O0", "-Xclang", "-disable-O0-optnone", "-fno-discard-value-names"]
OPT_PASSES = "mem2reg,loop-rotate,instcombine,simplifycfg,loop-simplify,gvn,lcssa,print<revpass>"


class FIR:  # same as driver.py
    def __init__(self, path, name, params):
        self.path, self.name, self.params = path, name, params

    def add(self, func=None):
        out = subprocess.run([f"{BIN}/opt", "-S", "-p", "print<revpass>", self.path,
                              "-disable-output"], capture_output=True).stderr.decode()
        fir = re.search(r"(REV Start)((.|\n)*?)(REV End)", out, re.M).group(2)
        _, _, ast = foldFromStr(fir)
        abstr = ToEggAbstract(self.params)
        pattern = abstr.run(ast)
        if func is None:
            sig = ", ".join(["name: Val"] + [str(v).replace("%", "").replace(".", "_") + ": Val"
                                             for v in abstr.get_params()])
            exec(f"def {self.name}({sig}) -> Val: ...", globals())
            func = egraph.function(globals()[self.name], cost=0)
        self.func = func
        egraph.register(rewrite(pattern, ruleset=transform).to(func(abstr.root, *abstr.get_params())))


def sh(cmd):
    r = subprocess.run(cmd, capture_output=True)
    return r.returncode, r.stdout.decode(errors="replace"), r.stderr.decode(errors="replace")


def analyse(src, cflags, func, workdir):
    rec = {"file": src, "func": func, "nests": []}
    base = os.path.join(workdir, os.path.basename(src)[:-2])
    t_all = time.perf_counter()
    rc, _, err = sh([f"{BIN}/clang", "-S", *CLANG_ARGS, *cflags, f"-I{os.path.dirname(src)}",
                     src, "-o", base + ".ll"])
    rec["clang_s"] = time.perf_counter() - t_all
    if rc:
        rec["status"] = "clang_fail"; rec["error"] = err[-600:]; return rec
    ll = base + ".ll"
    if func:
        rc, _, err = sh([f"{BIN}/llvm-extract", "-S", f"--func={func}", ll, "-o", base + ".k.ll"])
        if rc:
            rec["status"] = "extract_fail"; rec["error"] = err[-600:]; return rec
        ll = base + ".k.ll"
    t0 = time.perf_counter()
    rc, _, err = sh([f"{BIN}/opt", "-S", "-p", OPT_PASSES, "-enable-load-in-loop-pre=false",
                     ll, "-o", base + ".opt.ll"])
    rec["opt_s"] = time.perf_counter() - t0
    if rc:
        rec["status"] = "opt_fail"; rec["error"] = err[-600:]; return rec
    firs = [m.group(2) for m in re.finditer(r"(REV Start)((.|\n)*?)(REV End)", err, re.M)]
    for fir in firs:
        nest = {"fir_head": fir.strip()[:160]}
        try:
            t0 = time.perf_counter()
            formats, rtc, ast = foldFromStr(fir)
            nest["parse_s"] = time.perf_counter() - t0
        except Exception as e:  # driver.py: "skipping ..."
            nest["status"] = "parse_fail"; nest["error"] = repr(e)[:300]
            rec["nests"].append(nest); continue
        nest["formats"] = list(formats[0]) if formats else []
        nest["rtc"] = [str(c) for c in rtc]
        try:
            t0 = time.perf_counter()
            with egraph:
                e = ast.toEgg()
                egraph.register(e)
                egraph.run(5, ruleset=transform)
                egraph.run(1, ruleset=expand)
                egraph.run(3, ruleset=transform)
                res = str(egraph.extract(e))
                nest["eqsat_s"] = time.perf_counter() - t0
                dot = egraph._egraph.to_graphviz_string()
                # e-nodes: one '"id"[label=' per node (primitives are inlined
                # in labels, i.e. comparable to all_function_sizes); e-classes:
                # one 'subgraph "cluster_N"' per class.
                nest["egraph_nodes"] = len(re.findall(r'^\s*"[^"]+"\[label=', dot, re.M))
                nest["egraph_classes"] = len(re.findall(r'subgraph "cluster_\d+"', dot))
        except Exception as e:
            nest["status"] = "eqsat_fail"; nest["error"] = repr(e)[:300]
            rec["nests"].append(nest); continue
        hits = [k for k in KERNELS if re.search(rf"\b{k}\(", res)]
        nest["status"] = "match" if hits else "no_match"
        nest["matched"] = hits
        nest["extracted"] = res[:1500]
        rec["nests"].append(nest)
    rec["status"] = "ok" if firs else "no_fir"
    rec["total_s"] = time.perf_counter() - t_all
    rec["n_match"] = sum(n.get("status") == "match" for n in rec["nests"])
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cflags", default="")
    ap.add_argument("--func", default=None, help="extract only this function before revpass")
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()

    t_setup = time.perf_counter()
    with egraph:
        # Uppercase names as in driver.py: FIR.add exec()s a global function
        # named after the kernel ("gemm", "gemv", ...), which would clobber
        # lowercase variables.
        GEMM = FIR("/lleq/analysis/gemm_ref.ll", "gemm", ["%ni", "%nj", "%nk", "%alpha", "%beta", "%C", "%A", "%B"])
        GEMM.add()
        GEMV = FIR("/lleq/analysis/gemv.ll", "gemv", ["%m", "%n", "%alpha", "%a", "%x", "%beta", "%y"])
        GEMV.add()
        FIR("/lleq/analysis/gemv_sink_perm.ll", "gemv",
            ["%m", "%n", "%alpha", "%a", "%x", Val.constant_real(0.0), "%y"]).add(GEMV.func)
        FIR("/lleq/analysis/histogram.ll", "histogram", ["%N", "%buckets", "%key", "%add"]).add()
        KERNELS = ["gemm", "gemv", "histogram"]
        setup_s = time.perf_counter() - t_setup

        results = []
        with tempfile.TemporaryDirectory() as wd:
            for f in a.files:
                func = a.func
                if func == "AUTO":  # PolyBench: kernel_<dirname>
                    func = "kernel_" + os.path.basename(f)[:-2].replace("-", "_")
                r = analyse(f, a.cflags.split(), func, wd)
                print(f"{os.path.basename(f):24s} {r['status']:12s} "
                      f"nests={len(r['nests'])} match={r.get('n_match', 0)} "
                      f"{[n.get('matched') or n['status'] for n in r['nests']]}", flush=True)
                r["setup_s"] = setup_s
                results.append(r)
    with open(a.out, "w") as fh:
        json.dump(results, fh, indent=1)
