#!/usr/bin/env python3
"""Measure sequential Egglog-enabled whole builds for MFEM or Llama."""

import argparse
import csv
import datetime as dt
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess

from run_polybench_compile_cost import (
    FIELDS,
    PRIMARY,
    ROOT,
    append_row,
    parse_time,
    read_completed,
    sha256,
)


LLAMA_FORWARD = (
    ("token_embedding", "kernel_llama_token_embedding"),
    ("attention_rmsnorm", "kernel_llama_attention_rmsnorm"),
    ("qkv_projection", "kernel_llama_qkv_projection"),
    ("rope_interleaved", "kernel_llama_rope"),
    ("rope_split", "kernel_llama_rope_split"),
    ("kv_cache_rw", "kernel_llama_kv_cache_rw"),
    ("attention_scores", "kernel_llama_attention_scores"),
    ("attention_mask_if", "kernel_llama_attention_mask"),
    ("attention_mask_select", "kernel_llama_attention_mask_select"),
    ("attention_softmax", "kernel_llama_attention_softmax"),
    ("attention_output", "kernel_llama_attention_output"),
    ("output_projection", "kernel_llama_output_projection"),
    ("residual_add", "kernel_llama_residual_add"),
    ("ffn_rmsnorm", "kernel_llama_ffn_rmsnorm"),
    ("gate_up_projection", "kernel_llama_gate_up_projection"),
    ("swiglu", "kernel_llama_swiglu"),
    ("down_projection", "kernel_llama_down_projection"),
    ("final_rmsnorm", "kernel_llama_final_rmsnorm"),
    ("lm_head_projection", "kernel_llama_lm_head_projection"),
)


def mfem_fixtures():
    manifest = ROOT / "issues/mfem_c_kernels/manifest.csv"
    with manifest.open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream)
                if row["variant"] == "normalized"]
    fixtures = [
        {
            "kernel": row["id"],
            "source": ROOT / "issues/mfem_c_kernels" / row["source"],
            "function": row["function"],
            "support_source": None,
        }
        for row in rows
    ]
    app_root = ROOT / "issues/mfem_c_kernels/application_extractions"
    with (app_root / "manifest.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            fixtures.append({
                "kernel": row["function"],
                "source": app_root / row["source"],
                "function": row["function"],
                "support_source": (
                    app_root / row["support_source"]
                    if row["support_source"] else None
                ),
            })
    return fixtures


def llama_fixtures():
    forward_source = ROOT / "third_party/cnn-extracted/llama_forward_ops.c"
    fixtures = [
        {"kernel": name, "source": forward_source, "function": function}
        for name, function in LLAMA_FORWARD
    ]
    fixtures.append({
        "kernel": "extended_forward",
        "source": ROOT / "third_party/cnn-extracted/llama2_extended_forward_bench.c",
        "function": "kernel_llama2_extended_forward",
        "support_source": None,
    })
    llama2_source = PRIMARY / "third_party/llama2.c/run.c"
    fixtures.extend(
        {"kernel": f"llama2c_{function}", "source": llama2_source,
         "function": function, "support_source": None}
        for function in ("matmul", "rmsnorm", "softmax")
    )
    return fixtures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("mfem", "llama"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    if args.repetitions < 1:
        raise SystemExit("--repetitions must be positive")
    if not args.output.is_absolute():
        raise SystemExit("--output must be an absolute path")

    fixtures = mfem_fixtures() if args.suite == "mfem" else llama_fixtures()
    for fixture in fixtures:
        if not fixture["source"].is_file():
            raise SystemExit(f"missing source: {fixture['source']}")
        if (fixture.get("support_source") is not None and
                not fixture["support_source"].is_file()):
            raise SystemExit(f"missing support source: {fixture['support_source']}")

    args.output.mkdir(parents=True, exist_ok=True)
    summary = args.output / "runs.csv"
    completed = read_completed(summary)
    build_script = ROOT / "scripts/correctness/polygeist_build.sh"
    tool_paths = {
        "cgeist": PRIMARY / "build/bin/cgeist",
        "polygeist_opt": PRIMARY / "build/bin/polygeist-opt",
        "mlir_opt": PRIMARY / "llvm-project/build/bin/mlir-opt",
        "mlir_translate": PRIMARY / "llvm-project/build/bin/mlir-translate",
        "clang": PRIMARY / "llvm-project/build/bin/clang",
        "build_script": build_script,
    }
    for name, path in tool_paths.items():
        if not path.exists():
            raise SystemExit(f"missing {name}: {path}")

    source_manifest = [
        {
            "kernel": fixture["kernel"],
            "source": str(fixture["source"]),
            "function": fixture["function"],
            "support_source": (str(fixture.get("support_source"))
                               if fixture.get("support_source") else None),
            "sha256": sha256(fixture["source"]),
        }
        for fixture in fixtures
    ]
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(source_manifest, indent=2) + "\n")
    metadata = {
        "schema": 1,
        "suite": args.suite,
        "scope": "C source through linked AArch64 executable",
        "target": "Jetson AArch64 cross-compile on x86 host; executable not run",
        "fixture_count": len(fixtures),
        "modes": ["egglog"],
        "repetitions_per_mode": args.repetitions,
        "sequential": True,
        "semantic_fallback_disabled": True,
        "matcher_iteration_limit": 8,
        "matcher_ast_node_ceiling": 32,
        "per_candidate_wall_timeout_seconds": None,
        "timeout_seconds_per_build": args.timeout,
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "host": platform.node(),
        "platform": platform.platform(),
        "logical_cpu_count": os.cpu_count(),
        "load_average_at_campaign_start": os.getloadavg(),
        "manifest": str(manifest_path.relative_to(ROOT)),
        "manifest_sha256": sha256(manifest_path),
        "tools": {name: {"path": str(path), "sha256": sha256(path)}
                  for name, path in tool_paths.items()},
        "runner_sha256": sha256(Path(__file__)),
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    jobs = [
        (fixture, repetition)
        for repetition in range(1, args.repetitions + 1)
        for fixture in fixtures
    ]
    for job_index, (fixture, repetition) in enumerate(jobs, 1):
        kernel = fixture["kernel"]
        identity = (kernel, "egglog", repetition)
        if identity in completed:
            print(f"[{job_index}/{len(jobs)}] skip {kernel} egglog r{repetition}",
                  flush=True)
            continue

        run_dir = args.output / "runs" / kernel / f"egglog-r{repetition}"
        run_dir.mkdir(parents=True, exist_ok=True)
        executable = run_dir / f"{kernel}.aarch64"
        harness = run_dir / "link_harness.c"
        support_include = (
            f'#include "{fixture["support_source"]}"\n'
            if fixture.get("support_source") else ""
        )
        harness.write_text(
            "#include <stdint.h>\n" + support_include +
            f"extern void {fixture['function']}(void);\n"
            f"static void (*volatile keep_kernel)(void) = {fixture['function']};\n"
            "int main(void) { return keep_kernel == 0; }\n"
        )
        log = run_dir / "build.log"
        time_log = run_dir / "time.txt"
        telemetry = run_dir / "matcher-telemetry.json"
        command = [
            "/usr/bin/timeout", f"{args.timeout}s", "/usr/bin/time", "-f",
            "wall_seconds=%e\nuser_seconds=%U\nsystem_seconds=%S\npeak_rss_kib=%M",
            "-o", str(time_log), str(build_script), "--target=jetson",
            f"--function={fixture['function']}", f"--harness={harness}",
            "-o", str(executable), str(fixture["source"]), "-O3",
        ]
        env = os.environ.copy()
        env.update({
            "POLYGEIST_ROOT": str(ROOT),
            "PATH": ":".join([
                str(PRIMARY / "build/bin"),
                str(PRIMARY / "llvm-project/build/bin"),
                env.get("PATH", ""),
            ]),
            "MLIR_OPT": str(tool_paths["mlir_opt"]),
            "MLIR_TRANSLATE": str(tool_paths["mlir_translate"]),
            "CLANG": str(tool_paths["clang"]),
            "PYTHON": "/usr/bin/python3",
            "POLYGEIST_MATCHER_MODE": "egglog",
            "POLYGEIST_MATCHER_DISABLE_SEMANTIC_FALLBACK": "1",
            "POLYGEIST_MATCHER_TELEMETRY_JSON": str(telemetry),
            "POLYGEIST_CUTENSORNET_ROOT": "/tmp/polygeist_cutensornet_aarch64/unified",
            "POLYGEIST_LOWER_SUBMAP_BEFORE_DEBUFFERIZE": "1",
        })
        print(f"[{job_index}/{len(jobs)}] {kernel} egglog r{repetition}",
              flush=True)
        started = dt.datetime.now(dt.timezone.utc)
        with log.open("w") as stream:
            process = subprocess.run(command, cwd=ROOT, env=env,
                                     stdout=stream, stderr=subprocess.STDOUT)
        finished = dt.datetime.now(dt.timezone.utc)
        timing = parse_time(time_log)
        append_row(summary, {
            "kernel": kernel,
            "mode": "egglog",
            "repetition": repetition,
            "status": ("ok" if process.returncode == 0 else
                       "timeout" if process.returncode == 124 else "error"),
            "returncode": process.returncode,
            "timed_out": str(process.returncode == 124).lower(),
            "wall_seconds": timing.get("wall_seconds", ""),
            "user_seconds": timing.get("user_seconds", ""),
            "system_seconds": timing.get("system_seconds", ""),
            "peak_rss_kib": timing.get("peak_rss_kib", ""),
            "started_utc": started.isoformat(),
            "finished_utc": finished.isoformat(),
            "source": str(fixture["source"]),
            "source_sha256": sha256(fixture["source"]),
            "executable_sha256": sha256(executable) if executable.exists() else "",
            "log": str(log.relative_to(args.output)),
            "time_log": str(time_log.relative_to(args.output)),
            "telemetry": str(telemetry.relative_to(args.output)),
            "command": shlex.join(command),
        })
        completed.add(identity)


if __name__ == "__main__":
    main()
