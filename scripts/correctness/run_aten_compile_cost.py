#!/usr/bin/env python3
"""Measure fresh end-to-end ATen fixture builds sequentially."""

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


DEFAULT_OUTPUT = (
    ROOT / "issues/equality_saturation_eval/aten_whole_compile_20260910"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--kernel", action="append", default=[])
    parser.add_argument("--mode", action="append",
                        choices=("egglog", "syntactic"), default=[])
    args = parser.parse_args()
    if args.repetitions < 1:
        raise SystemExit("--repetitions must be positive")

    source_root = ROOT / "issues/aten_c_kernels"
    sources = sorted(source_root.glob("aten_*.c"))
    requested = set(args.kernel)
    if requested:
        normalized = {name if name.startswith("aten_") else f"aten_{name}"
                      for name in requested}
        sources = [source for source in sources if source.stem in normalized]
        missing = normalized - {source.stem for source in sources}
        if missing:
            raise SystemExit(f"unknown kernels: {sorted(missing)}")
    if not sources:
        raise SystemExit("no ATen fixtures selected")
    selected_modes = tuple(dict.fromkeys(args.mode or ("egglog", "syntactic")))

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
        {"kernel": source.stem, "source": str(source.relative_to(ROOT)),
         "sha256": sha256(source)} for source in sources
    ]
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(source_manifest, indent=2) + "\n")
    metadata = {
        "schema": 1,
        "scope": "standalone C fixture through linked AArch64 executable",
        "target": "Jetson AArch64 cross-compile on x86 host; executable not run",
        "fixture_count": len(sources),
        "modes": list(selected_modes),
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

    jobs = []
    for repetition in range(1, args.repetitions + 1):
        for index, source in enumerate(sources):
            modes = selected_modes
            if len(modes) > 1 and (repetition + index) % 2:
                modes = tuple(reversed(modes))
            for mode in modes:
                jobs.append((source, mode, repetition))

    for job_index, (source, mode, repetition) in enumerate(jobs, 1):
        kernel = source.stem
        identity = (kernel, mode, repetition)
        if identity in completed:
            print(f"[{job_index}/{len(jobs)}] skip {kernel} {mode} r{repetition}", flush=True)
            continue

        run_dir = args.output / "runs" / kernel / f"{mode}-r{repetition}"
        run_dir.mkdir(parents=True, exist_ok=True)
        executable = run_dir / f"{kernel}.aarch64"
        harness = run_dir / "link_harness.c"
        harness.write_text(
            "#include <stdint.h>\n"
            f"extern void {kernel}(void);\n"
            f"static void (*volatile keep_kernel)(void) = {kernel};\n"
            "int main(void) { return keep_kernel == 0; }\n"
        )
        log = run_dir / "build.log"
        time_log = run_dir / "time.txt"
        telemetry = run_dir / "matcher-telemetry.json"
        command = [
            "/usr/bin/timeout", f"{args.timeout}s", "/usr/bin/time", "-f",
            "wall_seconds=%e\nuser_seconds=%U\nsystem_seconds=%S\npeak_rss_kib=%M",
            "-o", str(time_log), str(build_script), "--target=jetson",
            f"--function={kernel}", f"--harness={harness}", "-o",
            str(executable), str(source), "-O3",
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
            "POLYGEIST_MATCHER_MODE": mode,
            "POLYGEIST_MATCHER_DISABLE_SEMANTIC_FALLBACK": "1",
            "POLYGEIST_MATCHER_TELEMETRY_JSON": str(telemetry),
            "POLYGEIST_CUTENSORNET_ROOT": "/tmp/polygeist_cutensornet_aarch64/unified",
            "POLYGEIST_LOWER_SUBMAP_BEFORE_DEBUFFERIZE": "1",
        })
        print(f"[{job_index}/{len(jobs)}] {kernel} {mode} r{repetition}", flush=True)
        started = dt.datetime.now(dt.timezone.utc)
        with log.open("w") as stream:
            process = subprocess.run(command, cwd=ROOT, env=env,
                                     stdout=stream, stderr=subprocess.STDOUT)
        finished = dt.datetime.now(dt.timezone.utc)
        timing = parse_time(time_log)
        append_row(summary, {
            "kernel": kernel,
            "mode": mode,
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
            "source": str(source.relative_to(ROOT)),
            "source_sha256": sha256(source),
            "executable_sha256": sha256(executable) if executable.exists() else "",
            "log": str(log.relative_to(args.output)),
            "time_log": str(time_log.relative_to(args.output)),
            "telemetry": str(telemetry.relative_to(args.output)),
            "command": shlex.join(command),
        })
        completed.add(identity)


if __name__ == "__main__":
    main()
