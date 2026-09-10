#!/usr/bin/env python3
"""Summarize Egglog-only whole-build campaigns for publication."""

import argparse
import csv
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import statistics


FIELDS = (
    "kernel", "successful_builds", "failed_builds", "status",
    "wall_median_seconds", "wall_q1_seconds", "wall_q3_seconds",
    "peak_rss_median_kib", "matcher_median_ms",
    "selected_matches_median", "failure_reason",
)


def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def telemetry_value(campaign, row, key):
    path = campaign / row.get("telemetry", "")
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text()).get(key)
    except (OSError, json.JSONDecodeError):
        return None
    return float(value) if isinstance(value, (int, float)) else None


def failure_reason(campaign, rows):
    reasons = Counter()
    for row in rows:
        if row["status"] == "ok":
            continue
        path = campaign / row.get("log", "")
        text = path.read_text(errors="replace") if path.is_file() else ""
        errors = re.findall(r"^ERROR:\s*(.+)$", text, re.MULTILINE)
        if errors:
            reason = re.sub(r"; see /tmp/\S+", "", errors[-1])
        elif "undefined reference" in text:
            reason = "final link failed: unresolved external symbol"
        elif "wrapper.c:" in text and "error:" in text:
            reason = "generated wrapper compilation failed"
        elif row["status"] == "timeout":
            reason = "whole-build timeout"
        else:
            reason = "build failed; see retained log"
        reasons[reason] += 1
    return reasons.most_common(1)[0][0] if reasons else ""


def optional_median(values):
    return statistics.median(values) if values else None


def format_optional(value, digits=3):
    return "" if value is None else f"{value:.{digits}f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--suite", required=True)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    with (campaign / "runs.csv").open(newline="") as stream:
        runs = list(csv.DictReader(stream))
    metadata = json.loads((campaign / "metadata.json").read_text())

    by_kernel = defaultdict(list)
    for row in runs:
        by_kernel[row["kernel"]].append(row)

    per_kernel = []
    for kernel in sorted(by_kernel):
        rows = by_kernel[kernel]
        successful = [row for row in rows if row["status"] == "ok"]
        walls = [float(row["wall_seconds"]) for row in successful
                 if row.get("wall_seconds")]
        rss = [float(row["peak_rss_kib"]) for row in successful
               if row.get("peak_rss_kib")]
        matcher = [value for row in successful
                   if (value := telemetry_value(
                       campaign, row, "matcher_elapsed_ms")) is not None]
        matches = [value for row in successful
                   if (value := telemetry_value(
                       campaign, row, "selected_matches")) is not None]
        ok_count = len(successful)
        status = ("pass" if ok_count == len(rows) else
                  "fail" if ok_count == 0 else "partial")
        per_kernel.append({
            "kernel": kernel,
            "successful_builds": ok_count,
            "failed_builds": len(rows) - ok_count,
            "status": status,
            "wall_median_seconds": format_optional(optional_median(walls)),
            "wall_q1_seconds": format_optional(percentile(walls, 0.25)),
            "wall_q3_seconds": format_optional(percentile(walls, 0.75)),
            "peak_rss_median_kib": format_optional(optional_median(rss), 1),
            "matcher_median_ms": format_optional(optional_median(matcher), 1),
            "selected_matches_median": format_optional(optional_median(matches), 1),
            "failure_reason": failure_reason(campaign, rows),
        })

    with (campaign / "per_kernel.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(per_kernel)

    successful_runs = [row for row in runs if row["status"] == "ok"]
    walls = [float(row["wall_seconds"]) for row in successful_runs
             if row.get("wall_seconds")]
    rss = [float(row["peak_rss_kib"]) for row in successful_runs
           if row.get("peak_rss_kib")]
    matcher = [value for row in successful_runs
               if (value := telemetry_value(
                   campaign, row, "matcher_elapsed_ms")) is not None]
    status_counts = Counter(row["status"] for row in per_kernel)
    summary = {
        "schema": 1,
        "suite": args.suite,
        "mode": "egglog",
        "fixture_count": len(per_kernel),
        "repetitions_per_fixture": metadata.get("repetitions_per_mode", 5),
        "attempted_builds": len(runs),
        "successful_builds": len(successful_runs),
        "failed_builds": len(runs) - len(successful_runs),
        "fully_successful_fixtures": status_counts["pass"],
        "partially_successful_fixtures": status_counts["partial"],
        "failed_fixtures": status_counts["fail"],
        "successful_wall_median_seconds": optional_median(walls),
        "successful_wall_q1_seconds": percentile(walls, 0.25),
        "successful_wall_q3_seconds": percentile(walls, 0.75),
        "successful_peak_rss_median_kib": optional_median(rss),
        "successful_matcher_median_ms": optional_median(matcher),
        "timeout_builds": sum(row["status"] == "timeout" for row in runs),
    }
    (campaign / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")

    report = f"""# {args.suite} Egglog whole-build results

- Scope: {summary['fixture_count']} source-selectable fixtures, five fresh
  sequential builds per fixture, {summary['attempted_builds']} attempts total.
- Mode: Egglog/equality saturation enabled; handwritten semantic fallback
  disabled. There is no equality-saturation-off comparison in this campaign.
- Boundary: C source through linked Jetson AArch64 executable; executables were
  not run. Timing and RSS medians below include successful builds only.
- Successful attempts: {summary['successful_builds']}/{summary['attempted_builds']}.
- Fully successful fixtures: {summary['fully_successful_fixtures']}/{summary['fixture_count']}.
- Partial fixtures: {summary['partially_successful_fixtures']}; consistently
  failing fixtures: {summary['failed_fixtures']}.
- Successful-build wall median: {summary['successful_wall_median_seconds']:.3f} s
  [Q1 {summary['successful_wall_q1_seconds']:.3f}, Q3 {summary['successful_wall_q3_seconds']:.3f}].
- Successful-build peak-RSS median: {summary['successful_peak_rss_median_kib'] / 1024:.1f} MiB.
- Successful-build matcher median: {summary['successful_matcher_median_ms']:.1f} ms.
- Whole-build timeouts: {summary['timeout_builds']}.

The raw attempt ledger is `runs.csv`; per-fixture medians and failure reasons
are in `per_kernel.csv`; exact sources, hashes, compiler binaries, parameters,
and host information are in `manifest.json` and `metadata.json`.
"""
    (campaign / "SUMMARY.md").write_text(report)


if __name__ == "__main__":
    main()
