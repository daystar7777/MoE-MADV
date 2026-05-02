#!/usr/bin/env python3
import argparse
import itertools
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HOTSET = ROOT / "data" / "deepseek_q4_probe_hotset_k16.json"
DEFAULT_LOG_ROOT = ROOT / "logs"

PROMPTS = {
    "json": 'Return JSON only: {"status":"ok","note":"matrix"}',
    "plain": "In one sentence, explain why page cache matters for MoE inference.",
    "code": "Return code only: write a tiny Python function add(a, b).",
    "korean": "한 문장으로 MoE 추론에서 페이지 캐시가 중요한 이유를 설명해줘.",
    "prefill_long_plain": (
        "Summarize the following engineering note in three compact bullets. "
        "We are running a quantized mixture-of-experts model on a local machine. "
        "The model file is mapped with mmap, and routed expert weights are pulled "
        "from NVMe into the operating-system page cache when the CPU kernels touch "
        "those pages. The current bottleneck is not pure arithmetic throughput. "
        "Most wall time appears to be spent waiting for expert weight pages to "
        "become resident. A static hotset prewarm can help cold-start behavior, "
        "but it is too blunt for token-by-token decoding because adjacent routed "
        "expert sets overlap only partially. The next optimization should separate "
        "prefill and decode: prefill can tolerate broader merged reads, while "
        "decode needs early and precise routing-aware hints."
    ),
    "prefill_long_code": (
        "Review this Python snippet and return JSON with keys risk, fix, and test. "
        "def update_cache(cache, key, loader):\n"
        "    if key not in cache:\n"
        "        cache[key] = loader(key)\n"
        "    value = cache[key]\n"
        "    if value is None:\n"
        "        del cache[key]\n"
        "    return value\n"
        "Consider concurrency, exceptions, repeated calls, and observability. "
        "Keep the answer compact but specific."
    ),
    "decode_json_seed": 'Return a JSON array of short objects. Start with [{"step":1,"status":"',
    "decode_plain_seed": "Continue this practical optimization checklist:",
}

PREWARM_QUICK = [
    {"name": "gap0_chunk1", "merge_gap_mib": 0.0, "chunk_mib": 1},
    {"name": "gap4_chunk1", "merge_gap_mib": 4.0, "chunk_mib": 1},
    {"name": "gap8_chunk1", "merge_gap_mib": 8.0, "chunk_mib": 1},
    {"name": "gap4_chunk8", "merge_gap_mib": 4.0, "chunk_mib": 8},
    {"name": "gap4_chunk32", "merge_gap_mib": 4.0, "chunk_mib": 32},
    {"name": "gap4_chunk64", "merge_gap_mib": 4.0, "chunk_mib": 64},
]

INFER_QUICK = [
    {
        "name": "best_gap4_chunk1_madvise_on",
        "prewarm": True,
        "merge_gap_mib": 4.0,
        "chunk_mib": 1,
        "madvise": True,
    },
    {
        "name": "gap4_chunk1_madvise_off",
        "prewarm": True,
        "merge_gap_mib": 4.0,
        "chunk_mib": 1,
        "madvise": False,
    },
    {
        "name": "no_prewarm_madvise_on",
        "prewarm": False,
        "merge_gap_mib": 4.0,
        "chunk_mib": 1,
        "madvise": True,
    },
    {
        "name": "no_prewarm_madvise_off",
        "prewarm": False,
        "merge_gap_mib": 4.0,
        "chunk_mib": 1,
        "madvise": False,
    },
    {
        "name": "gap0_chunk1_madvise_on",
        "prewarm": True,
        "merge_gap_mib": 0.0,
        "chunk_mib": 1,
        "madvise": True,
    },
    {
        "name": "gap8_chunk1_madvise_on",
        "prewarm": True,
        "merge_gap_mib": 8.0,
        "chunk_mib": 1,
        "madvise": True,
    },
    {
        "name": "gap4_chunk8_madvise_on",
        "prewarm": True,
        "merge_gap_mib": 4.0,
        "chunk_mib": 8,
        "madvise": True,
    },
]


def parse_csv(value, allowed=None):
    result = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        if allowed and part not in allowed:
            raise argparse.ArgumentTypeError(f"unknown value {part!r}; choose from {', '.join(sorted(allowed))}")
        result.append(part)
    if not result:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    return result


def run_capture(cmd, env=None):
    started = time.monotonic()
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return proc.returncode, time.monotonic() - started, proc.stdout


def parse_prewarm_output(text):
    row = {}
    match = re.search(r"pread ranges: raw=(\d+), unique=(\d+), merged=(\d+), merge_gap=([0-9.]+) MiB", text)
    if match:
        row.update(
            {
                "raw_ranges": int(match.group(1)),
                "unique_ranges": int(match.group(2)),
                "merged_ranges": int(match.group(3)),
                "merge_gap_mib_reported": float(match.group(4)),
            }
        )
    match = re.search(
        r"pread bytes: planned=([0-9.]+) GiB, unique=([0-9.]+) GiB, actual=([0-9.]+) GiB, "
        r"overread=([0-9.]+) GiB \(([0-9.]+)%\)",
        text,
    )
    if match:
        row.update(
            {
                "planned_gib": float(match.group(1)),
                "unique_gib": float(match.group(2)),
                "actual_gib": float(match.group(3)),
                "overread_gib": float(match.group(4)),
                "overread_ratio": float(match.group(5)) / 100.0,
            }
        )
    match = re.search(r"done: warmed ([0-9.]+) GiB in ([0-9.]+)s \(([0-9.]+) GiB/s\)", text)
    if match:
        row.update(
            {
                "warmed_gib": float(match.group(1)),
                "prewarm_s": float(match.group(2)),
                "prewarm_gib_s": float(match.group(3)),
            }
        )
    return row


def metric(summary, dotted, default=None):
    cur = summary
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def profile_row_from_json(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = data.get("summary", {})
    token_timing = summary.get("token_timing", {})
    prefill_profile = token_timing.get("prefill_profile", {})
    decode_profile = token_timing.get("decode_profile", {})
    trace_timing = summary.get("trace_timing", {})
    trace_prefill_profile = trace_timing.get("prefill_profile", {})
    trace_decode_profile = trace_timing.get("decode_profile", {})
    trace_phase_windows = trace_timing.get("phase_windows_s", {})

    def trace_phase_wall(phase):
        window = trace_phase_windows.get(phase)
        if not window or len(window) != 2:
            return None
        return max(0.0, float(window[1]) - float(window[0]))

    return {
        "profile": str(path),
        "returncode": summary.get("returncode"),
        "wall_s": summary.get("wall_s"),
        "elapsed_s": summary.get("elapsed_s"),
        "io_active_ratio": summary.get("io_active_ratio"),
        "compute_or_other_ratio": summary.get("compute_or_other_ratio"),
        "avg_cpu_cores": summary.get("avg_cpu_cores"),
        "pageins": summary.get("pageins"),
        "disk_read_gib": summary.get("disk_read_gib"),
        "peak_rss_gib": summary.get("peak_rss_gib"),
        "prewarm_s": summary.get("prewarm_s"),
        "prewarm_gib_s": summary.get("prewarm_gib_s"),
        "prewarm_actual_gib": summary.get("prewarm_actual_gib"),
        "prewarm_overread_ratio": summary.get("prewarm_overread_ratio"),
        "prompt_tps": summary.get("prompt_tps"),
        "generation_tps": summary.get("generation_tps"),
        "prompt_tokens": metric(summary, "token_timing.prompt_tokens"),
        "generated_tokens": metric(summary, "token_timing.generated_tokens"),
        "prefill_s_est": metric(summary, "token_timing.prefill_s_est"),
        "decode_s_est": metric(summary, "token_timing.decode_s_est"),
        "prefill_wall_s": prefill_profile.get("wall_s"),
        "prefill_io_active_ratio": prefill_profile.get("io_active_ratio"),
        "prefill_disk_read_gib": prefill_profile.get("disk_read_gib"),
        "prefill_avg_cpu_cores": prefill_profile.get("avg_cpu_cores"),
        "decode_wall_s": decode_profile.get("wall_s"),
        "decode_io_active_ratio": decode_profile.get("io_active_ratio"),
        "decode_disk_read_gib": decode_profile.get("disk_read_gib"),
        "decode_avg_cpu_cores": decode_profile.get("avg_cpu_cores"),
        "trace_events": trace_timing.get("events"),
        "trace_rounds": trace_timing.get("rounds"),
        "trace_prefill_wall_s": trace_phase_wall("prefill"),
        "trace_prefill_io_active_ratio": trace_prefill_profile.get("io_active_ratio"),
        "trace_prefill_disk_read_gib": trace_prefill_profile.get("disk_read_gib"),
        "trace_prefill_avg_cpu_cores": trace_prefill_profile.get("avg_cpu_cores"),
        "trace_decode_wall_s": trace_phase_wall("decode"),
        "trace_decode_io_active_ratio": trace_decode_profile.get("io_active_ratio"),
        "trace_decode_disk_read_gib": trace_decode_profile.get("disk_read_gib"),
        "trace_decode_avg_cpu_cores": trace_decode_profile.get("avg_cpu_cores"),
        "generated_text": metric(summary, "token_timing.generated_text"),
    }


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def fmt(value, digits=2, suffix=""):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}{suffix}"
    return f"{value}{suffix}"


def write_markdown(path, prewarm_rows, infer_rows):
    lines = [
        "# DeepSeek Q4 Performance Matrix",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
    ]
    if prewarm_rows:
        lines.extend(
            [
                "## Prewarm I/O",
                "",
                "| case | gap MiB | chunk MiB | ranges | read GiB | overread | seconds | GiB/s |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in prewarm_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        row["case"],
                        fmt(row.get("merge_gap_mib")),
                        fmt(row.get("chunk_mib"), 0),
                        fmt(row.get("merged_ranges"), 0),
                        fmt(row.get("actual_gib")),
                        fmt((row.get("overread_ratio") or 0.0) * 100, 1, "%"),
                        fmt(row.get("prewarm_s")),
                        fmt(row.get("prewarm_gib_s")),
                    ]
                )
                + " |"
            )
        lines.append("")
    if infer_rows:
        lines.extend(
            [
                "## Inference",
                "",
                "| case | prompt | repeat | prewarm | madvise | gap | chunk | wall s | I/O active | disk GiB | prompt t/s | gen t/s | prefill I/O | decode I/O |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in infer_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        row["case"],
                        row["prompt_case"],
                        fmt(row.get("repeat"), 0),
                        "1" if row.get("prewarm") else "0",
                        "1" if row.get("madvise") else "0",
                        fmt(row.get("merge_gap_mib")),
                        fmt(row.get("chunk_mib"), 0),
                        fmt(row.get("wall_s")),
                        fmt((row.get("io_active_ratio") or 0.0) * 100, 1, "%"),
                        fmt(row.get("disk_read_gib")),
                        fmt(row.get("prompt_tps")),
                        fmt(row.get("generation_tps")),
                        fmt((row.get("prefill_io_active_ratio") or 0.0) * 100, 1, "%"),
                        fmt((row.get("decode_io_active_ratio") or 0.0) * 100, 1, "%"),
                    ]
                )
                + " |"
            )
        lines.append("")
        lines.extend(
            [
                "## Phase Detail",
                "",
                "| case | prompt | repeat | prefill s | prefill disk GiB | prefill CPU | decode s | decode disk GiB | decode CPU | trace rounds |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in infer_rows:
            prefill_wall = row.get("trace_prefill_wall_s", row.get("prefill_wall_s"))
            prefill_disk = row.get("trace_prefill_disk_read_gib", row.get("prefill_disk_read_gib"))
            prefill_cpu = row.get("trace_prefill_avg_cpu_cores", row.get("prefill_avg_cpu_cores"))
            decode_wall = row.get("trace_decode_wall_s", row.get("decode_wall_s"))
            decode_disk = row.get("trace_decode_disk_read_gib", row.get("decode_disk_read_gib"))
            decode_cpu = row.get("trace_decode_avg_cpu_cores", row.get("decode_avg_cpu_cores"))
            lines.append(
                "| "
                + " | ".join(
                    [
                        row["case"],
                        row["prompt_case"],
                        fmt(row.get("repeat"), 0),
                        fmt(prefill_wall),
                        fmt(prefill_disk),
                        fmt(prefill_cpu),
                        fmt(decode_wall),
                        fmt(decode_disk),
                        fmt(decode_cpu),
                        fmt(row.get("trace_rounds"), 0),
                    ]
                )
                + " |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def selected_cases(all_cases, names):
    if names == ["all"]:
        return list(all_cases)
    by_name = {case["name"]: case for case in all_cases}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise ValueError(f"unknown case(s): {', '.join(missing)}")
    return [by_name[name] for name in names]


def main():
    parser = argparse.ArgumentParser(description="Run a controlled DeepSeek Q4 condition matrix")
    parser.add_argument("--mode", choices=("prewarm", "infer", "both"), default="both")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--hotset", type=Path, default=DEFAULT_HOTSET)
    parser.add_argument("--prewarm-cases", type=lambda s: parse_csv(s), default=["all"])
    parser.add_argument("--infer-cases", type=lambda s: parse_csv(s), default=["best_gap4_chunk1_madvise_on"])
    parser.add_argument("--prompts", type=lambda s: parse_csv(s, PROMPTS), default=["json"])
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--prewarm-budget-gib", type=float, default=8.5)
    parser.add_argument("--sample-interval", type=float, default=0.5)
    parser.add_argument("--trace", action="store_true", help="Enable LLAMA_EXPERT_TRACE per inference run")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or DEFAULT_LOG_ROOT / f"deepseek_q4_perf_matrix_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    prewarm_rows = []
    infer_rows = []

    if args.mode in ("prewarm", "both"):
        cases = selected_cases(PREWARM_QUICK, args.prewarm_cases)
        for case in cases:
            cmd = [
                str(ROOT / "scripts" / "warm_deepseek_q4_expert_cache.py"),
                "--hotset-json",
                str(args.hotset),
                "--budget-gib",
                str(args.prewarm_budget_gib),
                "--experts-per-layer",
                "16",
                "--merge-gap-mib",
                str(case["merge_gap_mib"]),
                "--chunk-mib",
                str(case["chunk_mib"]),
                "--progress-mib",
                "4096",
            ]
            row = {
                "kind": "prewarm",
                "case": case["name"],
                **case,
                "command": cmd,
            }
            print(f"[prewarm] {case['name']}", flush=True)
            if not args.dry_run:
                returncode, elapsed_s, output = run_capture(cmd)
                (out_dir / f"prewarm_{case['name']}.log").write_text(output, encoding="utf-8")
                row.update({"returncode": returncode, "elapsed_s": elapsed_s, **parse_prewarm_output(output)})
                if returncode != 0:
                    print(output[-4000:], file=sys.stderr)
                    raise SystemExit(returncode)
            prewarm_rows.append(row)

    if args.mode in ("infer", "both"):
        cases = selected_cases(INFER_QUICK, args.infer_cases)
        for case, prompt_case, repeat in itertools.product(cases, args.prompts, range(args.repeats)):
            prompt = PROMPTS[prompt_case]
            repeat_suffix = f"_r{repeat + 1:02d}" if args.repeats > 1 else ""
            profile_path = out_dir / f"profile_{case['name']}_{prompt_case}{repeat_suffix}.json"
            trace_path = out_dir / f"trace_{case['name']}_{prompt_case}{repeat_suffix}.jsonl"
            cmd = [
                str(ROOT / "scripts" / "profile_deepseek_q4_run.py"),
                "--prompt",
                prompt,
                "--tokens",
                str(args.tokens),
                "--context",
                str(args.context),
                "--sample-interval",
                str(args.sample_interval),
                "--prewarm-budget-gib",
                str(args.prewarm_budget_gib),
                "--out",
                str(profile_path),
            ]
            cmd.append("--prewarm" if case["prewarm"] else "--no-prewarm")
            cmd.extend(
                [
                    "--env",
                    f"PREWARM_HOTSET_JSON={args.hotset}",
                    "--env",
                    f"PREWARM_MERGE_GAP_MIB={case['merge_gap_mib']}",
                    "--env",
                    f"PREWARM_CHUNK_MIB={case['chunk_mib']}",
                    "--env",
                    f"GGML_MOE_MADVISE_WILLNEED={'1' if case['madvise'] else '0'}",
                ]
            )
            if args.trace:
                cmd.extend(["--env", f"LLAMA_EXPERT_TRACE={trace_path}"])

            row = {
                "kind": "infer",
                "case": case["name"],
                "prompt_case": prompt_case,
                "repeat": repeat + 1,
                "prompt": prompt,
                **case,
                "tokens": args.tokens,
                "context": args.context,
                "command": cmd,
            }
            print(f"[infer] {case['name']} / {prompt_case} / repeat {repeat + 1}", flush=True)
            if not args.dry_run:
                returncode, elapsed_s, output = run_capture(cmd)
                (out_dir / f"profile_{case['name']}_{prompt_case}{repeat_suffix}.log").write_text(
                    output,
                    encoding="utf-8",
                )
                row.update({"profile_runner_returncode": returncode, "profile_runner_elapsed_s": elapsed_s})
                if profile_path.exists():
                    row.update(profile_row_from_json(profile_path))
                if returncode != 0:
                    print(output[-4000:], file=sys.stderr)
                    raise SystemExit(returncode)
            infer_rows.append(row)

    write_jsonl(out_dir / "prewarm_matrix.jsonl", prewarm_rows)
    write_jsonl(out_dir / "inference_matrix.jsonl", infer_rows)
    write_markdown(out_dir / "summary.md", prewarm_rows, infer_rows)
    print(f"matrix_dir: {out_dir}")
    print(f"summary: {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
