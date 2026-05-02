#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_ROOT = ROOT / "logs"
DEFAULT_HOTSET = ROOT / "data" / "deepseek_q4_probe_hotset_k16.json"

sys.path.insert(0, str(ROOT / "scripts"))
import run_deepseek_q4_perf_matrix as matrix  # noqa: E402


SUITE_PROMPTS = {
    "prefill": ("prefill_long_plain", "prefill_long_code"),
    "decode": ("decode_json_seed", "decode_plain_seed"),
    "mixed": ("json", "plain", "code", "korean"),
}

SUITE_TOKENS = {
    "prefill": 2,
    "decode": 24,
    "mixed": 8,
}

DEFAULT_CASES = (
    "no_prewarm_madvise_off",
    "best_gap4_chunk1_madvise_on",
    "gap4_chunk1_madvise_off",
    "no_prewarm_madvise_on",
)


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_csv(value, allowed=None):
    items = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        if allowed and part not in allowed:
            raise argparse.ArgumentTypeError(f"unknown value {part!r}; choose from {', '.join(sorted(allowed))}")
        items.append(part)
    if not items:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    return items


def selected_cases(names):
    by_name = {case["name"]: case for case in matrix.INFER_QUICK}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise ValueError(f"unknown case(s): {', '.join(missing)}")
    return [by_name[name] for name in names]


def append_jsonl(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def existing_run_count(path):
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def update_latest_symlink(out_dir):
    latest = DEFAULT_LOG_ROOT / "deepseek_q4_longrun_latest"
    try:
        if latest.is_symlink():
            latest.unlink()
        if not latest.exists():
            latest.symlink_to(out_dir, target_is_directory=True)
    except OSError:
        pass


def run_profile(spec, args, out_dir, run_index):
    case = spec["case"]
    suite = spec["suite"]
    prompt_case = spec["prompt_case"]
    prompt = matrix.PROMPTS[prompt_case]
    tokens = spec["tokens"]

    stem = f"run_{run_index:04d}_{suite}_{case['name']}_{prompt_case}"
    profile_path = out_dir / "profiles" / f"{stem}.json"
    trace_path = out_dir / "traces" / f"{stem}.jsonl"
    log_path = out_dir / "logs" / f"{stem}.log"
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(ROOT / "scripts" / "profile_deepseek_q4_run.py"),
        "--prompt",
        prompt,
        "--tokens",
        str(tokens),
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

    started_at = utc_now()
    started = time.monotonic()
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    elapsed = time.monotonic() - started
    ended_at = utc_now()
    log_path.write_text(proc.stdout, encoding="utf-8")

    row = {
        "run_index": run_index,
        "suite": suite,
        "case": case["name"],
        "prompt_case": prompt_case,
        "prompt": prompt,
        "tokens": tokens,
        "context": args.context,
        "prewarm": case["prewarm"],
        "merge_gap_mib": case["merge_gap_mib"],
        "chunk_mib": case["chunk_mib"],
        "madvise": case["madvise"],
        "trace": args.trace,
        "started_at": started_at,
        "ended_at": ended_at,
        "runner_elapsed_s": elapsed,
        "profile_runner_returncode": proc.returncode,
        "profile": str(profile_path),
        "trace_path": str(trace_path) if args.trace else None,
        "log": str(log_path),
        "command": cmd,
    }
    if profile_path.exists():
        row.update(matrix.profile_row_from_json(profile_path))
    else:
        row["returncode"] = proc.returncode
        row["error_tail"] = proc.stdout[-4000:]
    return row


def write_schedule(out_dir, args, specs):
    payload = {
        "format": "deepseek-q4-longrun-schedule-v1",
        "created_at": utc_now(),
        "duration_hours": args.duration_hours,
        "max_runs": args.max_runs,
        "sleep_s": args.sleep_s,
        "context": args.context,
        "sample_interval": args.sample_interval,
        "trace": args.trace,
        "hotset": str(args.hotset),
        "specs": [
            {
                "suite": spec["suite"],
                "case": spec["case"]["name"],
                "prompt_case": spec["prompt_case"],
                "tokens": spec["tokens"],
            }
            for spec in specs
        ],
    }
    (out_dir / "schedule.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def summarize(out_dir):
    cmd = [
        str(ROOT / "scripts" / "summarize_deepseek_q4_perf_dataset.py"),
        "--dataset",
        str(out_dir / "runs.jsonl"),
        "--out-dir",
        str(out_dir),
    ]
    return subprocess.run(cmd, cwd=str(ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)


def main():
    parser = argparse.ArgumentParser(description="Collect DeepSeek Q4 performance data for a bounded long run")
    parser.add_argument("--duration-hours", type=float, default=5.0)
    parser.add_argument("--max-runs", type=int, default=0, help="0 means run until duration is reached")
    parser.add_argument("--sleep-s", type=float, default=30.0)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--suites", type=lambda s: parse_csv(s, SUITE_PROMPTS), default=["prefill", "decode"])
    parser.add_argument("--cases", type=lambda s: parse_csv(s), default=list(DEFAULT_CASES))
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--sample-interval", type=float, default=0.5)
    parser.add_argument("--prewarm-budget-gib", type=float, default=8.5)
    parser.add_argument("--hotset", type=Path, default=DEFAULT_HOTSET)
    parser.add_argument("--no-trace", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.trace = not args.no_trace

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or DEFAULT_LOG_ROOT / f"deepseek_q4_longrun_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    update_latest_symlink(out_dir)

    cases = selected_cases(args.cases)
    specs = []
    for suite in args.suites:
        for case in cases:
            for prompt_case in SUITE_PROMPTS[suite]:
                specs.append(
                    {
                        "suite": suite,
                        "case": case,
                        "prompt_case": prompt_case,
                        "tokens": SUITE_TOKENS[suite],
                    }
                )
    write_schedule(out_dir, args, specs)

    if args.dry_run:
        print(f"out_dir: {out_dir}")
        print(f"specs: {len(specs)}")
        for index, spec in enumerate(specs, start=1):
            print(
                f"[{index}] {spec['suite']} / {spec['case']['name']} / "
                f"{spec['prompt_case']} / tokens={spec['tokens']}"
            )
        print(f"schedule: {out_dir / 'schedule.json'}")
        return

    dataset = out_dir / "runs.jsonl"
    run_index = existing_run_count(dataset) + 1
    started = time.monotonic()
    deadline = started + args.duration_hours * 3600.0
    max_runs = args.max_runs if args.max_runs > 0 else None

    print(f"out_dir: {out_dir}", flush=True)
    print(f"specs: {len(specs)}", flush=True)
    print(f"duration_hours: {args.duration_hours}", flush=True)
    while time.monotonic() < deadline and (max_runs is None or run_index <= max_runs):
        spec = specs[(run_index - 1) % len(specs)]
        print(
            f"[{run_index}] {spec['suite']} / {spec['case']['name']} / "
            f"{spec['prompt_case']} / tokens={spec['tokens']}",
            flush=True,
        )
        row = run_profile(spec, args, out_dir, run_index)
        append_jsonl(dataset, row)
        summary_proc = summarize(out_dir)
        (out_dir / "summarize.log").write_text(summary_proc.stdout, encoding="utf-8")
        if row.get("returncode", row.get("profile_runner_returncode")) not in (0, None):
            print(f"[{run_index}] failed; see {row.get('log')}", flush=True)
        else:
            print(
                f"[{run_index}] wall={row.get('wall_s', 0):.2f}s "
                f"gen={row.get('generation_tps', 0):.2f}t/s "
                f"disk={row.get('disk_read_gib', 0):.2f}GiB",
                flush=True,
            )
        run_index += 1
        if time.monotonic() < deadline and (max_runs is None or run_index <= max_runs):
            time.sleep(args.sleep_s)

    summarize(out_dir)
    print(f"done: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
