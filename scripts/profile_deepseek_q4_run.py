#!/usr/bin/env python3
import argparse
import ctypes
import json
import os
import re
import selectors
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXP_ROOT = ROOT.parent / "deepseek-v4-experiments"
DEFAULT_WRAPPER = ROOT / "scripts" / "run_deepseek_q4_gguf_demo.sh"
DEFAULT_HOTSET = ROOT / "data" / "deepseek_q4_probe_hotset_k16.json"
DEFAULT_OUT = ROOT / "logs" / "deepseek_q4_profile_latest.json"
DEFAULT_MODEL = EXP_ROOT / "models" / "lovedheart-deepseek-v4-flash-gguf" / "DeepSeek-V4-Flash-MXFP4_MOE.gguf"
DEFAULT_TOKENIZER = EXP_ROOT / "llama.cpp-deepseek-v4-flash" / "build" / "bin" / "llama-tokenize"


RUSAGE_INFO_V4 = 4


class RUsageInfoV4(ctypes.Structure):
    _fields_ = [
        ("ri_uuid", ctypes.c_uint8 * 16),
        ("ri_user_time", ctypes.c_uint64),
        ("ri_system_time", ctypes.c_uint64),
        ("ri_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_interrupt_wkups", ctypes.c_uint64),
        ("ri_pageins", ctypes.c_uint64),
        ("ri_wired_size", ctypes.c_uint64),
        ("ri_resident_size", ctypes.c_uint64),
        ("ri_phys_footprint", ctypes.c_uint64),
        ("ri_proc_start_abstime", ctypes.c_uint64),
        ("ri_proc_exit_abstime", ctypes.c_uint64),
        ("ri_child_user_time", ctypes.c_uint64),
        ("ri_child_system_time", ctypes.c_uint64),
        ("ri_child_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_child_interrupt_wkups", ctypes.c_uint64),
        ("ri_child_pageins", ctypes.c_uint64),
        ("ri_child_elapsed_abstime", ctypes.c_uint64),
        ("ri_diskio_bytesread", ctypes.c_uint64),
        ("ri_diskio_byteswritten", ctypes.c_uint64),
        ("ri_cpu_time_qos_default", ctypes.c_uint64),
        ("ri_cpu_time_qos_maintenance", ctypes.c_uint64),
        ("ri_cpu_time_qos_background", ctypes.c_uint64),
        ("ri_cpu_time_qos_utility", ctypes.c_uint64),
        ("ri_cpu_time_qos_legacy", ctypes.c_uint64),
        ("ri_cpu_time_qos_user_initiated", ctypes.c_uint64),
        ("ri_cpu_time_qos_user_interactive", ctypes.c_uint64),
        ("ri_billed_system_time", ctypes.c_uint64),
        ("ri_serviced_system_time", ctypes.c_uint64),
        ("ri_logical_writes", ctypes.c_uint64),
        ("ri_lifetime_max_phys_footprint", ctypes.c_uint64),
        ("ri_instructions", ctypes.c_uint64),
        ("ri_cycles", ctypes.c_uint64),
        ("ri_billed_energy", ctypes.c_uint64),
        ("ri_serviced_energy", ctypes.c_uint64),
        ("ri_interval_max_phys_footprint", ctypes.c_uint64),
        ("ri_runnable_time", ctypes.c_uint64),
    ]


def proc_pid_rusage(pid):
    if sys.platform != "darwin":
        return None
    libproc = proc_pid_rusage.libproc
    info = RUsageInfoV4()
    ret = libproc.proc_pid_rusage(int(pid), RUSAGE_INFO_V4, ctypes.byref(info))
    if ret != 0:
        return None
    numer, denom = proc_pid_rusage.timebase

    def ticks_to_ns(value):
        return int(int(value) * numer // denom)

    return {
        "user_ns": ticks_to_ns(info.ri_user_time),
        "system_ns": ticks_to_ns(info.ri_system_time),
        "pageins": int(info.ri_pageins),
        "disk_read_bytes": int(info.ri_diskio_bytesread),
        "disk_write_bytes": int(info.ri_diskio_byteswritten),
        "resident_bytes": int(info.ri_resident_size),
        "phys_footprint_bytes": int(info.ri_phys_footprint),
        "instructions": int(info.ri_instructions),
        "cycles": int(info.ri_cycles),
        "runnable_ns": ticks_to_ns(info.ri_runnable_time),
    }


proc_pid_rusage.libproc = ctypes.CDLL("/usr/lib/libproc.dylib") if sys.platform == "darwin" else None
proc_pid_rusage.timebase = (1, 1)
if proc_pid_rusage.libproc is not None:
    proc_pid_rusage.libproc.proc_pid_rusage.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(RUsageInfoV4),
    ]
    proc_pid_rusage.libproc.proc_pid_rusage.restype = ctypes.c_int
    class MachTimebaseInfo(ctypes.Structure):
        _fields_ = [("numer", ctypes.c_uint32), ("denom", ctypes.c_uint32)]
    libsystem = ctypes.CDLL("/usr/lib/libSystem.dylib")
    info = MachTimebaseInfo()
    if libsystem.mach_timebase_info(ctypes.byref(info)) == 0 and info.denom:
        proc_pid_rusage.timebase = (int(info.numer), int(info.denom))


def process_table():
    out = subprocess.run(
        ["ps", "-axo", "pid=,ppid="],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    ).stdout
    children = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        pid, ppid = map(int, parts)
        children.setdefault(ppid, []).append(pid)
    return children


def descendants(root_pid):
    children = process_table()
    result = {int(root_pid)}
    stack = [int(root_pid)]
    while stack:
        pid = stack.pop()
        for child in children.get(pid, []):
            if child not in result:
                result.add(child)
                stack.append(child)
    return sorted(result)


def sample_process_tree(root_pid):
    rows = []
    for pid in descendants(root_pid):
        data = proc_pid_rusage(pid)
        if data is not None:
            data["pid"] = pid
            rows.append(data)
    return rows


def metrics_delta(prev, cur):
    if prev is None:
        return {
            "user_ns": 0,
            "system_ns": 0,
            "pageins": 0,
            "disk_read_bytes": 0,
            "disk_write_bytes": 0,
            "instructions": 0,
            "cycles": 0,
            "runnable_ns": 0,
        }
    delta = {}
    for key in (
        "user_ns",
        "system_ns",
        "pageins",
        "disk_read_bytes",
        "disk_write_bytes",
        "instructions",
        "cycles",
        "runnable_ns",
    ):
        delta[key] = max(0, cur.get(key, 0) - prev.get(key, 0))
    return delta


def parse_child_output(text):
    parsed = {}
    match = re.search(r"done: warmed ([0-9.]+) GiB in ([0-9.]+)s \(([0-9.]+) GiB/s\)", text)
    if match:
        parsed["prewarm_gib"] = float(match.group(1))
        parsed["prewarm_s"] = float(match.group(2))
        parsed["prewarm_gib_s"] = float(match.group(3))
    match = re.search(
        r"pread ranges: raw=(\d+), unique=(\d+), merged=(\d+), merge_gap=([0-9.]+) MiB",
        text,
    )
    if match:
        parsed["prewarm_raw_ranges"] = int(match.group(1))
        parsed["prewarm_unique_ranges"] = int(match.group(2))
        parsed["prewarm_merged_ranges"] = int(match.group(3))
        parsed["prewarm_merge_gap_mib"] = float(match.group(4))
    match = re.search(
        r"pread bytes: planned=([0-9.]+) GiB, unique=([0-9.]+) GiB, actual=([0-9.]+) GiB, "
        r"overread=([0-9.]+) GiB \(([0-9.]+)%\)",
        text,
    )
    if match:
        parsed["prewarm_planned_gib"] = float(match.group(1))
        parsed["prewarm_unique_gib"] = float(match.group(2))
        parsed["prewarm_actual_gib"] = float(match.group(3))
        parsed["prewarm_overread_gib"] = float(match.group(4))
        parsed["prewarm_overread_ratio"] = float(match.group(5)) / 100.0
    match = re.search(r"\[ Prompt:\s*([0-9.]+) t/s \| Generation:\s*([0-9.]+) t/s \]", text)
    if match:
        parsed["prompt_tps"] = float(match.group(1))
        parsed["generation_tps"] = float(match.group(2))
    return parsed


def tokenize_text(tokenizer, model_path, text):
    if not text or not tokenizer.exists() or not model_path.exists():
        return None
    result = subprocess.run(
        [str(tokenizer), "-m", str(model_path), "-p", text, "--no-bos"],
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != 0:
        return None
    tokens = []
    for line in result.stdout.splitlines():
        match = re.match(r"\s*(\d+)\s+->", line)
        if match:
            tokens.append(int(match.group(1)))
    return tokens or None


def extract_assistant_text(child_output, prompt):
    marker = "> " + prompt
    pos = child_output.rfind(marker)
    if pos < 0:
        return ""
    text = child_output[pos + len(marker):]
    text = re.sub(r"\n?\[ Prompt:.*", "", text, flags=re.DOTALL)
    return text.strip()


def interval_delta_for_window(samples, start_s, end_s, io_threshold_bytes):
    if len(samples) < 2 or end_s <= start_s:
        return {
            "wall_s": 0.0,
            "io_active_wall_s": 0.0,
            "io_active_ratio": 0.0,
            "cpu_core_s": 0.0,
            "avg_cpu_cores": 0.0,
            "pageins": 0,
            "disk_read_gib": 0.0,
        }

    wall = 0.0
    io_wall = 0.0
    cpu_core_s = 0.0
    pageins = 0
    disk_read = 0
    for prev, cur in zip(samples, samples[1:]):
        a = max(prev["t_s"], start_s)
        b = min(cur["t_s"], end_s)
        if b <= a:
            continue
        sample_dt = max(cur["t_s"] - prev["t_s"], 1e-9)
        frac = (b - a) / sample_dt
        delta = cur["delta"]
        wall += b - a
        if delta.get("disk_read_bytes", 0) >= io_threshold_bytes or delta.get("pageins", 0) > 0:
            io_wall += b - a
        cpu_core_s += (delta.get("user_ns", 0) + delta.get("system_ns", 0)) / 1e9 * frac
        pageins += int(delta.get("pageins", 0) * frac)
        disk_read += int(delta.get("disk_read_bytes", 0) * frac)

    return {
        "wall_s": wall,
        "io_active_wall_s": io_wall,
        "io_active_ratio": io_wall / wall if wall > 0 else 0.0,
        "cpu_core_s": cpu_core_s,
        "avg_cpu_cores": cpu_core_s / wall if wall > 0 else 0.0,
        "pageins": pageins,
        "disk_read_gib": disk_read / 1024**3,
    }


def load_trace_windows(trace_path, timing_seen_s=None, elapsed_s=None):
    if not trace_path:
        return None
    path = Path(trace_path)
    if not path.exists():
        return None
    events = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("kind") == "gate":
                events.append(event)
    if not events:
        return None

    rounds = []
    current = []
    last_layer = -1
    for event in events:
        layer = int(event["layer"])
        if current and layer <= last_layer:
            rounds.append(current)
            current = []
        current.append(event)
        last_layer = layer
    if current:
        rounds.append(current)

    max_trace_s = max(float(event["t_us"]) / 1e6 for event in events)
    anchor_s = timing_seen_s if timing_seen_s is not None else elapsed_s
    offset_s = (anchor_s - max_trace_s) if anchor_s is not None else 0.0

    phase_ranges = {}
    round_summaries = []
    for index, round_events in enumerate(rounds):
        phase = "prefill" if int(round_events[0].get("n_tokens", 0)) > 1 else "decode"
        start = min(float(event["t_us"]) / 1e6 for event in round_events) + offset_s
        end = max(float(event["t_us"]) / 1e6 for event in round_events) + offset_s
        phase_ranges.setdefault(phase, [start, end])
        phase_ranges[phase][0] = min(phase_ranges[phase][0], start)
        phase_ranges[phase][1] = max(phase_ranges[phase][1], end)
        round_summaries.append(
            {
                "index": index,
                "phase": phase,
                "n_tokens": int(round_events[0].get("n_tokens", 0)),
                "layers": len(round_events),
                "profile_window_s": [start, end],
                "duration_s": max(0.0, end - start),
            }
        )

    return {
        "trace": str(path),
        "events": len(events),
        "rounds": len(rounds),
        "trace_to_profile_offset_s": offset_s,
        "phase_windows_s": phase_ranges,
        "rounds_summary": round_summaries,
    }


def summarize(samples, io_threshold_bytes):
    if len(samples) < 2:
        return {}

    total_wall = samples[-1]["t_s"] - samples[0]["t_s"]
    totals = {
        "user_ns": 0,
        "system_ns": 0,
        "pageins": 0,
        "disk_read_bytes": 0,
        "disk_write_bytes": 0,
        "instructions": 0,
        "cycles": 0,
        "runnable_ns": 0,
    }
    io_active_wall = 0.0
    peak_rss = 0
    peak_footprint = 0

    for prev, cur in zip(samples, samples[1:]):
        dt = cur["t_s"] - prev["t_s"]
        delta = cur["delta"]
        for key in totals:
            totals[key] += delta.get(key, 0)
        if delta.get("disk_read_bytes", 0) >= io_threshold_bytes or delta.get("pageins", 0) > 0:
            io_active_wall += max(0.0, dt)
        peak_rss = max(peak_rss, cur.get("resident_bytes", 0))
        peak_footprint = max(peak_footprint, cur.get("phys_footprint_bytes", 0))

    cpu_core_s = (totals["user_ns"] + totals["system_ns"]) / 1e9
    return {
        "wall_s": total_wall,
        "io_active_wall_s": io_active_wall,
        "io_active_ratio": io_active_wall / total_wall if total_wall > 0 else 0.0,
        "compute_or_other_wall_s": max(0.0, total_wall - io_active_wall),
        "compute_or_other_ratio": max(0.0, total_wall - io_active_wall) / total_wall if total_wall > 0 else 0.0,
        "cpu_core_s": cpu_core_s,
        "avg_cpu_cores": cpu_core_s / total_wall if total_wall > 0 else 0.0,
        "pageins": totals["pageins"],
        "disk_read_gib": totals["disk_read_bytes"] / 1024**3,
        "disk_write_gib": totals["disk_write_bytes"] / 1024**3,
        "peak_rss_gib": peak_rss / 1024**3,
        "peak_phys_footprint_gib": peak_footprint / 1024**3,
        "instructions": totals["instructions"],
        "cycles": totals["cycles"],
        "io_threshold_bytes_per_sample": io_threshold_bytes,
    }


def build_env(args):
    env = os.environ.copy()
    env.update(
        {
            "PROMPT": args.prompt,
            "TOKENS": str(args.tokens),
            "CONTEXT": str(args.context),
            "STDERR": "1",
            "MODEL_PATH": str(args.model_path),
        }
    )
    if args.prewarm:
        env["PREWARM_EXPERTS"] = "1"
        env["PREWARM_BUDGET_GIB"] = str(args.prewarm_budget_gib)
        env["PREWARM_HOTSET_JSON"] = str(args.hotset)
    if args.no_prewarm:
        env.pop("PREWARM_EXPERTS", None)
    for item in args.env or []:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"--env expects KEY=VALUE, got {item!r}")
        env[key] = value
    return env


def main():
    parser = argparse.ArgumentParser(description="Profile DeepSeek Q4 run load-vs-compute proxy metrics")
    parser.add_argument("--wrapper", type=Path, default=DEFAULT_WRAPPER)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer-bin", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--prompt", default='Return JSON only: {"status":"ok","note":"profile"}')
    parser.add_argument("--tokens", type=int, default=12)
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--sample-interval", type=float, default=0.5)
    parser.add_argument("--io-threshold-mib", type=float, default=4.0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--prewarm", action="store_true")
    parser.add_argument("--no-prewarm", action="store_true")
    parser.add_argument("--hotset", type=Path, default=DEFAULT_HOTSET)
    parser.add_argument("--prewarm-budget-gib", type=float, default=8.5)
    parser.add_argument("--echo-child", action="store_true")
    parser.add_argument("--env", action="append", help="Extra environment assignment KEY=VALUE")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.prewarm and args.no_prewarm:
        raise ValueError("choose only one of --prewarm and --no-prewarm")
    if not args.wrapper.exists():
        raise FileNotFoundError(args.wrapper)

    cmd = [str(args.wrapper), *args.extra_args]
    env = build_env(args)
    prompt_tokens = tokenize_text(args.tokenizer_bin, args.model_path, args.prompt)

    started = time.monotonic()
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    if proc.stdout is None:
        raise RuntimeError("stdout pipe missing")

    os.set_blocking(proc.stdout.fileno(), False)
    sel = selectors.DefaultSelector()
    sel.register(proc.stdout, selectors.EVENT_READ)

    samples = []
    prev_by_pid = {}
    output_parts = []
    event_times = {}
    next_sample = started

    def add_sample():
        now = time.monotonic()
        rows = sample_process_tree(proc.pid)
        delta_total = {
            "user_ns": 0,
            "system_ns": 0,
            "pageins": 0,
            "disk_read_bytes": 0,
            "disk_write_bytes": 0,
            "instructions": 0,
            "cycles": 0,
            "runnable_ns": 0,
        }
        resident = 0
        footprint = 0
        for row in rows:
            pid = row["pid"]
            delta = metrics_delta(prev_by_pid.get(pid), row)
            prev_by_pid[pid] = row
            for key in delta_total:
                delta_total[key] += delta[key]
            resident += row.get("resident_bytes", 0)
            footprint += row.get("phys_footprint_bytes", 0)
        samples.append(
            {
                "t_s": now - started,
                "pids": [row["pid"] for row in rows],
                "resident_bytes": resident,
                "phys_footprint_bytes": footprint,
                "delta": delta_total,
            }
        )

    add_sample()
    while True:
        now = time.monotonic()
        timeout = max(0.0, min(args.sample_interval, next_sample - now))
        for key, _ in sel.select(timeout):
            try:
                chunk = os.read(key.fileobj.fileno(), 65536)
            except BlockingIOError:
                chunk = b""
            if chunk:
                output_parts.append(chunk)
                if args.echo_child:
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.buffer.flush()
                text_so_far = b"".join(output_parts).decode("utf-8", errors="replace")
                for name, pattern in (
                    ("prewarm_done", "done: warmed"),
                    ("llama_loading", "Loading model"),
                    ("prompt_displayed", "\n> "),
                    ("timing_seen", "[ Prompt:"),
                ):
                    if name not in event_times and pattern in text_so_far:
                        event_times[name] = time.monotonic() - started

        now = time.monotonic()
        if now >= next_sample:
            add_sample()
            while next_sample <= now:
                next_sample += args.sample_interval

        if proc.poll() is not None:
            while True:
                try:
                    chunk = os.read(proc.stdout.fileno(), 65536)
                except BlockingIOError:
                    break
                if not chunk:
                    break
                output_parts.append(chunk)
                if args.echo_child:
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.buffer.flush()
            add_sample()
            break

    ended = time.monotonic()
    child_output = b"".join(output_parts).decode("utf-8", errors="replace")
    parsed_output = parse_child_output(child_output)
    assistant_text = extract_assistant_text(child_output, args.prompt)
    generated_tokens = tokenize_text(args.tokenizer_bin, args.model_path, assistant_text)
    summary = summarize(samples, int(args.io_threshold_mib * 1024 * 1024))
    summary.update(parsed_output)
    summary["returncode"] = proc.returncode
    summary["elapsed_s"] = ended - started
    summary["event_times_s"] = event_times

    token_summary = {
        "prompt_tokens": len(prompt_tokens) if prompt_tokens is not None else None,
        "generated_tokens": len(generated_tokens) if generated_tokens is not None else None,
        "generated_text": assistant_text,
    }
    if (
        token_summary["prompt_tokens"]
        and token_summary["generated_tokens"]
        and parsed_output.get("prompt_tps")
        and parsed_output.get("generation_tps")
    ):
        prefill_s = token_summary["prompt_tokens"] / parsed_output["prompt_tps"]
        decode_s = token_summary["generated_tokens"] / parsed_output["generation_tps"]
        timing_end = event_times.get("timing_seen", summary["elapsed_s"])
        decode_start = max(0.0, timing_end - decode_s)
        prefill_start = max(0.0, decode_start - prefill_s)
        token_summary.update(
            {
                "prefill_s_est": prefill_s,
                "decode_s_est": decode_s,
                "prefill_window_s": [prefill_start, decode_start],
                "decode_window_s": [decode_start, timing_end],
                "prefill_profile": interval_delta_for_window(
                    samples,
                    prefill_start,
                    decode_start,
                    int(args.io_threshold_mib * 1024 * 1024),
                ),
                "decode_profile": interval_delta_for_window(
                    samples,
                    decode_start,
                    timing_end,
                    int(args.io_threshold_mib * 1024 * 1024),
                ),
            }
        )
    summary["token_timing"] = token_summary

    trace_path = env.get("LLAMA_EXPERT_TRACE")
    trace_windows = load_trace_windows(
        trace_path,
        timing_seen_s=event_times.get("timing_seen"),
        elapsed_s=summary["elapsed_s"],
    )
    if trace_windows is not None:
        for phase, (start_s, end_s) in trace_windows["phase_windows_s"].items():
            trace_windows[f"{phase}_profile"] = interval_delta_for_window(
                samples,
                start_s,
                end_s,
                int(args.io_threshold_mib * 1024 * 1024),
            )
        summary["trace_timing"] = trace_windows

    payload = {
        "format": "deepseek-q4-profile-v1",
        "command": cmd,
        "env": {
            key: env.get(key)
            for key in (
                "PROMPT",
                "TOKENS",
                "CONTEXT",
                "PREWARM_EXPERTS",
                "PREWARM_BUDGET_GIB",
                "PREWARM_HOTSET_JSON",
                "PREWARM_MERGE_GAP_MIB",
                "PREWARM_CHUNK_MIB",
                "GGML_DISABLE_CPU_REPACK",
                "GGML_MOE_MADVISE_WILLNEED",
                "LLAMA_MMAP_RANDOM",
            )
            if env.get(key) is not None
        },
        "summary": summary,
        "samples": samples,
        "child_output_tail": child_output[-8000:],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"profile: {args.out}")
    print(f"returncode: {proc.returncode}")
    print(f"wall: {summary.get('wall_s', 0.0):.2f}s")
    print(
        f"io-active proxy: {summary.get('io_active_wall_s', 0.0):.2f}s "
        f"({summary.get('io_active_ratio', 0.0) * 100:.1f}%)"
    )
    print(
        f"compute/other proxy: {summary.get('compute_or_other_wall_s', 0.0):.2f}s "
        f"({summary.get('compute_or_other_ratio', 0.0) * 100:.1f}%)"
    )
    print(f"avg CPU cores: {summary.get('avg_cpu_cores', 0.0):.2f}")
    print(f"pageins: {summary.get('pageins', 0)}")
    print(f"disk read: {summary.get('disk_read_gib', 0.0):.2f} GiB")
    print(f"peak RSS: {summary.get('peak_rss_gib', 0.0):.2f} GiB")
    if "prewarm_s" in summary:
        print(
            f"prewarm load: {summary['prewarm_gib']:.2f} GiB in {summary['prewarm_s']:.2f}s "
            f"({summary['prewarm_gib_s']:.2f} GiB/s)"
        )
    if "prompt_tps" in summary:
        print(f"llama timing: prompt {summary['prompt_tps']:.2f} t/s, generation {summary['generation_tps']:.2f} t/s")
    tt = summary.get("token_timing", {})
    if tt.get("prefill_s_est") is not None:
        print(
            f"prefill: {tt['prompt_tokens']} tok, {tt['prefill_s_est']:.2f}s est, "
            f"I/O-active {tt['prefill_profile']['io_active_ratio'] * 100:.1f}%"
        )
        print(
            f"decode: {tt['generated_tokens']} tok, {tt['decode_s_est']:.2f}s est, "
            f"I/O-active {tt['decode_profile']['io_active_ratio'] * 100:.1f}%"
        )
    trace_timing = summary.get("trace_timing")
    if trace_timing:
        for phase in ("prefill", "decode"):
            prof = trace_timing.get(f"{phase}_profile")
            window = trace_timing.get("phase_windows_s", {}).get(phase)
            if prof and window:
                print(
                    f"trace {phase}: {window[1] - window[0]:.2f}s, "
                    f"I/O-active {prof['io_active_ratio'] * 100:.1f}%, "
                    f"avg CPU cores {prof['avg_cpu_cores']:.2f}"
                )

    if proc.returncode != 0:
        sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
