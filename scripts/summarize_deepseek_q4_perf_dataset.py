#!/usr/bin/env python3
import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "logs" / "deepseek_q4_longrun_latest" / "runs.jsonl"


METRICS = (
    "wall_s",
    "disk_read_gib",
    "io_active_ratio",
    "prompt_tps",
    "generation_tps",
    "effective_prefill_wall_s",
    "effective_decode_wall_s",
    "effective_prefill_disk_read_gib",
    "effective_decode_disk_read_gib",
    "effective_prefill_io_active_ratio",
    "effective_decode_io_active_ratio",
    "prewarm_s",
)


def load_rows(path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def number(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def mean(values):
    values = [v for v in values if v is not None and math.isfinite(v)]
    return sum(values) / len(values) if values else None


def stdev(values):
    values = [v for v in values if v is not None and math.isfinite(v)]
    if len(values) < 2:
        return None
    avg = sum(values) / len(values)
    return math.sqrt(sum((v - avg) ** 2 for v in values) / (len(values) - 1))


def fmt(value, digits=2, pct=False):
    if value is None:
        return ""
    if pct:
        return f"{value * 100:.1f}%"
    return f"{value:.{digits}f}"


def effective_rows(rows):
    enriched = []
    for row in rows:
        out = dict(row)
        out["effective_prefill_wall_s"] = (
            row.get("trace_prefill_wall_s")
            if row.get("trace_prefill_wall_s") is not None
            else row.get("prefill_wall_s")
        )
        out["effective_decode_wall_s"] = (
            row.get("trace_decode_wall_s")
            if row.get("trace_decode_wall_s") is not None
            else row.get("decode_wall_s")
        )
        out["effective_prefill_disk_read_gib"] = (
            row.get("trace_prefill_disk_read_gib")
            if row.get("trace_prefill_disk_read_gib") is not None
            else row.get("prefill_disk_read_gib")
        )
        out["effective_decode_disk_read_gib"] = (
            row.get("trace_decode_disk_read_gib")
            if row.get("trace_decode_disk_read_gib") is not None
            else row.get("decode_disk_read_gib")
        )
        out["effective_prefill_io_active_ratio"] = (
            row.get("trace_prefill_io_active_ratio")
            if row.get("trace_prefill_io_active_ratio") is not None
            else row.get("prefill_io_active_ratio")
        )
        out["effective_decode_io_active_ratio"] = (
            row.get("trace_decode_io_active_ratio")
            if row.get("trace_decode_io_active_ratio") is not None
            else row.get("decode_io_active_ratio")
        )
        enriched.append(out)
    return enriched


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        if row.get("returncode") not in (0, None):
            continue
        key = (row.get("suite", ""), row.get("case", ""), str(row.get("prewarm")), str(row.get("madvise")))
        groups[key].append(row)

    summaries = []
    for (suite, case, prewarm, madvise), items in sorted(groups.items()):
        summary = {
            "suite": suite,
            "case": case,
            "prewarm": prewarm,
            "madvise": madvise,
            "n": len(items),
        }
        for metric in METRICS:
            vals = [number(item.get(metric)) for item in items]
            summary[f"{metric}_mean"] = mean(vals)
            summary[f"{metric}_stdev"] = stdev(vals)
        summaries.append(summary)
    return summaries


def write_rows_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary_csv(path, summaries):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["suite", "case", "prewarm", "madvise", "n"]
    for metric in METRICS:
        fieldnames.extend([f"{metric}_mean", f"{metric}_stdev"])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow(row)


def bar_svg(path, title, summaries, metric, ylabel, lower_is_better=False):
    data = [
        (
            f"{row['suite']}\n{row['case'].replace('_', ' ')}",
            row.get(f"{metric}_mean"),
            row.get(f"{metric}_stdev"),
        )
        for row in summaries
        if row.get(f"{metric}_mean") is not None
    ]
    if not data:
        return
    width = max(900, 120 + len(data) * 120)
    height = 520
    left = 70
    right = 30
    top = 60
    bottom = 150
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_v = max(value + (err or 0.0) for _, value, err in data)
    max_v = max(max_v, 1e-9)
    colors = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#14b8a6"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2:.1f}" y="30" text-anchor="middle" font-family="Arial" font-size="20" font-weight="700">{title}</text>',
        f'<text x="20" y="{top + plot_h/2:.1f}" transform="rotate(-90 20 {top + plot_h/2:.1f})" text-anchor="middle" font-family="Arial" font-size="13">{ylabel}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{width-right}" y2="{top + plot_h}" stroke="#111827" stroke-width="1"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111827" stroke-width="1"/>',
    ]
    for i in range(6):
        value = max_v * i / 5
        y = top + plot_h - (value / max_v) * plot_h
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="11" fill="#4b5563">{value:.1f}</text>')

    slot = plot_w / len(data)
    bar_w = min(70, slot * 0.62)
    for idx, (label, value, err) in enumerate(data):
        x = left + idx * slot + (slot - bar_w) / 2
        bar_h = (value / max_v) * plot_h
        y = top + plot_h - bar_h
        color = colors[idx % len(colors)]
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" rx="3" fill="{color}"/>')
        if err:
            y_err = top + plot_h - ((value + err) / max_v) * plot_h
            y_mid = top + plot_h - (value / max_v) * plot_h
            cx = x + bar_w / 2
            parts.append(f'<line x1="{cx:.1f}" y1="{y_err:.1f}" x2="{cx:.1f}" y2="{y_mid:.1f}" stroke="#111827"/>')
            parts.append(f'<line x1="{cx-8:.1f}" y1="{y_err:.1f}" x2="{cx+8:.1f}" y2="{y_err:.1f}" stroke="#111827"/>')
        parts.append(f'<text x="{x + bar_w/2:.1f}" y="{y-6:.1f}" text-anchor="middle" font-family="Arial" font-size="11">{value:.2f}</text>')
        label_lines = label.split("\n")
        for j, line in enumerate(label_lines):
            parts.append(
                f'<text x="{x + bar_w/2:.1f}" y="{top + plot_h + 22 + j*14:.1f}" '
                f'text-anchor="middle" font-family="Arial" font-size="10" fill="#374151">{line}</text>'
            )
    note = "Lower is better" if lower_is_better else "Higher is better"
    parts.append(f'<text x="{width-right}" y="{height-18}" text-anchor="end" font-family="Arial" font-size="11" fill="#6b7280">{note}</text>')
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def write_markdown(path, rows, summaries, chart_paths):
    lines = [
        "# DeepSeek V4 Flash Q4 Long-Run Dataset",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Runs: {len(rows)}",
        "",
        "## Aggregate Summary",
        "",
        "| suite | case | n | wall s | disk GiB | prompt t/s | gen t/s | prefill s | decode s | prefill I/O | decode I/O |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["suite"],
                    row["case"],
                    str(row["n"]),
                    fmt(row.get("wall_s_mean")),
                    fmt(row.get("disk_read_gib_mean")),
                    fmt(row.get("prompt_tps_mean")),
                    fmt(row.get("generation_tps_mean")),
                    fmt(row.get("effective_prefill_wall_s_mean")),
                    fmt(row.get("effective_decode_wall_s_mean")),
                    fmt(row.get("effective_prefill_io_active_ratio_mean"), pct=True),
                    fmt(row.get("effective_decode_io_active_ratio_mean"), pct=True),
                ]
            )
            + " |"
        )

    if chart_paths:
        lines.extend(["", "## Charts", ""])
        for chart in chart_paths:
            rel = chart.relative_to(path.parent)
            lines.append(f"![{chart.stem}]({rel.as_posix()})")
            lines.append("")

    lines.extend(
        [
            "## Files",
            "",
            "- `runs.jsonl`: raw per-run rows",
            "- `runs.csv`: raw per-run rows for spreadsheets",
            "- `summary.csv`: grouped aggregate data",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Summarize DeepSeek Q4 long-run performance dataset")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    dataset = args.dataset
    out_dir = args.out_dir or dataset.parent
    rows = effective_rows(load_rows(dataset))
    summaries = aggregate(rows)

    write_rows_csv(out_dir / "runs.csv", rows)
    write_summary_csv(out_dir / "summary.csv", summaries)

    charts_dir = out_dir / "charts"
    chart_paths = [
        charts_dir / "wall_by_case.svg",
        charts_dir / "generation_tps_by_case.svg",
        charts_dir / "prefill_wall_by_case.svg",
        charts_dir / "decode_wall_by_case.svg",
        charts_dir / "disk_read_by_case.svg",
    ]
    bar_svg(chart_paths[0], "Wall Time by Case", summaries, "wall_s", "seconds", lower_is_better=True)
    bar_svg(chart_paths[1], "Generation Throughput by Case", summaries, "generation_tps", "tokens/sec")
    bar_svg(chart_paths[2], "Prefill Window by Case", summaries, "effective_prefill_wall_s", "seconds", lower_is_better=True)
    bar_svg(chart_paths[3], "Decode Window by Case", summaries, "effective_decode_wall_s", "seconds", lower_is_better=True)
    bar_svg(chart_paths[4], "Disk Read by Case", summaries, "disk_read_gib", "GiB", lower_is_better=True)
    chart_paths = [path for path in chart_paths if path.exists()]

    write_markdown(out_dir / "README.md", rows, summaries, chart_paths)
    print(f"dataset: {dataset}")
    print(f"runs: {len(rows)}")
    print(f"summary: {out_dir / 'README.md'}")


if __name__ == "__main__":
    main()
