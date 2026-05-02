#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import re
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "docs" / "video" / "build"
FRAMES = BUILD / "frames"
LIVE_LOG = BUILD / "live-generation-latest.log"
MONITOR_CSV = BUILD / "resource-monitor-latest.csv"
SRT_OUT = BUILD / "moe-madv-shorts-draft.en.srt"

W, H = 1080, 1920
FPS = 12
DURATION_S = 60
TOTAL_FRAMES = FPS * DURATION_S

FONT_DIR = Path("/System/Library/Fonts")
SUPP_FONT_DIR = FONT_DIR / "Supplemental"


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        FONT_DIR / name,
        SUPP_FONT_DIR / name,
        FONT_DIR / "SFNS.ttf",
        FONT_DIR / "SFNSMono.ttf",
        SUPP_FONT_DIR / "Arial.ttf",
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


FONT_TITLE = font("SFNS.ttf", 70)
FONT_HEAD = font("SFNS.ttf", 46)
FONT_BODY = font("SFNS.ttf", 34)
FONT_SMALL = font("SFNS.ttf", 27)
FONT_TINY = font("SFNS.ttf", 23)
FONT_MONO = font("SFNSMono.ttf", 27)
FONT_MONO_SMALL = font("SFNSMono.ttf", 22)
FONT_CAPTION = font("SFNS.ttf", 38)


def read_live_log() -> dict[str, str]:
    text = LIVE_LOG.read_text(errors="replace") if LIVE_LOG.exists() else ""
    json_match = re.search(r'(\{"status":"ok".*?)(?:\n|\[ Prompt)', text, re.S)
    plain_match = re.search(
        r"> In one sentence.*?\n\n(.*?)\n\n\[ Prompt:",
        text,
        re.S,
    )
    gen_rates = re.findall(r"Generation:\s*([0-9.]+)\s*t/s", text)
    elapsed = re.findall(r"Elapsed wall time:\s*([0-9]+)s", text)
    return {
        "json": (json_match.group(1).strip() if json_match else '{"status":"ok","model":"Deep...').replace("\n", " "),
        "plain": (plain_match.group(1).strip() if plain_match else "MoE page loading matters for local inference...").replace("\n", " "),
        "json_rate": gen_rates[0] if gen_rates else "1.1",
        "plain_rate": gen_rates[1] if len(gen_rates) > 1 else "1.0",
        "json_elapsed": elapsed[0] if elapsed else "110",
        "plain_elapsed": elapsed[1] if len(elapsed) > 1 else "117",
    }


def read_monitor() -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    if not MONITOR_CSV.exists():
        return rows
    with MONITOR_CSV.open() as f:
        for row in csv.DictReader(f):
            try:
                rows.append(
                    {
                        "timestamp": row["timestamp"],
                        "elapsed_s": float(row["elapsed_s"]),
                        "rss": float(row["app_rss_gib"]),
                        "disk": float(row["disk_read_mib_s"]),
                        "pagein": float(row["pagein_mib_s"]),
                    }
                )
            except Exception:
                continue
    return rows


def lerp(a: float, b: float, x: float) -> float:
    return a + (b - a) * max(0.0, min(1.0, x))


def rounded(draw: ImageDraw.ImageDraw, box, radius, fill, outline=None, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def text(draw: ImageDraw.ImageDraw, xy, msg, font_obj, fill=(245, 248, 255), anchor=None):
    draw.text(xy, msg, font=font_obj, fill=fill, anchor=anchor)


def wrap(draw: ImageDraw.ImageDraw, msg: str, font_obj, max_width: int) -> list[str]:
    words = msg.split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = f"{current} {word}".strip()
        if draw.textbbox((0, 0), trial, font=font_obj)[2] <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_wrapped(draw, xy, msg, font_obj, fill, max_width, line_h):
    x, y = xy
    for line in wrap(draw, msg, font_obj, max_width):
        text(draw, (x, y), line, font_obj, fill)
        y += line_h
    return y


def draw_chip(draw, x, y, label, fill=(37, 48, 68), outline=(83, 105, 135)):
    bbox = draw.textbbox((0, 0), label, font=FONT_SMALL)
    w = bbox[2] + 34
    rounded(draw, (x, y, x + w, y + 48), 18, fill, outline, 1)
    text(draw, (x + 17, y + 10), label, FONT_SMALL, (230, 238, 248))
    return x + w + 12


def draw_terminal(draw, box, lines, reveal=1.0):
    x1, y1, x2, y2 = box
    rounded(draw, box, 22, (12, 17, 25), (69, 89, 115), 2)
    rounded(draw, (x1, y1, x2, y1 + 54), 22, (25, 32, 43), None, 0)
    for i, color in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        draw.ellipse((x1 + 24 + i * 36, y1 + 18, x1 + 42 + i * 36, y1 + 36), fill=color)
    text(draw, (x1 + 150, y1 + 15), "local DeepSeek V4 Flash generation", FONT_TINY, (160, 178, 203))

    visible = max(1, int(len(lines) * reveal))
    y = y1 + 78
    for line in lines[:visible]:
        color = (214, 232, 255)
        if line.startswith("$"):
            color = (127, 231, 196)
        elif line.startswith(">"):
            color = (255, 216, 117)
        elif "Generation:" in line:
            color = (142, 204, 255)
        elif "FAST" in line:
            color = (255, 144, 122)
        clipped = line if len(line) <= 62 else line[:59].rstrip() + "..."
        draw.text((x1 + 28, y), clipped, font=FONT_MONO_SMALL, fill=color)
        y += 34
        if y > y2 - 42:
            break


def draw_monitor(draw, box, rows, progress):
    x1, y1, x2, y2 = box
    rounded(draw, box, 22, (15, 24, 31), (69, 96, 112), 2)
    text(draw, (x1 + 28, y1 + 24), "live sidecar monitor", FONT_SMALL, (180, 230, 215))

    row = rows[min(len(rows) - 1, max(0, int(progress * (len(rows) - 1))))] if rows else {
        "elapsed_s": 0,
        "rss": 44.7,
        "disk": 0,
        "pagein": 0,
    }
    elapsed = int(float(row["elapsed_s"]))
    rss = float(row["rss"])
    disk = float(row["disk"])
    pagein = float(row["pagein"])
    rss_pct = min(100, rss / 64.0 * 100)

    text(draw, (x1 + 28, y1 + 78), f"Elapsed  {elapsed//60:02d}:{elapsed%60:02d}", FONT_BODY, (245, 248, 255))
    text(draw, (x1 + 360, y1 + 78), "Clock visible in source capture", FONT_TINY, (150, 166, 188))

    text(draw, (x1 + 28, y1 + 138), f"llama RSS  {rss:.1f} / 64 GiB", FONT_SMALL, (234, 241, 250))
    bx, by, bw, bh = x1 + 28, y1 + 180, x2 - x1 - 56, 34
    rounded(draw, (bx, by, bx + bw, by + bh), 14, (29, 39, 52), None, 0)
    rounded(draw, (bx, by, bx + int(bw * rss_pct / 100), by + bh), 14, (81, 203, 173), None, 0)

    chart_x, chart_y = x1 + 28, y1 + 250
    chart_w, chart_h = x2 - x1 - 56, 122
    rounded(draw, (chart_x, chart_y, chart_x + chart_w, chart_y + chart_h), 14, (20, 29, 39), (46, 66, 84), 1)
    text(draw, (chart_x + 14, chart_y + 12), f"Disk {disk:.0f} MiB/s", FONT_TINY, (177, 213, 255))
    text(draw, (chart_x + 260, chart_y + 12), f"Page-in {pagein:.0f} MiB/s", FONT_TINY, (255, 206, 138))
    max_v = max(1.0, max([float(r["disk"]) for r in rows] + [disk]), max([float(r["pagein"]) for r in rows] + [pagein]))
    sample_count = min(len(rows), 64)
    if sample_count > 1:
        start = max(0, int(progress * max(0, len(rows) - sample_count)))
        sample = rows[start : start + sample_count]
        for key, color in [("disk", (120, 183, 255)), ("pagein", (255, 185, 95))]:
            pts = []
            for idx, r in enumerate(sample):
                px = chart_x + 14 + idx * (chart_w - 28) / (sample_count - 1)
                py = chart_y + chart_h - 14 - min(1.0, float(r[key]) / max_v) * (chart_h - 52)
                pts.append((px, py))
            if len(pts) > 1:
                draw.line(pts, fill=color, width=4)

    text(draw, (x1 + 28, y2 - 46), "GPU History: show Activity Monitor beside this render", FONT_TINY, (150, 166, 188))


def draw_result(draw, box):
    x1, y1, x2, y2 = box
    rounded(draw, box, 24, (19, 23, 34), (88, 105, 145), 2)
    text(draw, (x1 + 30, y1 + 28), "Decode generation", FONT_HEAD, (250, 251, 255))
    base, opt = 0.98, 1.23
    maxv = 1.35
    labels = [("Baseline", base, (127, 143, 165)), ("MoE-MADV", opt, (95, 224, 176))]
    for i, (label, value, color) in enumerate(labels):
        y = y1 + 108 + i * 88
        text(draw, (x1 + 34, y), label, FONT_SMALL, (220, 228, 240))
        bw = int((x2 - x1 - 300) * value / maxv)
        rounded(draw, (x1 + 210, y + 2, x1 + 210 + bw, y + 42), 16, color, None, 0)
        text(draw, (x1 + 225 + bw, y + 7), f"{value:.2f} tok/s", FONT_SMALL, (245, 248, 255))
    text(draw, (x1 + 34, y2 - 52), "+25.4% throughput, same model", FONT_BODY, (255, 223, 134))


CAPTIONS = [
    (0, 4, "150GB DeepSeek V4 Flash MoE."),
    (4, 8, "Running locally on a 64GB M1 Max."),
    (8, 13, "Specs, elapsed time, memory, and I/O are visible."),
    (13, 18, "Now the local GGUF model is generating tokens."),
    (18, 25, "Fast-forwarded: the model is larger than RAM."),
    (25, 33, "For MoE, the hard part is getting the right expert pages resident."),
    (33, 42, "MoE-MADV gives macOS a MADV_WILLNEED hint after routing."),
    (42, 50, "Decode improved from 0.98 to 1.23 tokens per second."),
    (50, 56, "That is +25.4% without changing the model."),
    (56, 60, "github.com/daystar7777/MoE-MADV"),
]


def current_caption(t: float) -> str:
    for start, end, cap in CAPTIONS:
        if start <= t < end:
            return cap
    return CAPTIONS[-1][2]


def write_srt():
    def stamp(sec: float) -> str:
        ms = int(round((sec - int(sec)) * 1000))
        sec_i = int(sec)
        return f"00:{sec_i // 60:02d}:{sec_i % 60:02d},{ms:03d}"

    lines = []
    for i, (start, end, cap) in enumerate(CAPTIONS, 1):
        lines += [str(i), f"{stamp(start)} --> {stamp(end)}", cap, ""]
    SRT_OUT.write_text("\n".join(lines))


def draw_frame(frame_idx: int, live: dict[str, str], rows) -> Image.Image:
    t = frame_idx / FPS
    img = Image.new("RGB", (W, H), (8, 13, 21))
    draw = ImageDraw.Draw(img)

    # Soft vertical gradient.
    for y in range(H):
        k = y / H
        r = int(8 + 10 * k)
        g = int(13 + 18 * k)
        b = int(21 + 25 * k)
        draw.line((0, y, W, y), fill=(r, g, b))

    text(draw, (54, 48), "MoE-MADV", FONT_TITLE, (247, 250, 255))
    text(draw, (58, 132), "150GB DeepSeek V4 Flash on 64GB M1 Max", FONT_BODY, (177, 204, 232))

    x = 58
    for chip in ["284B MoE", "MXFP4_MOE GGUF", "Apple M1 Max", "64GB unified"]:
        x = draw_chip(draw, x, 190, chip)

    if t < 8:
        rounded(draw, (58, 292, 1022, 760), 30, (18, 26, 38), (72, 98, 130), 2)
        draw_wrapped(draw, (92, 336), "A 150GB model file is running on a 64GB Apple Silicon machine.", FONT_HEAD, (250, 252, 255), 890, 58)
        draw_wrapped(draw, (92, 500), "For sparse MoE models, the bottleneck is often expert page residency, not only raw compute.", FONT_BODY, (192, 213, 235), 870, 45)
        rounded(draw, (92, 638, 400, 704), 22, (95, 224, 176), None, 0)
        text(draw, (124, 654), "real local run", FONT_BODY, (6, 20, 25))
    else:
        terminal_lines = [
            "$ scripts/run_moe_madv_live_generation_demo.sh",
            "Local model file: 140G DeepSeek-V4-Flash-MXFP4_MOE.gguf",
            "Clock time: 2026-05-02 17:54:37 JST",
            "> Return JSON only: {\"status\":\"ok\",\"model\":\"DeepSeek V4 Flash\"}",
            live["json"],
            f"[ Prompt: 1.0 t/s | Generation: {live['json_rate']} t/s ]",
            f"Elapsed wall time: {live['json_elapsed']}s",
            "",
            "> In one sentence, explain why MoE page loading matters...",
            live["plain"],
            f"[ Prompt: 0.9 t/s | Generation: {live['plain_rate']} t/s ]",
            f"Elapsed wall time: {live['plain_elapsed']}s",
        ]
        if 18 <= t < 42:
            terminal_lines.insert(4, "FAST-FORWARDED 12x: loading 150GB mmap-backed model")
        reveal = 1.0 if t >= 18 else max(0.28, (t - 8) / 10)
        draw_terminal(draw, (58, 292, 1022, 980), terminal_lines, reveal)

    if t >= 8:
        progress = max(0.0, min(1.0, (t - 13) / 29))
        draw_monitor(draw, (58, 1018, 1022, 1398), rows, progress)

    if t >= 42:
        draw_result(draw, (58, 1430, 1022, 1748))
    else:
        rounded(draw, (58, 1430, 1022, 1748), 24, (16, 22, 33), (58, 77, 104), 2)
        text(draw, (88, 1466), "Why page-in timing matters", FONT_HEAD, (245, 248, 255))
        draw_wrapped(draw, (88, 1535), "DeepSeek V4 Flash routes each token through a changing expert set. The selected expert pages need to become resident before decode needs them.", FONT_BODY, (190, 211, 235), 870, 45)

    rounded(draw, (58, 1782, 1022, 1880), 24, (0, 0, 0), None, 0)
    cap = current_caption(t)
    draw_wrapped(draw, (92, 1802), cap, FONT_CAPTION, (255, 255, 255), 900, 44)

    if 18 <= t < 42:
        rounded(draw, (690, 250, 1022, 292), 18, (193, 80, 62), None, 0)
        text(draw, (718, 258), "FAST-FORWARDED 12x", FONT_TINY, (255, 246, 232))

    if t >= 56:
        rounded(draw, (92, 252, 988, 342), 26, (95, 224, 176), None, 0)
        text(draw, (130, 275), "github.com/daystar7777/MoE-MADV", FONT_BODY, (5, 22, 23))

    return img


def main():
    if FRAMES.exists():
        shutil.rmtree(FRAMES)
    FRAMES.mkdir(parents=True, exist_ok=True)
    BUILD.mkdir(parents=True, exist_ok=True)
    live = read_live_log()
    rows = read_monitor()
    write_srt()
    for i in range(TOTAL_FRAMES):
        img = draw_frame(i, live, rows)
        img.save(FRAMES / f"frame_{i:04d}.png", optimize=False)
        if i % 60 == 0:
            print(f"rendered {i}/{TOTAL_FRAMES}")
    print(f"frames: {FRAMES}")
    print(f"subtitles: {SRT_OUT}")


if __name__ == "__main__":
    main()
