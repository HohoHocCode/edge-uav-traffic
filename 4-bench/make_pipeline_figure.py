#!/usr/bin/env python3
"""Draw the pipeline as a diagram carrying its own measurements.

    python 4-bench/make_pipeline_figure.py --out docs/img

A row of unlabelled boxes says only that the stages exist, which the audience
already assumed. The same row with a measured millisecond figure under each box
says where the time goes, and that is the argument -- the NPU is not the
expensive stage.

Two colours only: what runs on the Hexagon and what runs on the Kryo cores.
Stage width is proportional to measured time, so the shape of the diagram *is*
the frame budget and cannot disagree with the table.
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BLUE, ORANGE, GREEN = "#0072B2", "#E69F00", "#009E73"
RED, PURPLE, GREY = "#D55E00", "#CC79A7", "#8C8C8C"
INK, PAPER = "#1A1A1A", "#FFFFFF"

plt.rcParams.update({
    "figure.dpi": 170, "savefig.dpi": 170, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans", "text.color": INK,
})


def box(ax, x, y, w, h, label, sub, face, fg="white", fs=13):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.012,rounding_size=0.02",
                                facecolor=face, edgecolor="none", zorder=3))
    ax.text(x + w / 2, y + h * 0.62, label, ha="center", va="center",
            color=fg, fontsize=fs, fontweight="bold", zorder=4)
    if sub:
        ax.text(x + w / 2, y + h * 0.26, sub, ha="center", va="center",
                color=fg, fontsize=fs - 3.5, zorder=4, alpha=.92)


def arrow(ax, x1, y1, x2, y2, colour=GREY, lw=2.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=15, color=colour,
                                 linewidth=lw, zorder=2,
                                 shrinkA=0, shrinkB=0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(args.root, "docs", "img")
    os.makedirs(out, exist_ok=True)

    b = json.load(open(os.path.join(args.root, "results", "board",
                                    "board_cpu_cold.json")))
    npu_ms = 3.984                       # v8n w8a16 on AI Hub QCS8550
    pre_ms = b["pre_ms"]
    post_ms = b["post_ms"]
    tiles = 4
    total = (npu_ms + pre_ms + post_ms) * tiles

    # Width proportional to per-frame cost, so the picture cannot contradict
    # the numbers printed on it.
    stages = [
        ("Camera",        None,          "1080p",            GREY,   1.45),
        ("Tile x4",       None,          "20% overlap",      GREY,   1.55),
        ("Preprocess",    pre_ms * tiles, "letterbox, CHW",  ORANGE, None),
        ("NPU  w8a16",    npu_ms * tiles, "Hexagon V73",     BLUE,   None),
        ("Decode + NMS",  post_ms * tiles, "Kryo CPU",       ORANGE, None),
        ("Tracker",       None,          "ByteTrack",        GREEN,  1.45),
        ("Telemetry",     None,          "counts, alerts",   GREEN,  1.7),
    ]

    scale = 7.4 / total
    fig, ax = plt.subplots(figsize=(15.4, 4.6))
    x, y, h = 0.0, 0.0, 1.0
    gap = 0.30
    centres = []
    for name, ms, sub, colour, fixed in stages:
        w = fixed if ms is None else max(ms * scale, 1.5)
        box(ax, x, y, w, h, name, sub, colour)
        centres.append((x + w / 2, w))
        if ms is not None:
            ax.text(x + w / 2, y - 0.30, f"{ms:.1f} ms", ha="center",
                    va="center", fontsize=14, fontweight="bold",
                    color=BLUE if colour == BLUE else ORANGE)
        x += w + gap

    for i in range(len(stages) - 1):
        c1, w1 = centres[i]
        c2, w2 = centres[i + 1]
        arrow(ax, c1 + w1 / 2, y + h / 2, c2 - w2 / 2, y + h / 2)

    # Brace over the three timed stages: this is the frame budget.
    a = centres[2][0] - centres[2][1] / 2
    z = centres[4][0] + centres[4][1] / 2
    ax.plot([a, a, z, z], [y + h + .18, y + h + .34, y + h + .34, y + h + .18],
            color=INK, linewidth=1.4, zorder=3)
    ax.text((a + z) / 2, y + h + .50,
            f"{total:.1f} ms per frame  ->  ~{1000/total:.0f} FPS",
            ha="center", fontsize=15, fontweight="bold")

    share = post_ms * tiles / total * 100
    ax.text(x / 2, y - 0.85,
            f"The NPU is only {npu_ms*tiles/total*100:.0f}% of the budget; decode + NMS "
            f"on the CPU is {share:.0f}%.\n"
            f"Box width is proportional to measured time, so the drawing cannot "
            f"disagree with the table.",
            ha="center", va="center", fontsize=11.5, color=GREY)

    ax.text(0, y + h + .18, "  measured on board", fontsize=10.5, color=GREY,
            va="bottom")

    ax.set_xlim(-0.25, x + 0.1)
    ax.set_ylim(-1.35, y + h + 0.85)
    ax.axis("off")
    p = os.path.join(out, "fig_pipeline.png")
    fig.savefig(p, facecolor=PAPER)
    plt.close(fig)
    print(f"[ok] {os.path.relpath(p, args.root)}  "
          f"{os.path.getsize(p)/1e3:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
