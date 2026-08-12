#!/usr/bin/env python3
"""Turn benchmark CSVs into the tables and figures that go in the report.

    python 4-bench/report.py --quality results/quality_mask.csv \
        --latency results/latency.csv --out results/report

Produces:
    report.md                  every table, ready to paste
    fig_robustness.png         AP / APs vs condition, with retention
    fig_latency_split.png      where the milliseconds actually go
    fig_degradation_cost.png   the accuracy/latency double penalty

The figures are deliberately plain: no gradients, no 3-D, one message each.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Condition display order and grouping — keeps the x-axis meaningful rather
# than alphabetical.
ORDER = [
    "clean",
    "rain_light", "rain_medium", "rain_heavy",
    "bright_down", "bright_down_heavy", "bright_up",
    "blur_light", "blur_medium",
    "fog_medium",
]
GROUP = {
    "clean": "reference",
    "rain_light": "rain", "rain_medium": "rain", "rain_heavy": "rain",
    "bright_down": "exposure", "bright_down_heavy": "exposure", "bright_up": "exposure",
    "blur_light": "blur", "blur_medium": "blur",
    "fog_medium": "fog",
}
GROUP_COLOUR = {
    "reference": "#444444", "rain": "#2f6fb5", "exposure": "#d68910",
    "blur": "#7d3c98", "fog": "#148f77",
}


def _order_key(c: str) -> int:
    return ORDER.index(c) if c in ORDER else len(ORDER)


def load_quality(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop_duplicates(subset=["condition", "model_sha256_16", "backend",
                                    "imgsz", "ignore_policy"], keep="last")
    df["_k"] = df["condition"].map(_order_key)
    return df.sort_values("_k").drop(columns="_k").reset_index(drop=True)


def md_table(rows: list[list[str]], header: list[str], align: list[str] | None = None) -> str:
    align = align or ["---"] * len(header)
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join(align) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
def robustness_table(df: pd.DataFrame) -> str:
    ref = df[df["condition"] == "clean"]
    ap0 = float(ref["AP"].iloc[0]) if len(ref) else float("nan")
    aps0 = float(ref["APs"].iloc[0]) if len(ref) else float("nan")
    post0 = float(ref["post_ms_avg"].iloc[0]) if len(ref) else float("nan")

    rows = []
    for _, r in df.iterrows():
        ap_ret = 100.0 * r["AP"] / ap0 if ap0 > 0 else float("nan")
        aps_ret = 100.0 * r["APs"] / aps0 if aps0 > 0 else float("nan")
        post_d = 100.0 * (r["post_ms_avg"] / post0 - 1.0) if post0 > 0 else float("nan")
        rows.append([
            f"`{r['condition']}`",
            f"{r['AP']:.4f}", f"{r['AP50']:.4f}", f"{r['APs']:.4f}",
            f"{ap_ret:.0f}%", f"{aps_ret:.0f}%",
            f"{r['det_per_image']:.0f}",
            f"{r['post_ms_avg']:.1f}", f"{post_d:+.0f}%",
        ])
    return md_table(
        rows,
        ["condition", "AP", "AP50", "APs", "AP kept", "APs kept",
         "det/img", "post ms", "Δ post"],
        ["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"],
    )


def latency_table(df: pd.DataFrame) -> str:
    rows = []
    for _, r in df.iterrows():
        total = r["pre_ms_avg"] + r["infer_ms_avg"] + r["post_ms_avg"]
        rows.append([
            f"`{r['condition']}`", f"{r['pre_ms_avg']:.1f}",
            f"{r['infer_ms_avg']:.1f}", f"{r['post_ms_avg']:.1f}",
            f"{total:.1f}", f"{1000.0 / max(total, 1e-9):.1f}",
        ])
    return md_table(
        rows, ["condition", "pre", "infer", "post", "total ms", "FPS"],
        ["---", "---:", "---:", "---:", "---:", "---:"],
    )


def _latency_caveat(df: pd.DataFrame) -> str:
    """Warn when the host was too noisy for the latency columns to mean anything.

    ``infer_ms`` is a fixed-shape forward pass: identical work on every frame
    regardless of what the frame contains. If it varies across conditions, the
    variation is host load or thermal state, not the data — and the ``post_ms``
    deltas measured alongside it are contaminated by the same noise. Better to
    say so in the report than to let a reader infer a content effect from a
    scheduling artefact.
    """
    inf = df["infer_ms_avg"]
    spread = float(inf.max() / max(inf.min(), 1e-9) - 1.0) * 100.0
    if spread < 5.0:
        return (f"_Host was stable during this run: `infer_ms` varies only "
                f"{spread:.1f}% across conditions, so the `post_ms` deltas "
                f"reflect the data rather than the machine._")

    worst = df.loc[inf.idxmax(), "condition"]
    return (
        f"> **Latency columns in this run are not reliable.** `infer_ms` is a\n"
        f"> fixed-shape forward pass — identical work on every frame — yet it\n"
        f"> varies **{spread:.0f}%** across conditions here (worst: "
        f"`{worst}`).\n"
        f"> That is host load or thermal state, not the data, and the same\n"
        f"> noise contaminates the `post_ms` deltas beside it. Quote the AP\n"
        f"> columns, which are deterministic; re-measure latency on an idle\n"
        f"> machine, or better, on the target device."
    )


def per_class_table(df: pd.DataFrame) -> str:
    cols = [c for c in df.columns if c.startswith("AP_") and c not in
            ("AP_retention",)]
    if not cols:
        return "_no per-class columns in this CSV_"
    clean = df[df["condition"] == "clean"]
    if not len(clean):
        return "_no clean reference row_"
    r = clean.iloc[0]
    rows = sorted(
        ([c.replace("AP_", ""), float(r[c])] for c in cols if pd.notna(r[c])),
        key=lambda x: -x[1],
    )
    return md_table([[n, f"{v:.4f}"] for n, v in rows], ["class", "AP (clean)"],
                    ["---", "---:"])


# --------------------------------------------------------------------------- #
def fig_robustness(df: pd.DataFrame, path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.6))
    x = np.arange(len(df))
    w = 0.38
    colours = [GROUP_COLOUR[GROUP.get(c, "reference")] for c in df["condition"]]

    ax.bar(x - w / 2, df["AP"], w, label="AP", color=colours, alpha=0.95)
    ax.bar(x + w / 2, df["APs"], w, label="APs (small)", color=colours, alpha=0.5,
           hatch="//")

    ref = df[df["condition"] == "clean"]
    if len(ref):
        ax.axhline(float(ref["AP"].iloc[0]), color="#444444", ls="--", lw=1,
                   label="clean AP")

    ax.set_xticks(x)
    ax.set_xticklabels(df["condition"], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Average Precision")
    ax.set_title("Detection quality under degraded capture — VisDrone-DET val (548 images)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_latency_split(df: pd.DataFrame, path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = np.arange(len(df))
    pre, inf, post = df["pre_ms_avg"], df["infer_ms_avg"], df["post_ms_avg"]

    ax.bar(x, pre, label="pre (CPU)", color="#9aa3b2")
    ax.bar(x, inf, bottom=pre, label="infer (compute unit)", color="#2f6fb5")
    ax.bar(x, post, bottom=pre + inf, label="post / NMS (CPU)", color="#d64545")

    ax.set_xticks(x)
    ax.set_xticklabels(df["condition"], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("milliseconds per frame")
    ax.set_title("Where the milliseconds go — the CPU tail grows with degradation")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_double_penalty(df: pd.DataFrame, path: str) -> None:
    """The finding worth its own figure: worse accuracy *and* worse latency."""
    ref = df[df["condition"] == "clean"]
    if not len(ref):
        return
    ap0 = float(ref["AP"].iloc[0])
    post0 = float(ref["post_ms_avg"].iloc[0])

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    for _, r in df.iterrows():
        g = GROUP.get(r["condition"], "reference")
        ax.scatter(100.0 * r["AP"] / ap0, 100.0 * (r["post_ms_avg"] / post0 - 1.0),
                   s=90, color=GROUP_COLOUR[g], edgecolor="white", zorder=3)
        ax.annotate(r["condition"], (100.0 * r["AP"] / ap0,
                                     100.0 * (r["post_ms_avg"] / post0 - 1.0)),
                    textcoords="offset points", xytext=(7, 4), fontsize=8)

    ax.axhline(0, color="#888", lw=1, ls=":")
    ax.axvline(100, color="#888", lw=1, ls=":")
    ax.set_xlabel("AP retained vs clean (%)  →  better")
    ax.set_ylabel("CPU postprocessing cost vs clean (%)  →  worse")
    ax.set_title("The double penalty:\ndegradation costs accuracy AND latency",
                 fontsize=11)
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quality", required=True)
    ap.add_argument("--latency", default=None)
    ap.add_argument("--out", default="results/report")
    ap.add_argument("--img-prefix", default="",
                    help="prefix for image links in the markdown, e.g. 'img/' "
                         "when the report is published from docs/ while the "
                         "figures live in docs/img/")
    ap.add_argument("--md-name", default="report.md")
    args = ap.parse_args()

    if not os.path.exists(args.quality):
        print(f"[fatal] {args.quality} not found", file=sys.stderr)
        return 2

    os.makedirs(args.out, exist_ok=True)
    df = load_quality(args.quality)
    print(f"[info] {len(df)} condition rows")

    fig_robustness(df, os.path.join(args.out, "fig_robustness.png"))
    fig_latency_split(df, os.path.join(args.out, "fig_latency_split.png"))
    fig_double_penalty(df, os.path.join(args.out, "fig_degradation_cost.png"))

    meta = df.iloc[0]
    lines = [
        "# SkySentry benchmark report",
        "",
        f"- model: `{meta['model']}` (sha256 `{meta['model_sha256_16']}`)",
        f"- backend: `{meta['backend']}`, imgsz {meta['imgsz']}",
        f"- eval set: VisDrone-DET val, {meta['n_images']} images, 10 classes",
        f"- ignore-region policy: `{meta['ignore_policy']}`",
        f"- conf {meta['conf_thres']}, IoU {meta['iou_thres']}",
        "",
        "## 1. Robustness under degraded capture",
        "",
        robustness_table(df),
        "",
        f"![robustness]({args.img_prefix}fig_robustness.png)",
        "",
        "## 2. Latency split",
        "",
        "`infer` is the compute unit; `pre` and `post` are always CPU. They are",
        "reported separately because they scale with different things — `infer`",
        "with resolution, `post` with scene density.",
        "",
        latency_table(df),
        "",
        f"![latency]({args.img_prefix}fig_latency_split.png)",
        "",
        "## 3. Degradation cost — and why it flips with the threshold",
        "",
        f"At the AP-measurement threshold (conf {meta['conf_thres']}) degradation",
        "can create spurious candidates that survive scoring, so NMS on the CPU",
        "has more boxes to process and postprocessing gets *slower*. At a",
        "deployment threshold the effect can reverse: degradation pushes",
        "confidences below the bar, fewer boxes reach NMS, and the pipeline gets",
        "**faster because it sees less**.",
        "",
        "That second case is the more dangerous one operationally — latency",
        "telemetry looks healthy at exactly the moment the detector is going",
        "blind, and since the vehicle count falls too, a congestion monitor",
        "reads 'normal' when the truth is 'cannot see'.",
        "",
        "Any claim about the latency cost of degradation must state the",
        "confidence threshold it was measured at **and the model it was measured",
        f"on** — this report covers `{meta['model']}` only. Deployment-threshold",
        "figures come from a separate run of",
        "`scripts/make_comparison_video.py`, which writes its own CSV.",
        "",
        _latency_caveat(df),
        "",
        f"![double penalty]({args.img_prefix}fig_degradation_cost.png)",
        "",
        "## 4. Per-class AP on clean data",
        "",
        per_class_table(df),
        "",
    ]

    if args.latency and os.path.exists(args.latency):
        ldf = pd.read_csv(args.latency)
        rows = []
        for _, r in ldf.iterrows():
            if r.get("status") != "ok":
                continue
            rows.append([
                f"`{r['backend']}`", int(r["imgsz"]), int(r["n_candidates"]),
                f"{r['infer_ms_avg']:.2f}", f"{r['post_ms_avg']:.2f}",
                f"{r['total_ms_avg']:.2f}", f"{r['fps_est']:.1f}",
            ])
        if rows:
            lines += [
                "## 5. Latency ladder (synthetic load)",
                "",
                md_table(rows, ["backend", "imgsz", "candidates", "infer ms",
                                "post ms", "total ms", "FPS"],
                         ["---", "---:", "---:", "---:", "---:", "---:", "---:"]),
                "",
            ]

    md_path = os.path.join(args.out, args.md_name)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[ok] {md_path}")
    for n in ("fig_robustness.png", "fig_latency_split.png",
              "fig_degradation_cost.png"):
        print(f"[ok] {os.path.join(args.out, n)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
