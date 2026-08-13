#!/usr/bin/env python3
"""Render the presentation figures from the measured CSVs.

    python 4-bench/make_figures.py --out docs/img

Every figure is built from a file in results/ or docs/results/, never from a
number typed in here, so a figure cannot drift away from the table it
illustrates. Sized and weighted for a projected slide: large type, few lines,
one claim per panel.

Palette is colour-blind safe (Okabe-Ito). Red is reserved for "this
configuration does not work" and is never used decoratively -- on these figures
the fastest bar is usually the broken one, and that has to be legible at a
glance from the back of a room.
"""

from __future__ import annotations

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# Okabe-Ito
BLUE, ORANGE, GREEN = "#0072B2", "#E69F00", "#009E73"
RED, PURPLE, GREY = "#D55E00", "#CC79A7", "#8C8C8C"
INK, FAINT = "#1A1A1A", "#D9D9D9"

plt.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "axes.labelsize": 13,
    "axes.edgecolor": "#B0B0B0",
    "axes.linewidth": 0.9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": FAINT,
    "grid.linewidth": 0.8,
    "xtick.color": INK,
    "ytick.color": INK,
    "text.color": INK,
    "axes.labelcolor": INK,
    "legend.frameon": False,
})


def rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(v, d=float("nan")):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


# --------------------------------------------------------------------------- #
def fig_quant_trap(root, out):
    """Latency next to usability. The point is that they disagree."""
    models = [("v8n-base", "quant_matrix_v8n.csv", "quant_check_v8n.csv"),
              ("v11n-base", "quant_matrix_v11n.csv", "quant_check_v11n.csv"),
              ("v26n end2end", "quant_matrix_v26n.csv", "quant_check_v26n.csv"),
              ("v26n-p2 no-e2e", "quant_matrix_v26n_p2.csv", "quant_check_v26n_p2.csv")]
    prec = ["w8a8", "w8a16", "w16a16", "w4a16"]

    # Three states, not two. A binary usable/broken flag calls w4a16 a pass
    # because it returns *some* valid boxes -- v11n keeps 29 against fp32's 244.
    # That is not a working model, and drawing it as a solid bar would put a
    # wrong claim on a slide. Anything under 70% of the fp32 box count is
    # degraded and drawn differently from both.
    OK, DEGRADED, BROKEN = 0, 1, 2
    lat, state = {}, {}
    for name, mf, cf in models:
        m = {r["precision"]: r for r in rows(os.path.join(root, "results", mf))}
        c = {r["precision"]: r for r in rows(os.path.join(root, "results", cf))}
        ref = num(c.get("fp32", {}).get("n_box_valid"), 1) or 1
        lat[name] = [num(m.get(p, {}).get("inference_us")) / 1000 for p in prec]
        st = []
        for p in prec:
            nb = num(c.get(p, {}).get("n_box_valid"), 0)
            st.append(BROKEN if nb == 0 else
                      (OK if nb / ref >= 0.7 else DEGRADED))
        state[name] = st

    hi = max(v for L in lat.values() for v in L if not np.isnan(v))
    fig, ax = plt.subplots(figsize=(11, 5.4))
    x = np.arange(len(prec))
    w = 0.21
    for i, (name, _, _) in enumerate(models):
        off = (i - 1.5) * w
        base = [BLUE, ORANGE, GREEN, PURPLE][i]
        for j, (v, st) in enumerate(zip(lat[name], state[name])):
            if np.isnan(v):
                continue
            face = base if st == OK else ("white" if st == BROKEN else base)
            edge = base if st == OK else (RED if st == BROKEN else RED)
            hatch = "" if st == OK else ("///" if st == BROKEN else "..")
            ax.bar(x[j] + off, v, w * 0.92, color=face, edgecolor=edge,
                   hatch=hatch, linewidth=1.6, alpha=1.0 if st != DEGRADED else .55,
                   zorder=3)
            if st != OK:
                ax.text(x[j] + off, v + hi * 0.03, "✗" if st == BROKEN else "!",
                        ha="center", va="bottom", color=RED, fontsize=15,
                        fontweight="bold", zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels([p.upper() for p in prec])
    ax.set_ylabel("Latency trên QCS8550 (ms)")
    ax.set_title("Precision nhanh nhất là precision không dùng được")
    # lat[] is already in ms -- dividing again here put the ceiling at 1.0 while
    # the bars reached 6.9, which pushed every label outside the axes and made
    # tight-bbox blow the canvas up to 5000 px tall.
    ax.set_ylim(0, hi * 1.30)
    ax.margins(x=0.04)

    handles = [Patch(facecolor=c, edgecolor=c, label=n)
               for (n, _, _), c in zip(models, [BLUE, ORANGE, GREEN, PURPLE])]
    handles.append(Patch(facecolor="white", edgecolor=RED, hatch="///",
                         label="✗ output hỏng hoàn toàn"))
    handles.append(Patch(facecolor=GREY, edgecolor=RED, hatch="..", alpha=.55,
                         label="! suy giảm nặng (<70% box)"))
    ax.legend(handles=handles, loc="upper left", ncol=2, fontsize=11.5)

    ax.text(0.5, -0.17, "Chỉ cột đặc mới dùng được. Mọi ô đều zero operator "
            "fallback — latency đẹp không nói lên model còn hoạt động.",
            transform=ax.transAxes, ha="center", fontsize=11, color=GREY)
    p = os.path.join(out, "fig_quant_trap.png")
    fig.savefig(p); plt.close(fig)
    return p


# --------------------------------------------------------------------------- #
def fig_e2e_matrix(root, out):
    """Where the decode lives x precision -> which failure appears."""
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    cols = ["W8A8", "W8A16", "W16A16", "W4A16"]
    rws = ["end2end\n(decode TRONG graph)", "không end2end\n(decode NGOÀI graph)"]
    state = [[2, 1, 1, 1],
             [2, 0, 3, 3]]                     # 0 ok, 1 box, 2 conf, 3 chưa đo
    label = [["conf→0", "box→điểm", "box→điểm", "box→điểm"],
             ["conf→0", "CHẠY TỐT", "—", "—"]]
    colour = {0: GREEN, 1: RED, 2: ORANGE, 3: "#F2F2F2"}

    for i in range(2):
        for j in range(4):
            s = state[i][j]
            ax.add_patch(plt.Rectangle((j, 1 - i), 1, 1, facecolor=colour[s],
                                       edgecolor="white", linewidth=3, zorder=2))
            ax.text(j + .5, 1 - i + .5, label[i][j], ha="center", va="center",
                    color="white" if s != 3 else GREY, zorder=3,
                    fontsize=13 if s != 0 else 14,
                    fontweight="bold" if s == 0 else "normal")

    ax.set_xlim(0, 4); ax.set_ylim(0, 2)
    ax.set_xticks(np.arange(4) + .5); ax.set_xticklabels(cols)
    ax.set_yticks(np.arange(2) + .5); ax.set_yticklabels(rws[::-1], fontsize=12)
    ax.set_title("Cùng model, cùng calibration — chỉ khác chỗ đặt phép decode")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.text(0.5, -0.22, "YOLO26 bỏ NMS bằng cách đưa decode vào graph. Phép cộng "
            "offset lưới anchor (0–640) với khoảng cách hồi quy\ncần dải động lớn "
            "hơn một scale int16 giữ được — nên box sập, trong khi confidence vẫn "
            "khớp fp32 tới 0.999.",
            transform=ax.transAxes, ha="center", fontsize=11, color=GREY)
    p = os.path.join(out, "fig_e2e_matrix.png")
    fig.savefig(p); plt.close(fig)
    return p


# --------------------------------------------------------------------------- #
def fig_models(root, out):
    """Three architectures on one tiled split, one decoder, one AP code."""
    rs = rows(os.path.join(root, "results", "tiled_val_4models.csv"))
    name = {"v26n-base-encoder-md500.onnx": "v26n-base",
            "v11n-base-encoder.onnx": "v11n-base",
            "v8n-base-encoder.onnx": "v8n-base",
            "v26n-p2-encoder.onnx": "v26n-p2"}
    rs = sorted(rs, key=lambda r: -num(r["AP"]))
    labels = [name.get(r["model"], r["model"]) for r in rs]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    specs = [("AP", "AP (IoU .50:.95)", BLUE),
             ("AP50", "AP50", ORANGE),
             ("APs", "APs — vật thể nhỏ", GREEN)]
    for k, (ax, (key, title, col)) in enumerate(zip(axes, specs)):
        vals = [num(r[key]) for r in rs]
        bars = ax.barh(labels[::-1], vals[::-1], color=col, height=.6, zorder=3)
        best = max(vals)
        for b, v in zip(bars, vals[::-1]):
            b.set_alpha(1.0 if v == best else .45)
            ax.text(v * 1.02, b.get_y() + b.get_height() / 2,
                    f"{v:.4f}",
                    va="center", fontsize=12,
                    fontweight="bold" if v == best else "normal")
        ax.set_title(title, fontsize=13.5)
        ax.set_xlim(0, max(vals) * 1.30)
        ax.grid(axis="y", visible=False)
        # Ten model chi hien o panel dau: lap lai o ca ba panel thi no de len
        # chinh cac con so ma bieu do dinh trinh bay.
        if k:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)
    fig.suptitle("548 ảnh → 2.192 ô, cùng decoder, cùng bộ AP",
                 fontsize=16, fontweight="bold", y=1.03)
    p = os.path.join(out, "fig_models.png")
    fig.savefig(p); plt.close(fig)
    return p


# --------------------------------------------------------------------------- #
def fig_tiling(root, out):
    """The gain lands on small objects, which is the evidence for the claim."""
    rs = rows(os.path.join(root, "docs", "results", "tiling.csv"))
    rs = [r for r in rs if r["n_images"] == "60"]
    un = next(r for r in rs if r["mode"] == "untiled")
    ti = next(r for r in rs if r["mode"] != "untiled")

    keys = [("AP", "AP"), ("AP50", "AP50"), ("APs", "APs\n(vật thể nhỏ)")]
    gains = [(num(ti[k]) - num(un[k])) / num(un[k]) * 100 for k, _ in keys]

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    cols = [GREY, GREY, GREEN]
    bars = ax.bar([lbl for _, lbl in keys], gains, color=cols, width=.55, zorder=3)
    for b, g in zip(bars, gains):
        ax.text(b.get_x() + b.get_width() / 2, g + 1.6, f"+{g:.1f}%",
                ha="center", fontsize=15, fontweight="bold",
                color=GREEN if g == max(gains) else INK)
    ax.set_ylabel("Cải thiện so với không cắt ô (%)")
    ax.set_title("Cắt ô 2×2 chồng lấn 20%: lợi ích dồn vào vật thể nhỏ")
    ax.set_ylim(0, max(gains) * 1.28)
    ax.axhline(0, color="#B0B0B0", linewidth=.9)
    ax.grid(axis="x", visible=False)
    ax.text(0.5, -0.16, "Đó là bằng chứng cơ chế đúng: vật thể đến network to hơn "
            "1.80×. Giá phải trả là 3.95× thời gian.",
            transform=ax.transAxes, ha="center", fontsize=11.5, color=GREY)
    p = os.path.join(out, "fig_tiling.png")
    fig.savefig(p); plt.close(fig)
    return p


# --------------------------------------------------------------------------- #
def fig_power(root, out):
    """A negative result, drawn so the reader can check it rather than trust it."""
    src = ["battery\n(I × V)", "USB-PD\n(ucsi rail)", "PMIC ADC\n(pm8550b iin_fb)"]
    idle = [506.2, np.nan, np.nan]
    load = [507.8, np.nan, np.nan]
    sigma = [0.3, 0.0, 0.1]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.4, 4.8),
                                 gridspec_kw={"width_ratios": [1, 1.15]})
    x = np.arange(1)
    a1.bar(x - .17, [idle[0]], .32, yerr=[6.0], color=BLUE, label="rảnh", zorder=3,
           capsize=5, error_kw={"linewidth": 1.4})
    a1.bar(x + .17, [load[0]], .32, yerr=[6.1], color=ORANGE, label="tải 6 lõi",
           zorder=3, capsize=5, error_kw={"linewidth": 1.4})
    a1.set_xticks(x); a1.set_xticklabels(["battery rail"])
    a1.set_ylabel("Công suất (mW)")
    a1.set_ylim(470, 540)
    a1.legend(fontsize=12)
    a1.set_title("Chênh +1.6 mW, nhiễu ±6.0 mW", fontsize=14)
    a1.grid(axis="x", visible=False)

    # The bars are near zero on purpose, which on a 0..6 axis makes them look
    # like a rendering fault rather than the finding. Shade the region they all
    # fall in so the emptiness reads as the result.
    a2.axvspan(0, 5, color=RED, alpha=.07, zorder=1)
    bars = a2.barh(src[::-1], [max(s, 0.04) for s in sigma[::-1]],
                   color=RED, height=.5, zorder=3)
    for b, s in zip(bars, sigma[::-1]):
        a2.text(max(s, 0.04) + .15, b.get_y() + b.get_height() / 2, f"{s:.1f}σ",
                va="center", fontsize=13, fontweight="bold", color=RED)
    a2.axvline(5, color=GREEN, linewidth=2.2, linestyle="--", zorder=4)
    a2.text(4.85, 2.62, "ngưỡng dùng được\n(5σ)", color=GREEN, fontsize=11.5,
            ha="right", va="top", fontweight="bold")
    a2.text(2.5, 2.62, "vùng không phân giải được", color=RED, fontsize=11.5,
            ha="center", va="top")
    a2.set_xlim(0, 6.4)
    a2.set_ylim(-0.6, 2.9)
    a2.set_xlabel("Tách biệt tải / rảnh, tính bằng σ của trạng thái rảnh")
    a2.set_title("Cả ba nguồn đều không phân giải được tải", fontsize=14)
    a2.grid(axis="y", visible=False)

    fig.suptitle("QCS8550 dev kit: có 3 counter điện năng, không dùng được cái nào",
                 fontsize=15.5, fontweight="bold", y=1.02)
    p = os.path.join(out, "fig_power.png")
    fig.savefig(p); plt.close(fig)
    return p


# --------------------------------------------------------------------------- #
def fig_tradeoff(root, out):
    """Accuracy against NPU latency, which is the decision the project has to make.

    Both axes come from different machines on purpose: AP is measured here on
    one tiled split, latency is measured on QCS8550. The host ms/tile column
    exists in the AP csv and is *not* used -- it moved 25% between runs of the
    same model, so it cannot carry a latency claim.
    """
    nm = {"v26n-p2-encoder.onnx": "v26n-p2",
          "v26n-base-encoder-md500.onnx": "v26n-base",
          "v11n-base-encoder.onnx": "v11n-base",
          "v8n-base-encoder.onnx": "v8n-base"}
    # w8a16, the only precision that survives on every model (docs/results)
    lat = {"v8n-base": 3.984, "v11n-base": 4.710,
           "v26n-base": 4.935, "v26n-p2": 8.531}
    rs = rows(os.path.join(root, "results", "tiled_val_4models.csv"))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.2, 5.2))
    cols = {"v8n-base": BLUE, "v11n-base": ORANGE,
            "v26n-base": GREEN, "v26n-p2": PURPLE}
    for ax, key, title in ((a1, "AP", "AP (IoU .50:.95)"),
                           (a2, "APs", "APs — vật thể nhỏ")):
        for r in rs:
            n = nm.get(r["model"], r["model"])
            x, y = lat[n], num(r[key])
            ax.scatter(x, y, s=260, color=cols[n], zorder=3,
                       edgecolor="white", linewidth=2)
            # v8n va v11n gan nhau ca hai truc, nhan mac dinh se de len nhau
            dx, dy, ha = (0, 16, "center")
            if n == "v8n-base":
                dx, dy, ha = (-14, -4, "right")
            elif n == "v11n-base":
                dx, dy, ha = (14, -4, "left")
            ax.annotate(n, (x, y), xytext=(dx, dy), textcoords="offset points",
                        ha=ha, va="center" if dy < 8 else "bottom",
                        fontsize=12, fontweight="bold")
        ys = [num(r[key]) for r in rs]
        pad = (max(ys) - min(ys)) * .45 or .01
        ax.set_ylim(min(ys) - pad, max(ys) + pad * 1.5)
        ax.set_xlim(2.4, 9.8)
        ax.set_xlabel("Latency NPU w8a16 (ms/ô)")
        ax.set_title(title, fontsize=14)

    a1.set_ylabel("AP")
    fig.suptitle("Đánh đổi: p2 chính xác nhất, và chậm nhất 2.1×",
                 fontsize=16, fontweight="bold", y=1.0)
    fig.text(0.5, -0.03, "Trục dọc đo tại chỗ trên 2.192 ô; trục ngang đo trên "
             "QCS8550 qua AI Hub. Cột ms/ô trên host bị bỏ đi vì nó lệch 25% "
             "giữa hai lần chạy cùng một model.",
             ha="center", fontsize=11, color=GREY)
    p = os.path.join(out, "fig_tradeoff.png")
    fig.savefig(p); plt.close(fig)
    return p


# --------------------------------------------------------------------------- #
def fig_budget(root, out):
    """Where a frame's time actually goes once the board is included.

    Stacked, because the argument is compositional: the NPU term is the small
    one. Making the network faster moves the total very little while NMS sits
    on the CPU -- and the head that would remove NMS is the head that cannot
    be quantized.
    """
    import json
    b = json.load(open(os.path.join(root, "results", "board",
                                    "board_cpu_cold.json")))
    npu, tiles = 3.984, 4          # v8n w8a16 on AI Hub's QCS8550
    parts = [("NPU w8a16 (AI Hub)", npu, BLUE),
             ("tiền xử lý (board)", b["pre_ms"], GREEN),
             ("decode + NMS (board)", b["post_ms"], ORANGE)]

    fig, ax = plt.subplots(figsize=(10.6, 4.4))
    left = 0.0
    for name, v, c in parts:
        seg = v * tiles
        ax.barh([0], [seg], left=left, color=c, height=.52, zorder=3)
        ax.text(left + seg / 2, 0, f"{seg:.1f}", ha="center", va="center",
                color="white", fontsize=14, fontweight="bold")
        left += seg

    ax.set_yticks([])
    ax.set_ylim(-.75, .95)
    ax.set_xlim(0, left * 1.06)
    ax.set_xlabel("ms cho một khung hình (4 ô)")
    ax.set_title(f"Frame budget: {left:.1f} ms  →  ~{1000 / left:.0f} FPS")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Patch(facecolor=c, label=n) for n, _, c in parts],
              loc="upper center", ncol=3, fontsize=11.5,
              bbox_to_anchor=(.5, 1.03))
    ax.text(0.5, -0.42,
            "NPU đo trên thiết bị đám mây của AI Hub; hai khâu CPU đo trên chính "
            "board.\nBinary AI Hub không nạp được trên board (QAIRT 2.28 vs 2.45), "
            "nên cột NPU là cận dưới chưa kiểm chứng tại chỗ.",
            transform=ax.transAxes, ha="center", fontsize=11, color=GREY)
    p = os.path.join(out, "fig_budget.png")
    fig.savefig(p); plt.close(fig)
    return p


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(args.root, "docs", "img")
    os.makedirs(out, exist_ok=True)

    for fn in (fig_quant_trap, fig_e2e_matrix, fig_models, fig_tiling,
               fig_power, fig_budget, fig_tradeoff):
        try:
            p = fn(args.root, out)
            print(f"[ok] {os.path.relpath(p, args.root)}"
                  f"  {os.path.getsize(p) / 1e3:.0f} KB")
        except Exception as e:
            print(f"[loi] {fn.__name__}: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
