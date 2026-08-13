#!/usr/bin/env python3
"""Compose the final demo video: detections plus a live telemetry panel.

Takes the annotated video and the per-frame CSV that ``run_pipeline.py``
already produces, and renders them side by side so a viewer sees the
detections and the state they generate at the same time. That pairing is the
whole product claim -- the UAV does not send video home, it sends counts --
and a demo that shows only boxes does not make it.

    python 3-pipeline/run_pipeline.py --source assets/demo_clip.mp4 \
        --save-video results/demo_raw.mp4 --session-id demo
    python scripts/make_demo_video.py --video results/demo_raw.mp4 \
        --csv results/runtime_telemetry.csv --out results/demo_final.mp4

Nothing here re-runs the model: the panel is drawn from the recorded CSV, so
the numbers on screen are exactly the numbers in the benchmark record.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    raise SystemExit("opencv required")

BG = (18, 20, 25)
FG = (235, 235, 235)
MUTED = (150, 158, 170)
GRID = (45, 50, 60)
BLUE = (250, 165, 96)      # BGR
GREEN = (120, 222, 74)
LEVEL = {"normal": (110, 200, 110), "warn": (0, 200, 255), "alert": (60, 60, 235)}


def load_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(row: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def draw_series(panel, series, x0, y0, w, h, colour, label, maxv=None):
    cv2.rectangle(panel, (x0, y0), (x0 + w, y0 + h), GRID, 1)
    if not series:
        return
    m = maxv if maxv else max(5.0, max(series) * 1.15)
    pts = []
    n = len(series)
    for i, v in enumerate(series):
        px = x0 + int(i * w / max(n - 1, 1))
        py = y0 + h - int(min(v / m, 1.0) * h)
        pts.append((px, py))
    if len(pts) > 1:
        cv2.polylines(panel, [np.asarray(pts, np.int32)], False, colour, 2,
                      cv2.LINE_AA)
    cv2.putText(panel, label, (x0, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                MUTED, 1, cv2.LINE_AA)
    cv2.putText(panel, f"{series[-1]:.0f}", (x0 + w - 42, y0 + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, colour, 2, cv2.LINE_AA)


def build_panel(rows, idx, pw, ph, window=150):
    p = np.full((ph, pw, 3), BG, np.uint8)
    r = rows[idx]
    lo = max(0, idx - window)
    veh = [fnum(x, "vehicle_count") for x in rows[lo:idx + 1]]
    ppl = [fnum(x, "person_count") for x in rows[lo:idx + 1]]
    lat = [fnum(x, "total_ms") for x in rows[lo:idx + 1]]

    y = 26
    cv2.putText(p, "TRUNG TAM GIAM SAT", (16, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, FG, 2, cv2.LINE_AA)
    y += 18
    cv2.putText(p, "du lieu tu node bien", (16, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, MUTED, 1, cv2.LINE_AA)
    y += 22
    cv2.line(p, (16, y), (pw - 16, y), GRID, 1)

    # KPI row
    y += 30
    lvl = (r.get("congestion_level") or "normal").strip()
    col = LEVEL.get(lvl, MUTED)
    txt = {"normal": "BINH THUONG", "warn": "DONG", "alert": "UN TAC"}.get(lvl, lvl)
    cv2.rectangle(p, (16, y - 20), (16 + 150, y + 8), col, -1)
    cv2.putText(p, txt, (26, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (25, 25, 25),
                2, cv2.LINE_AA)

    cv2.putText(p, f"frame {r.get('frame_id', '')}", (pw - 116, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, MUTED, 1, cv2.LINE_AA)

    y += 46
    for lbl, val, colr in (("Phuong tien", veh[-1] if veh else 0, BLUE),
                           ("Nguoi", ppl[-1] if ppl else 0, GREEN),
                           ("Track dang bam", fnum(r, "n_tracks"), FG)):
        cv2.putText(p, lbl, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, MUTED, 1,
                    cv2.LINE_AA)
        cv2.putText(p, f"{val:.0f}", (pw - 70, y + 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.72, colr, 2, cv2.LINE_AA)
        y += 34

    y += 14
    gw, gh = pw - 32, 74
    draw_series(p, veh, 16, y, gw, gh, BLUE, "phuong tien / khung hinh")
    y += gh + 34
    draw_series(p, ppl, 16, y, gw, gh, GREEN, "nguoi / khung hinh")
    y += gh + 34
    draw_series(p, lat, 16, y, gw, gh, (200, 200, 200),
                "do tre (ms/khung hinh)")

    # footer
    cv2.line(p, (16, ph - 54), (pw - 16, ph - 54), GRID, 1)
    cv2.putText(p, "Chi gui trang thai, khong gui video", (16, ph - 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, MUTED, 1, cv2.LINE_AA)
    cv2.putText(p, "SkySentry - QCS8550 edge node", (16, ph - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, MUTED, 1, cv2.LINE_AA)
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="results/demo_final.mp4")
    ap.add_argument("--panel-width", type=int, default=420)
    args = ap.parse_args()

    for f in (args.video, args.csv):
        if not os.path.exists(f):
            print(f"[fatal] missing {f}", file=sys.stderr)
            return 2

    rows = load_csv(args.csv)
    if not rows:
        print("[fatal] telemetry CSV is empty", file=sys.stderr)
        return 2

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[fatal] cannot open {args.video}", file=sys.stderr)
        return 2
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[info] {args.video}: {nfr} frames {vw}x{vh} @ {fps:.1f}")
    print(f"[info] {args.csv}: {len(rows)} telemetry rows")
    if abs(nfr - len(rows)) > 2:
        # Not fatal -- the panel simply clamps -- but a large gap means the
        # CSV and the video came from different runs, and then every number
        # on screen belongs to a different frame than the boxes beside it.
        print(f"[warn] frame/row mismatch ({nfr} vs {len(rows)}); the panel "
              f"may lag the video. Regenerate both from one run.")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    pw = args.panel_width
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (vw + pw, vh))
    if not writer.isOpened():
        print("[fatal] cannot open writer", file=sys.stderr)
        return 2

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        panel = build_panel(rows, min(i, len(rows) - 1), pw, vh)
        writer.write(np.hstack([frame, panel]))
        i += 1
        if i % 100 == 0:
            print(f"  {i}/{nfr}", flush=True)

    cap.release()
    writer.release()
    mb = os.path.getsize(args.out) / 1e6
    print(f"[ok] {args.out}  {i} frames, {i / fps:.1f}s, {mb:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
