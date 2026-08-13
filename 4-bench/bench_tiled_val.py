#!/usr/bin/env python3
"""Score several models on an identically tiled validation split.

    python 4-bench/bench_tiled_val.py --models models/new/*.onnx --limit 200

The four architectures were trained in four Colab tabs, and each tab reported
its own mAP from its own run. Those figures are only comparable if nothing
differed but the architecture, which is exactly the assumption that keeps
turning out to be false in this project. This re-scores every model here, on
one tiled split, with one decoder and one AP implementation, so the only
variable left is the model file.

Tiles are produced in memory with the same geometry the training set used
(2x2, 20% overlap, a box kept when >= MIN_VIS of its area survives the cut).
Ignore regions are cut to the tile the same way and masked before scoring,
because a detection inside an ignored region is neither a hit nor a miss.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "3-pipeline"))

import cv2  # noqa: E402

from detector import Yolov8Detector  # noqa: E402
from metrics import DetectionEvaluator  # noqa: E402
from runtime import create_session  # noqa: E402
from tiled_detector import tile_rects  # noqa: E402
from visdrone_data import load_split  # noqa: E402


def clip_boxes(boxes, classes, tx, ty, tw, th, min_vis):
    """Cut boxes to a tile, keeping those with enough area left."""
    if len(boxes) == 0:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.int32)
    b = np.asarray(boxes, np.float32)
    ix1 = np.maximum(b[:, 0], tx)
    iy1 = np.maximum(b[:, 1], ty)
    ix2 = np.minimum(b[:, 2], tx + tw)
    iy2 = np.minimum(b[:, 3], ty + th)
    w = np.clip(ix2 - ix1, 0, None)
    h = np.clip(iy2 - iy1, 0, None)
    area = np.clip((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]), 1e-9, None)
    keep = (w * h) / area >= min_vis
    out = np.stack([ix1 - tx, iy1 - ty, ix2 - tx, iy2 - ty], 1)[keep]
    return out.astype(np.float32), np.asarray(classes, np.int32)[keep]


def mask_ignored(xyxy, conf, cls, ign):
    """Drop detections whose centre falls inside an ignored region."""
    if len(xyxy) == 0 or len(ign) == 0:
        return xyxy, conf, cls
    cx = (xyxy[:, 0] + xyxy[:, 2]) / 2
    cy = (xyxy[:, 1] + xyxy[:, 3]) / 2
    bad = np.zeros(len(xyxy), bool)
    for x1, y1, x2, y2 in ign:
        bad |= (cx >= x1) & (cx <= x2) & (cy >= y1) & (cy <= y2)
    return xyxy[~bad], conf[~bad], cls[~bad]


def run(model_path, samples, args) -> dict:
    sess = create_session(model_path, backend=args.backend)
    det = Yolov8Detector(sess, imgsz=args.imgsz, conf_thres=args.conf,
                         iou_thres=args.iou, max_det=args.max_det)
    ev = DetectionEvaluator(num_classes=args.nc)
    n_det = n_tile = 0
    lat = []
    t0 = time.perf_counter()

    for i, s in enumerate(samples):
        img = cv2.imread(s.image_path)
        if img is None:
            continue
        H, W = img.shape[:2]
        for ti, (tx, ty, tw, th) in enumerate(
                tile_rects(W, H, args.grid, args.overlap)):
            gt, gcls = clip_boxes(s.boxes, s.classes, tx, ty, tw, th, args.min_vis)
            ign, _ = clip_boxes(s.ignore_boxes,
                                np.zeros(len(s.ignore_boxes), np.int32),
                                tx, ty, tw, th, 0.01)
            crop = img[ty:ty + th, tx:tx + tw]
            d = det(crop)
            xyxy, conf, cls = mask_ignored(d.xyxy, d.conf, d.cls, ign)
            ev.add(f"{s.image_id}_t{ti}", xyxy, conf, cls, gt, gcls)
            n_det += len(xyxy)
            n_tile += 1
            lat.append(d.total_ms)
        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{len(samples)} anh", flush=True)

    res = ev.evaluate()
    res.update({
        "model": os.path.basename(model_path),
        "n_tiles": n_tile,
        "det_per_tile": n_det / max(n_tile, 1),
        "ms_per_tile": float(np.mean(lat)),
        "wall_s": round(time.perf_counter() - t0, 1),
    })
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--data", default=os.path.join(ROOT, "data",
                                                   "VisDrone2019-DET-val"))
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.65)
    ap.add_argument("--max-det", type=int, default=500)
    ap.add_argument("--nc", type=int, default=10)
    ap.add_argument("--grid", type=int, default=2)
    ap.add_argument("--overlap", type=float, default=0.20)
    ap.add_argument("--min-vis", type=float, default=0.40)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "results",
                                                  "tiled_val.csv"))
    args = ap.parse_args()

    paths = []
    for m in args.models:
        paths.extend(sorted(glob.glob(m)) or [m])
    samples = load_split(args.data)
    if args.limit:
        samples = samples[: args.limit]
    print(f"[info] {len(samples)} anh -> {len(samples) * args.grid ** 2} o, "
          f"{len(paths)} model\n")

    rows = []
    for p in paths:
        print(f"[run] {os.path.basename(p)}")
        rows.append(run(p, samples, args))
        r = rows[-1]
        print(f"      AP {r['AP']:.4f}  AP50 {r['AP50']:.4f}  APs {r['APs']:.4f}"
              f"  ({r['wall_s']:.0f}s)\n")

    print(f"{'model':34s}{'AP':>8}{'AP50':>8}{'APs':>8}{'det/o':>8}{'ms/o':>8}")
    for r in sorted(rows, key=lambda x: -x["AP"]):
        print(f"{r['model']:34s}{r['AP']:>8.4f}{r['AP50']:>8.4f}"
              f"{r['APs']:>8.4f}{r['det_per_tile']:>8.1f}{r['ms_per_tile']:>8.1f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n[ok] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
