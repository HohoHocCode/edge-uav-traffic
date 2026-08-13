#!/usr/bin/env python3
"""Measure what tiled inference actually buys, on the real validation split.

    python 4-bench/bench_tiled.py --model models/yolov8n_visdrone_640.onnx \
        --limit 100 --grid 2 --overlap 0.20

Tiling is usually argued for from first principles -- small objects arrive
larger, therefore accuracy improves. That argument is sound and still needs
checking, because tiling also costs four forward passes, splits objects across
edges, and changes the background statistics each crop presents. Whether the
net effect is positive is an empirical question about a particular model and a
particular dataset.

The same model, the same images and the same AP code are used for both rows, so
the only difference is the inference strategy. APs (small objects) is reported
separately because that is where tiling is supposed to pay, and a gain that
shows up in AP but not APs would mean something other than the stated mechanism
is happening.
"""

from __future__ import annotations

import argparse
import csv
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
from tiled_detector import TiledDetector, tile_rects  # noqa: E402
from visdrone_data import filter_ignored, load_split  # noqa: E402


def run(det, samples, tag: str, ignore_policy: str = "mask") -> dict:
    ev = DetectionEvaluator(num_classes=10)
    n_det = 0
    t0 = time.perf_counter()
    lat = []
    for i, s in enumerate(samples):
        img = cv2.imread(s.image_path)
        if img is None:
            continue
        d = det(img)
        boxes, scores, classes = d.xyxy, d.conf, d.cls
        if ignore_policy == "mask":
            boxes, scores, classes = filter_ignored(
                boxes, scores, classes, s.ignore_boxes)
        ev.add(s.image_id, boxes, scores, classes, s.boxes, s.classes)
        n_det += len(boxes)
        lat.append(d.total_ms)
        if (i + 1) % 25 == 0:
            print(f"    [{tag}] {i + 1}/{len(samples)}", flush=True)

    res = ev.evaluate()
    res.update({
        "tag": tag,
        "n_images": len(lat),
        "det_per_image": n_det / max(len(lat), 1),
        "ms_per_image": float(np.mean(lat)),
        "wall_s": time.perf_counter() - t0,
    })
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default=os.path.join(ROOT, "data",
                                                   "VisDrone2019-DET-val"))
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.65)
    ap.add_argument("--max-det", type=int, default=500)
    ap.add_argument("--grid", type=int, default=2)
    ap.add_argument("--overlap", type=float, default=0.20)
    ap.add_argument("--merge-iou", type=float, default=0.55)
    ap.add_argument("--keep-truncated", action="store_true",
                    help="do NOT drop detections on interior tile edges; use "
                         "to measure what that rule is worth")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "tiled.csv"))
    args = ap.parse_args()

    samples = load_split(args.data)
    if args.limit:
        samples = samples[: args.limit]
    print(f"[info] {len(samples)} images from {args.data}")

    sess = create_session(args.model, backend=args.backend)
    base = Yolov8Detector(sess, imgsz=args.imgsz, conf_thres=args.conf,
                          iou_thres=args.iou, max_det=args.max_det)
    tiled = TiledDetector(base, grid=args.grid, overlap=args.overlap,
                          merge_iou=args.merge_iou,
                          drop_truncated=not args.keep_truncated,
                          max_det=args.max_det)

    h, w = cv2.imread(samples[0].image_path).shape[:2]
    rects = tile_rects(w, h, args.grid, args.overlap)
    print(f"[info] {w}x{h} -> {len(rects)} tiles of "
          f"{rects[0][2]}x{rects[0][3]}, objects arrive "
          f"{tiled.scale_gain(w, h, args.imgsz):.2f}x larger")

    rows = [run(base, samples, "untiled"),
            run(tiled, samples, f"tiled-{args.grid}x{args.grid}-ov{args.overlap:g}")]

    print(f"\n{'':<22}{'AP':>8}{'AP50':>8}{'APs':>8}{'det/img':>9}{'ms/img':>9}")
    for r in rows:
        print(f"{r['tag']:<22}{r['AP']:>8.4f}{r['AP50']:>8.4f}{r['APs']:>8.4f}"
              f"{r['det_per_image']:>9.1f}{r['ms_per_image']:>9.1f}")

    a, b = rows
    def rel(k):
        return (b[k] - a[k]) / a[k] * 100 if a[k] else float("nan")
    print(f"\n{'delta':<22}{rel('AP'):>7.1f}%{rel('AP50'):>7.1f}%"
          f"{rel('APs'):>7.1f}%{'':>9}{b['ms_per_image']/a['ms_per_image']:>8.2f}x")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    new = not os.path.exists(args.out)
    with open(args.out, "a", newline="", encoding="utf-8") as f:
        w_ = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            w_.writeheader()
        w_.writerows(rows)
    print(f"\n[ok] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
