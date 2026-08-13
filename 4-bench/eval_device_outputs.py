#!/usr/bin/env python3
"""Score raw head outputs that the board produced, against VisDrone ground truth.

    python 4-bench/eval_device_outputs.py --manifest results/device_in/manifest.json \
        --outputs results/device_out --tag fp16-onboard

``qnn-net-run`` writes one ``Result_<i>/output_0.raw`` per input, in input-list
order. This reads them back, decodes each with *its own* letterbox parameters
from the manifest, runs the same NMS the pipeline uses, and evaluates with the
same AP implementation as ``bench_quality.py`` -- so the only difference from
the host row is where the matrix multiplies happened.

Which is the point: it makes the accuracy tables a device measurement instead
of an ONNX Runtime simulation of one. If the two agree, the host tables are
validated and can keep being used for the conditions that are too bulky to
push. If they disagree, the host tables were wrong and this says by how much.

Nothing about the decode is re-implemented here. ``Yolov8Detector.postprocess``
never touches its session, so it is constructed with ``session=None`` and
reused verbatim; a second decoder written for the device path is a second
decoder to keep in sync, and the first divergence would look like a hardware
result.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "3-pipeline"))

from detector import Yolov8Detector  # noqa: E402
from metrics import DetectionEvaluator  # noqa: E402
from visdrone_data import filter_ignored, load_split  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--outputs", required=True,
                    help="directory holding Result_0/, Result_1/, ...")
    ap.add_argument("--data", default=None,
                    help="ground-truth split; defaults to the manifest's")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.65)
    ap.add_argument("--max-det", type=int, default=500)
    ap.add_argument("--nc", type=int, default=10)
    ap.add_argument("--ignore-policy", default="mask", choices=["mask", "keep"])
    ap.add_argument("--tag", default="onboard")
    ap.add_argument("--out", default=os.path.join(ROOT, "results",
                                                  "quality_device.csv"))
    args = ap.parse_args()

    with open(args.manifest, encoding="utf-8") as f:
        man = json.load(f)

    samples = {s.image_id: s for s in load_split(args.data or man["data"])}

    det = Yolov8Detector(session=None, imgsz=man["imgsz"], conf_thres=args.conf,
                         iou_thres=args.iou, max_det=args.max_det)
    ev = DetectionEvaluator(num_classes=args.nc)

    n_det = 0
    n_missing = 0
    n_scored = 0
    for item in man["items"]:
        raw_path = os.path.join(args.outputs, f"Result_{item['index']}",
                                "output_0.raw")
        if not os.path.exists(raw_path):
            n_missing += 1
            continue

        a = np.fromfile(raw_path, dtype=np.float32)
        # (4 + nc) * anchors. Trust the file length rather than a hardcoded
        # 8400: a different imgsz changes the anchor count silently.
        ch = 4 + args.nc
        if a.size % ch:
            print(f"[fatal] {raw_path}: {a.size} floats is not a multiple of "
                  f"{ch}; is --nc right?", file=sys.stderr)
            return 2
        raw = a.reshape(1, ch, a.size // ch)

        s = samples.get(item["image_id"])
        if s is None:
            n_missing += 1
            continue

        boxes, scores, classes = det.postprocess(
            raw, item["gain"], tuple(item["pad"]),
            (item["src_h"], item["src_w"]),
        )
        if args.ignore_policy == "mask":
            boxes, scores, classes = filter_ignored(
                boxes, scores, classes, s.ignore_boxes
            )
        ev.add(s.image_id, boxes, scores, classes, s.boxes, s.classes)
        n_det += len(boxes)
        n_scored += 1

    if not n_scored:
        print("[fatal] nothing scored", file=sys.stderr)
        return 2

    res = ev.evaluate()
    res.update({
        "tag": args.tag,
        "condition": man.get("condition", "clean"),
        "n_images": n_scored,
        "n_missing": n_missing,
        "n_detections": n_det,
        "det_per_image": n_det / n_scored,
        "conf": args.conf,
        "iou": args.iou,
        "max_det": args.max_det,
        "ignore_policy": args.ignore_policy,
        "measured_by": "on-board (QCS8550 Hexagon, qnn-net-run)",
    })

    if n_missing:
        print(f"[warn] {n_missing} outputs missing or unmatched -- AP is over "
              f"the {n_scored} that were scored, not the full split")

    print(f"\n  AP    {res['AP']:.4f}")
    print(f"  AP50  {res['AP50']:.4f}")
    print(f"  APs   {res['APs']:.4f}")
    print(f"  images {n_scored}   detections/image {res['det_per_image']:.1f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    new = not os.path.exists(args.out)
    with open(args.out, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(res.keys()))
        if new:
            w.writeheader()
        w.writerow(res)
    print(f"[ok] appended to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
