#!/usr/bin/env python3
"""Quality + robustness benchmark on VisDrone-DET val.

Runs the detector over the validation split once per degradation condition and
reports AP / AP50 / APs, the per-stage latency split, and how much of each is
lost relative to the clean reference.

    python 4-bench/bench_quality.py --model models/yolov8n_visdrone_640.onnx \
        --data data/VisDrone2019-DET-val --conditions clean rain_medium blur_light

Every row is written to ``results/quality.csv``. The row carries the model
hash, the backend, the condition id and the ignore-region policy, because a
number without those four things cannot be compared to anything.

Note on the ignore policy: ``--ignore-policy mask`` drops detections that fall
inside a VisDrone ignored region before scoring; ``keep`` counts them as false
positives, which is what the common conversion scripts do implicitly. Both are
reported so the report can state which convention it quotes rather than
leaving the reader to guess.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "3-pipeline"))
sys.path.insert(0, os.path.join(ROOT, "2-augment"))

import cv2  # noqa: E402

import degradations as D  # noqa: E402
from detector import Yolov8Detector  # noqa: E402
from metrics import DetectionEvaluator  # noqa: E402
from runtime import create_session  # noqa: E402
from visdrone_data import CLASS_NAMES, filter_ignored, load_split  # noqa: E402


def sha256_short(path: str, n: int = 16) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def run_condition(
    det: Yolov8Detector,
    samples,
    condition: str,
    ignore_policy: str,
    progress_every: int = 50,
) -> dict:
    ev = DetectionEvaluator(num_classes=10)
    pre, inf, post = [], [], []
    n_det_total = 0
    t_wall = time.perf_counter()

    for i, s in enumerate(samples):
        img = cv2.imread(s.image_path)
        if img is None:
            continue
        if condition != "clean":
            # Seeded per image index: the degraded set is reproducible run to
            # run, and bit-exact for a given OpenCV build.
            img = D.apply_condition(img, condition, seed=i)

        d = det(img)
        boxes, scores, classes = d.xyxy, d.conf, d.cls

        if ignore_policy == "mask":
            boxes, scores, classes = filter_ignored(
                boxes, scores, classes, s.ignore_boxes
            )

        ev.add(s.image_id, boxes, scores, classes, s.boxes, s.classes)
        n_det_total += len(boxes)
        pre.append(d.pre_ms)
        inf.append(d.infer_ms)
        post.append(d.post_ms)

        if progress_every and (i + 1) % progress_every == 0:
            print(f"    [{condition}] {i + 1}/{len(samples)} images", flush=True)

    res = ev.evaluate()
    wall = time.perf_counter() - t_wall

    def stats(a):
        a = np.asarray(a)
        return (float(a.mean()), float(np.percentile(a, 50)), float(np.percentile(a, 95)))

    pre_a, pre_50, pre_95 = stats(pre)
    inf_a, inf_50, inf_95 = stats(inf)
    pos_a, pos_50, pos_95 = stats(post)

    res.update({
        "condition": condition,
        "ignore_policy": ignore_policy,
        "n_detections": n_det_total,
        "det_per_image": n_det_total / max(len(samples), 1),
        "pre_ms_avg": pre_a, "pre_ms_p95": pre_95,
        "infer_ms_avg": inf_a, "infer_ms_p50": inf_50, "infer_ms_p95": inf_95,
        "post_ms_avg": pos_a, "post_ms_p95": pos_95,
        "total_ms_avg": pre_a + inf_a + pos_a,
        "wall_s": wall,
    })
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "VisDrone2019-DET-val"))
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001,
                    help="low for AP; raise only for a deployment-threshold row")
    ap.add_argument("--iou", type=float, default=0.65)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--limit", type=int, default=0, help="0 = all images")
    ap.add_argument("--conditions", nargs="*", default=["clean"])
    ap.add_argument("--all-conditions", action="store_true")
    ap.add_argument("--ignore-policy", default="mask", choices=["mask", "keep"])
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "quality.csv"))
    ap.add_argument("--tag", default="", help="free-text label for this run")
    args = ap.parse_args()

    conditions = D.CONDITION_IDS if args.all_conditions else args.conditions
    for c in conditions:
        if c not in D.CONDITION_IDS:
            print(f"[fatal] unknown condition {c!r}; known: {D.CONDITION_IDS}",
                  file=sys.stderr)
            return 2

    samples = load_split(args.data, limit=args.limit or None)
    print(f"[info] {len(samples)} images from {args.data}")

    session = create_session(args.model, backend=args.backend)
    det = Yolov8Detector(session, imgsz=args.imgsz, conf_thres=args.conf,
                         iou_thres=args.iou, max_det=args.max_det)
    det.warmup(n=5)
    print(f"[info] backend={session.backend_name} "
          f"providers={session.info.providers_active}")

    model_hash = sha256_short(args.model)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    rows = []
    clean_ref: dict | None = None
    for cond in conditions:
        print(f"[run] condition={cond}")
        session.reset_timers()
        r = run_condition(det, samples, cond, args.ignore_policy)

        if cond == "clean":
            clean_ref = r
        r["AP_retention"] = (
            r["AP"] / clean_ref["AP"] if clean_ref and clean_ref["AP"] > 0 else float("nan")
        )
        r["APs_retention"] = (
            r["APs"] / clean_ref["APs"] if clean_ref and clean_ref["APs"] > 0 else float("nan")
        )
        r.update({
            "model": os.path.basename(args.model),
            "model_sha256_16": model_hash,
            "backend": session.backend_name,
            "imgsz": args.imgsz,
            "conf_thres": args.conf,
            "iou_thres": args.iou,
            "n_images": len(samples),
            "tag": args.tag,
        })
        rows.append(r)

        print(f"    AP {r['AP']:.4f} | AP50 {r['AP50']:.4f} | APs {r['APs']:.4f} "
              f"| APm {r['APm']:.4f} | APl {r['APl']:.4f}")
        print(f"    latency avg pre {r['pre_ms_avg']:.1f} / infer "
              f"{r['infer_ms_avg']:.1f} / post {r['post_ms_avg']:.1f} ms "
              f"| {r['det_per_image']:.1f} det/img | {r['wall_s']:.0f}s")

    # -- write csv ------------------------------------------------------
    flat = []
    for r in rows:
        row = {k: v for k, v in r.items() if k != "per_class_AP"}
        for ci, cname in enumerate(CLASS_NAMES):
            row[f"AP_{cname}"] = r["per_class_AP"].get(ci, float("nan"))
        flat.append(row)

    write_header = not os.path.exists(args.out) or os.path.getsize(args.out) == 0
    with open(args.out, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0].keys()), extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerows(flat)
    print(f"[ok] appended {len(flat)} row(s) -> {args.out}")

    json_out = os.path.splitext(args.out)[0] + ".json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"[ok] {json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
