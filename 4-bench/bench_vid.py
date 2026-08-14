#!/usr/bin/env python3
"""Frame-level detection diagnostic on the complete VisDrone-VID test split.

This measures the detector under the VID image distribution, but it is *not*
the complete Task 2 benchmark because it does not keep tracker state. Use
``bench_tracking.py --data <VID-root> --class-policy all10`` for the primary
VID result. Unlike Task 1, this diagnostic does not subsample: every frame is
evaluated with the COCO-style AP implementation used by ``bench_quality.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "3-pipeline"))

import cv2  # noqa: E402

from benchmark_detector import (  # noqa: E402
    create_benchmark_detector,
    warmup_benchmark_detector,
)
from metrics import DetectionEvaluator  # noqa: E402
from mot_data import list_sequences, load_sequence  # noqa: E402
from table_io import append_rows  # noqa: E402
from visdrone_data import CLASS_NAMES, filter_ignored  # noqa: E402


def sha256_short(path: str, n: int = 16) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:n]


def latency(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return float("nan"), float("nan"), float("nan")
    array = np.asarray(values, dtype=np.float64)
    return (
        float(array.mean()),
        float(np.percentile(array, 50)),
        float(np.percentile(array, 95)),
    )


def result_row(
    label: str,
    evaluator: DetectionEvaluator,
    detections: int,
    timings: dict[str, list[float]],
) -> dict:
    metrics = evaluator.evaluate()
    pre_avg, _, pre_p95 = latency(timings["pre"])
    infer_avg, infer_p50, infer_p95 = latency(timings["infer"])
    post_avg, _, post_p95 = latency(timings["post"])
    row = {
        "sequence": label,
        "n_frames": metrics["n_images"],
        "n_detections": detections,
        "det_per_frame": detections / max(metrics["n_images"], 1),
        "AP": metrics["AP"],
        "AP50": metrics["AP50"],
        "AP75": metrics["AP75"],
        "APs": metrics["APs"],
        "APm": metrics["APm"],
        "APl": metrics["APl"],
        "pre_ms_avg": pre_avg,
        "pre_ms_p95": pre_p95,
        "infer_ms_avg": infer_avg,
        "infer_ms_p50": infer_p50,
        "infer_ms_p95": infer_p95,
        "post_ms_avg": post_avg,
        "post_ms_p95": post_p95,
        "total_ms_avg": pre_avg + infer_avg + post_avg,
    }
    for index, name in enumerate(CLASS_NAMES):
        row[f"AP_{name}"] = metrics["per_class_AP"].get(index, float("nan"))
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default=os.path.join(ROOT, "test", "VisDrone2019-VID-test-dev"),
    )
    parser.add_argument(
        "--model",
        default=os.path.join(ROOT, "models", "yolov8n_visdrone_640.onnx"),
    )
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--device", default="0")
    parser.add_argument("--half", action="store_true", default=True)
    parser.add_argument("--no-half", dest="half", action="store_false")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--conf", type=float, default=0.001,
        help="low threshold required to retain the complete precision-recall curve",
    )
    parser.add_argument("--iou", type=float, default=0.65)
    parser.add_argument("--max-det", type=int, default=500)
    parser.add_argument("--sequences", nargs="*", default=None)
    parser.add_argument(
        "--limit-frames", type=int, default=0,
        help="0 = all frames; otherwise limit each selected sequence",
    )
    parser.add_argument("--ignore-policy", choices=("mask", "keep"), default="mask")
    parser.add_argument("--ignore-overlap", type=float, default=0.5)
    parser.add_argument("--progress-every", type=int, default=200)
    parser.add_argument(
        "--per-sequence", action="store_true",
        help="also emit AP per sequence; slower because each sequence is evaluated separately",
    )
    parser.add_argument(
        "--out", default=os.path.join(ROOT, "results", "vid_detection_quality.csv")
    )
    parser.add_argument("--tag", default="")
    args = parser.parse_args(argv)

    if not os.path.isfile(args.model):
        raise SystemExit(f"[fatal] model not found: {args.model}")
    names = list_sequences(args.data)
    if args.sequences:
        unknown = sorted(set(args.sequences) - set(names))
        if unknown:
            raise SystemExit(f"[fatal] unavailable sequence(s): {unknown}")
        names = [name for name in names if name in args.sequences]
    if not names:
        raise SystemExit(f"[fatal] no scorable sequences under {args.data}")

    sequences = [
        load_sequence(args.data, name, limit=args.limit_frames or None)
        for name in names
    ]
    print(
        f"[info] Task 2 VID: {len(sequences)} sequences, "
        f"{sum(len(sequence) for sequence in sequences)} full frames"
    )

    detector, backend_name, providers_active, class_names = create_benchmark_detector(
        args.model, args.backend, args.imgsz, args.conf, args.iou, args.max_det,
        device=args.device, half=args.half,
    )
    if class_names and list(class_names.values()) != CLASS_NAMES:
        raise SystemExit(
            f"[fatal] checkpoint class order differs from VisDrone: {class_names}"
        )
    warmup_benchmark_detector(detector, sequences[0].frames[0].path, n=5)
    print(f"[info] backend={backend_name} providers={providers_active}")

    overall_evaluator = DetectionEvaluator(num_classes=len(CLASS_NAMES))
    overall_timings: dict[str, list[float]] = defaultdict(list)
    overall_detections = 0
    rows: list[dict] = []

    for sequence in sequences:
        evaluator = (
            DetectionEvaluator(num_classes=len(CLASS_NAMES))
            if args.per_sequence else None
        )
        timings: dict[str, list[float]] = defaultdict(list)
        n_detections = 0
        for position, frame in enumerate(sequence.frames, start=1):
            image = cv2.imread(frame.path)
            if image is None:
                print(f"[warn] unreadable image: {frame.path}", file=sys.stderr)
                continue
            prediction = detector(image)
            boxes, scores, classes = prediction.xyxy, prediction.conf, prediction.cls
            if args.ignore_policy == "mask":
                boxes, scores, classes = filter_ignored(
                    boxes, scores, classes, frame.ignore_boxes, args.ignore_overlap
                )

            image_id = f"{sequence.name}/{frame.index:07d}"
            targets = (overall_evaluator,) if evaluator is None else (
                evaluator, overall_evaluator
            )
            for target in targets:
                target.add(
                    image_id,
                    boxes,
                    scores,
                    classes,
                    frame.boxes,
                    frame.classes,
                )
            n_detections += len(boxes)
            overall_detections += len(boxes)
            for key, value in (
                ("pre", prediction.pre_ms),
                ("infer", prediction.infer_ms),
                ("post", prediction.post_ms),
            ):
                timings[key].append(value)
                overall_timings[key].append(value)
            if args.progress_every and position % args.progress_every == 0:
                print(
                    f"    {sequence.name}: {position}/{len(sequence.frames)} frames",
                    flush=True,
                )

        if evaluator is not None:
            row = result_row(sequence.name, evaluator, n_detections, timings)
            rows.append(row)
            print(
                f"[seq] {sequence.name}: AP={row['AP']:.4f} "
                f"AP50={row['AP50']:.4f} frames={row['n_frames']}"
            )

    overall = result_row(
        "OVERALL", overall_evaluator, overall_detections, overall_timings
    )
    rows.append(overall)

    common = {
        "model": os.path.basename(args.model),
        "model_sha256_16": sha256_short(args.model),
        "backend": backend_name,
        "imgsz": args.imgsz,
        "conf_thres": args.conf,
        "iou_thres": args.iou,
        "max_det": args.max_det,
        "ignore_policy": args.ignore_policy,
        "n_sequences": len(sequences),
        "frame_policy": "all",
        "tag": args.tag,
    }
    for row in rows:
        row.update(common)

    print(
        f"[overall] AP={overall['AP']:.4f} AP50={overall['AP50']:.4f} "
        f"APs={overall['APs']:.4f} frames={overall['n_frames']}"
    )
    print(f"[ok] appended {append_rows(args.out, rows)} row(s) -> {args.out}")
    json_path = os.path.splitext(args.out)[0] + ".json"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, default=str)
    print(f"[ok] {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
