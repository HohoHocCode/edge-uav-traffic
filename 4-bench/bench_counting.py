#!/usr/bin/env python3
"""Per-class crowd/object counting benchmark on every VisDrone-VID frame.

Ground truth comes from ``test/VisDrone2019-CC-test-dev/counts_by_frame.csv``.
Predictions are detector counts after the same ignore-region policy used by
the DET, VID and MOT benchmarks.  The primary counting metrics are MAE and
RMSE; WAPE is included because class-wise MAPE is undefined on empty frames.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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
from mot_data import list_sequences, load_sequence  # noqa: E402
from table_io import append_rows  # noqa: E402
from visdrone_data import CLASS_NAMES, filter_ignored  # noqa: E402


def sha256_short(path: str, n: int = 16) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:n]


def load_count_ground_truth(path: str) -> dict[tuple[str, int], np.ndarray]:
    result: dict[tuple[str, int], np.ndarray] = {}
    with open(path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [name for name in CLASS_NAMES if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"count ground truth is missing columns: {missing}")
        for row in reader:
            key = (row["sequence"], int(row["frame_index"]))
            if key in result:
                raise ValueError(f"duplicate count ground-truth row: {key}")
            result[key] = np.asarray([int(row[name]) for name in CLASS_NAMES], np.int64)
    return result


def count_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    error = prediction - target
    absolute = np.abs(error)
    return {
        "MAE": float(absolute.mean()),
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "bias": float(error.mean()),
        "WAPE": float(absolute.sum() / target.sum()) if target.sum() else float("nan"),
        "gt_instances": int(target.sum()),
        "pred_instances": int(prediction.sum()),
    }


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values), q)) if values else float("nan")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--counts",
        default=os.path.join(
            ROOT, "test", "VisDrone2019-CC-test-dev", "counts_by_frame.csv"
        ),
    )
    parser.add_argument(
        "--vid-data",
        default=os.path.join(ROOT, "test", "VisDrone2019-VID-test-dev"),
        help="full frames/annotations, used for images and ignored regions",
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
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--max-det", type=int, default=500)
    parser.add_argument("--sequences", nargs="*", default=None)
    parser.add_argument(
        "--limit-frames", type=int, default=0,
        help="0 = all frames; otherwise limit the benchmark globally",
    )
    parser.add_argument("--ignore-policy", choices=("mask", "keep"), default="mask")
    parser.add_argument("--ignore-overlap", type=float, default=0.5)
    parser.add_argument("--progress-every", type=int, default=200)
    parser.add_argument(
        "--out", default=os.path.join(ROOT, "results", "counting_quality.csv")
    )
    parser.add_argument(
        "--predictions-out",
        default=os.path.join(ROOT, "results", "counting_predictions.csv"),
    )
    parser.add_argument("--tag", default="")
    args = parser.parse_args(argv)

    if not os.path.isfile(args.model):
        raise SystemExit(f"[fatal] model not found: {args.model}")
    if not os.path.isfile(args.counts):
        raise SystemExit(
            f"[fatal] count ground truth not found: {args.counts}\n"
            "        run scripts/prepare_test_benchmark.py first"
        )

    ground_truth = load_count_ground_truth(args.counts)
    names = list_sequences(args.vid_data)
    if args.sequences:
        unknown = sorted(set(args.sequences) - set(names))
        if unknown:
            raise SystemExit(f"[fatal] unavailable sequence(s): {unknown}")
        names = [name for name in names if name in args.sequences]
    sequences = [load_sequence(args.vid_data, name) for name in names]
    frames = [(sequence.name, frame) for sequence in sequences for frame in sequence.frames]
    if args.limit_frames:
        frames = frames[: args.limit_frames]
    if not frames:
        raise SystemExit("[fatal] no frames selected")
    selected_sequence_count = len({sequence for sequence, _ in frames})

    # The generated count table and the box annotations must describe exactly
    # the same class population before model inference starts.
    for sequence, frame in frames:
        key = (sequence, frame.index)
        if key not in ground_truth:
            raise SystemExit(f"[fatal] count ground truth has no row for {key}")
        from_boxes = np.bincount(frame.classes, minlength=len(CLASS_NAMES))
        if not np.array_equal(from_boxes, ground_truth[key]):
            raise SystemExit(f"[fatal] count/box ground truth mismatch at {key}")

    print(
        f"[info] Task 5 Crowd Counting: {len(frames)} full frames, "
        f"{selected_sequence_count} sequences, {len(CLASS_NAMES)} classes"
    )
    detector, backend_name, providers_active, class_names = create_benchmark_detector(
        args.model, args.backend, args.imgsz, args.conf, args.iou, args.max_det,
        device=args.device, half=args.half,
    )
    if class_names and list(class_names.values()) != CLASS_NAMES:
        raise SystemExit(
            f"[fatal] checkpoint class order differs from VisDrone: {class_names}"
        )
    warmup_benchmark_detector(detector, frames[0][1].path, n=5)
    print(f"[info] backend={backend_name} providers={providers_active}")

    gt_rows: list[np.ndarray] = []
    pred_rows: list[np.ndarray] = []
    timing: dict[str, list[float]] = defaultdict(list)
    prediction_rows: list[dict] = []
    for position, (sequence, frame) in enumerate(frames, start=1):
        image = cv2.imread(frame.path)
        if image is None:
            print(f"[warn] unreadable image: {frame.path}", file=sys.stderr)
            continue
        detection = detector(image)
        boxes, scores, classes = detection.xyxy, detection.conf, detection.cls
        if args.ignore_policy == "mask":
            boxes, scores, classes = filter_ignored(
                boxes, scores, classes, frame.ignore_boxes, args.ignore_overlap
            )
        valid_classes = classes[(classes >= 0) & (classes < len(CLASS_NAMES))]
        predicted = np.bincount(valid_classes, minlength=len(CLASS_NAMES)).astype(np.int64)
        target = ground_truth[(sequence, frame.index)]
        gt_rows.append(target)
        pred_rows.append(predicted)
        timing["pre"].append(detection.pre_ms)
        timing["infer"].append(detection.infer_ms)
        timing["post"].append(detection.post_ms)

        row = {"sequence": sequence, "frame_index": frame.index}
        for index, name in enumerate(CLASS_NAMES):
            row[f"gt_{name}"] = int(target[index])
            row[f"pred_{name}"] = int(predicted[index])
            row[f"error_{name}"] = int(predicted[index] - target[index])
        row["gt_total"] = int(target.sum())
        row["pred_total"] = int(predicted.sum())
        row["error_total"] = int(predicted.sum() - target.sum())
        prediction_rows.append(row)
        if args.progress_every and position % args.progress_every == 0:
            print(f"    {position}/{len(frames)} frames", flush=True)

    if not gt_rows:
        raise SystemExit("[fatal] no readable frames")
    targets = np.stack(gt_rows)
    predictions = np.stack(pred_rows)

    rows: list[dict] = []
    for index, name in enumerate(CLASS_NAMES):
        rows.append({"class_id": index, "class_name": name, **count_metrics(
            targets[:, index], predictions[:, index]
        )})

    class_rows = rows.copy()
    rows.append({
        "class_id": -1,
        "class_name": "ALL_CLASSES_MACRO",
        "MAE": float(np.mean([row["MAE"] for row in class_rows])),
        "RMSE": float(np.mean([row["RMSE"] for row in class_rows])),
        "bias": float(np.mean([row["bias"] for row in class_rows])),
        "WAPE": float(np.mean([
            row["WAPE"] for row in class_rows if not math.isnan(row["WAPE"])
        ])),
        "gt_instances": int(targets.sum()),
        "pred_instances": int(predictions.sum()),
    })
    rows.append({
        "class_id": -1,
        "class_name": "TOTAL_PER_FRAME",
        **count_metrics(targets.sum(axis=1), predictions.sum(axis=1)),
    })

    pre_avg = float(np.mean(timing["pre"]))
    infer_avg = float(np.mean(timing["infer"]))
    post_avg = float(np.mean(timing["post"]))
    common = {
        "n_frames": len(gt_rows),
        "model": os.path.basename(args.model),
        "model_sha256_16": sha256_short(args.model),
        "backend": backend_name,
        "imgsz": args.imgsz,
        "conf_thres": args.conf,
        "iou_thres": args.iou,
        "max_det": args.max_det,
        "ignore_policy": args.ignore_policy,
        "pre_ms_avg": pre_avg,
        "infer_ms_avg": infer_avg,
        "infer_ms_p95": percentile(timing["infer"], 95),
        "post_ms_avg": post_avg,
        "total_ms_avg": pre_avg + infer_avg + post_avg,
        "tag": args.tag,
    }
    for row in rows:
        row.update(common)

    print(f"[ok] appended {append_rows(args.out, rows)} row(s) -> {args.out}")
    json_path = os.path.splitext(args.out)[0] + ".json"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, default=str)
    print(f"[ok] {json_path}")

    if args.predictions_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.predictions_out)), exist_ok=True)
        with open(args.predictions_out, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0].keys()))
            writer.writeheader()
            writer.writerows(prediction_rows)
        print(f"[ok] {args.predictions_out}")

    total_row = rows[-1]
    print(
        f"[overall] total/frame MAE={total_row['MAE']:.3f} "
        f"RMSE={total_row['RMSE']:.3f} WAPE={total_row['WAPE']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
