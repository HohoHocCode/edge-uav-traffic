#!/usr/bin/env python3
"""Tracker quality benchmark on VisDrone-MOT test-dev.

Runs one or more Ultralytics trackers over every sequence that has both frames
and ground truth, and reports IDF1 / MOTA / IDSW so a tracker choice is decided
by measurement rather than by watching a rendered video.

    python 4-bench/bench_tracking.py --trackers botsort ocsort deepocsort

Each tracker gets its own row set in ``results/tracking_quality.csv`` plus the
raw per-frame hypotheses under ``results/mot/<tracker>/<sequence>.txt`` in
MOTChallenge format, so a metric can be recomputed without re-running the GPU.

Three conventions are declared per row, because a MOT number without them is
not comparable to anything:

* ``class_policy`` -- ``mot5`` scores the five categories of the official
  VisDrone MOT challenge (pedestrian, car, van, truck, bus); ``all10`` scores
  the full taxonomy. Non-scored classes become ignore regions, never
  background.
* ``ignore_policy`` -- ``mask`` drops hypotheses inside a VisDrone ignored
  region before scoring, ``keep`` counts them as false positives.
* ``iou_thresh`` -- the association gate, 0.5 by default.

Association is class-agnostic in both policies; see ``mot_metrics`` for why.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import cv2  # noqa: E402

from mot_data import (  # noqa: E402
    MOT_EVAL_CLASSES,
    filter_classes,
    list_sequences,
    load_sequence,
    write_mot_results,
)
from mot_metrics import (  # noqa: E402
    FORMAT_HEADER,
    MotAccumulator,
    MotCounts,
    format_row,
    summarize,
)
from table_io import append_rows  # noqa: E402
from visdrone_data import keep_mask  # noqa: E402

# Named shorthands resolve to Ultralytics' shipped configs. Anything else is
# treated as a path, which is how the tuned variants in configs/trackers are
# used without teaching this script about each one.
BUILTIN_TRACKERS = (
    "botsort", "bytetrack", "ocsort", "deepocsort", "tracktrack", "fasttrack",
)


def sha256_short(path: str, n: int = 16) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def resolve_tracker(name: str) -> tuple[str, str]:
    """``(label, config)`` for a shorthand name or a path to a YAML."""
    if name in BUILTIN_TRACKERS:
        return name, f"{name}.yaml"
    path = name if os.path.isfile(name) else os.path.join(ROOT, name)
    if not os.path.isfile(path):
        raise SystemExit(
            f"[fatal] tracker {name!r} is neither a builtin {BUILTIN_TRACKERS} "
            f"nor an existing YAML path"
        )
    return os.path.splitext(os.path.basename(path))[0], path


def run_sequence(
    model,
    seq,
    tracker_cfg: str,
    args,
    keep_classes: tuple[int, ...] | None,
) -> tuple[MotCounts, list[tuple], list[float]]:
    """Track one sequence end to end and score it.

    The first frame is submitted with ``persist=False`` so Ultralytics rebuilds
    the tracker: state carried over from the previous sequence would let ids
    survive a scene cut and score as identity errors that the tracker never
    made.
    """
    acc = MotAccumulator(iou_thresh=args.iou_thresh)
    rows: list[tuple] = []
    per_frame_ms: list[float] = []

    for k, frame in enumerate(seq.frames):
        img = cv2.imread(frame.path)
        if img is None:
            print(f"    [warn] unreadable frame {frame.path}", file=sys.stderr)
            continue

        t0 = time.perf_counter()
        result = model.track(
            img,
            persist=k > 0,
            tracker=tracker_cfg,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.nms_iou,
            max_det=args.max_det,
            device=args.device,
            quantize=16 if args.half else None,
            verbose=False,
        )[0]
        per_frame_ms.append((time.perf_counter() - t0) * 1000.0)

        boxes = result.boxes
        if boxes is not None and boxes.id is not None:
            hyp_xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
            hyp_ids = boxes.id.cpu().numpy().astype(np.int64)
            hyp_cls = boxes.cls.cpu().numpy().astype(np.int32)
            hyp_conf = boxes.conf.cpu().numpy().astype(np.float32)
        else:
            # No ids means no confirmed tracks this frame -- normal for the first
            # few frames of a sequence, not an error.
            hyp_xyxy = np.zeros((0, 4), np.float32)
            hyp_ids = np.zeros((0,), np.int64)
            hyp_cls = np.zeros((0,), np.int32)
            hyp_conf = np.zeros((0,), np.float32)

        gt = filter_classes(frame, keep_classes)

        if args.ignore_policy == "mask" and len(hyp_xyxy):
            keep = keep_mask(hyp_xyxy, gt.ignore_boxes, args.ignore_overlap)
            hyp_xyxy, hyp_ids = hyp_xyxy[keep], hyp_ids[keep]
            hyp_cls, hyp_conf = hyp_cls[keep], hyp_conf[keep]

        acc.update(gt.ids, gt.boxes, hyp_ids, hyp_xyxy)

        for (x1, y1, x2, y2), tid, cls, cf in zip(
            hyp_xyxy, hyp_ids, hyp_cls, hyp_conf
        ):
            rows.append((frame.index, int(tid), float(x1), float(y1),
                         float(x2 - x1), float(y2 - y1), float(cf), int(cls)))

        if args.progress_every and (k + 1) % args.progress_every == 0:
            print(f"    {seq.name} {k + 1}/{len(seq.frames)} frames", flush=True)

    return acc.finalize(), rows, per_frame_ms


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(
        ROOT, "test", "VisDrone2019-MOT-test-dev"))
    ap.add_argument("--model", default=os.path.join(
        ROOT, "models", "yolov8n_visdrone.pt"),
        help="a torch .pt: Ultralytics' trackers need its Detect features for "
             "with_reid=auto, and an ONNX graph cannot supply them")
    ap.add_argument("--trackers", nargs="+", default=["botsort"],
                    help=f"{BUILTIN_TRACKERS} or paths to tracker YAMLs")
    ap.add_argument("--sequences", nargs="*", default=None,
                    help="subset by name; default is every scorable sequence")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.10,
                    help="detector threshold. Low on purpose: BoT-SORT and "
                         "ByteTrack need sub-threshold detections for their "
                         "second association stage, and the tracker's own "
                         "track_high_thresh does the real gating")
    ap.add_argument("--nms-iou", type=float, default=0.45)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--device", default="0")
    ap.add_argument("--half", action="store_true", default=True)
    ap.add_argument("--no-half", dest="half", action="store_false")
    ap.add_argument("--iou-thresh", type=float, default=0.5,
                    help="IoU gate for GT<->hypothesis association")
    ap.add_argument("--class-policy", default="mot5", choices=["mot5", "all10"])
    ap.add_argument("--ignore-policy", default="mask", choices=["mask", "keep"])
    ap.add_argument("--ignore-overlap", type=float, default=0.5)
    ap.add_argument("--limit-frames", type=int, default=0,
                    help="0 = all frames; a small value smoke-tests the harness")
    ap.add_argument("--out", default=os.path.join(
        ROOT, "results", "tracking_quality.csv"))
    ap.add_argument("--save-mot", default=os.path.join(ROOT, "results", "mot"),
                    help="directory for MOTChallenge-format hypotheses; '' skips")
    ap.add_argument("--progress-every", type=int, default=200)
    ap.add_argument("--tag", default="", help="free-text label for this run")
    args = ap.parse_args(argv)

    if not os.path.isfile(args.model):
        raise SystemExit(f"[fatal] weights not found: {args.model}")

    names = list_sequences(args.data)
    if args.sequences:
        unknown = set(args.sequences) - set(names)
        if unknown:
            raise SystemExit(
                f"[fatal] not scorable here: {sorted(unknown)}\n"
                f"        available: {names}"
            )
        names = [n for n in names if n in args.sequences]
    if not names:
        raise SystemExit(f"[fatal] no scorable sequences under {args.data}")

    keep_classes = MOT_EVAL_CLASSES if args.class_policy == "mot5" else None

    print(f"[info] {len(names)} sequences from {args.data}")
    sequences = [
        load_sequence(args.data, n, limit=args.limit_frames or None) for n in names
    ]
    total_frames = sum(len(s) for s in sequences)
    print(f"[info] {total_frames} frames, "
          f"{sum(s.n_gt_ids for s in sequences)} gt ids, "
          f"{sum(s.n_gt_boxes for s in sequences)} gt boxes "
          f"(before the {args.class_policy} class policy)")

    from ultralytics import YOLO           # imported late: it pulls in torch

    model = YOLO(args.model)
    model_hash = sha256_short(args.model)

    # Warm up before any timing. The first forward pass pays CUDA context setup
    # and cuDNN autotuning, which lands entirely on whichever tracker happens to
    # run first and made it look 5x slower than the rest.
    warm = cv2.imread(sequences[0].frames[0].path)
    for _ in range(3):
        model.predict(warm, imgsz=args.imgsz, conf=args.conf, iou=args.nms_iou,
                      max_det=args.max_det, device=args.device,
                      quantize=16 if args.half else None, verbose=False)

    rows: list[dict] = []
    for tracker in args.trackers:
        label, cfg = resolve_tracker(tracker)
        print(f"\n[run] tracker={label}  config={cfg}")
        print(FORMAT_HEADER)

        overall = MotCounts()
        all_ms: list[float] = []
        t_wall = time.perf_counter()

        for seq in sequences:
            counts, mot_rows, ms = run_sequence(
                model, seq, cfg, args, keep_classes
            )
            overall = overall + counts
            all_ms.extend(ms)
            s = summarize(counts)
            print(format_row(seq.name, s))

            if args.save_mot:
                write_mot_results(
                    os.path.join(args.save_mot, label, seq.name + ".txt"), mot_rows
                )

            rows.append({
                "tracker": label, "tracker_config": cfg, "sequence": seq.name,
                "ms_per_frame_avg": float(np.mean(ms)) if ms else float("nan"),
                **s,
            })

        wall = time.perf_counter() - t_wall
        s = summarize(overall)
        print("-" * len(FORMAT_HEADER))
        print(format_row("OVERALL", s))
        print(f"    {np.mean(all_ms):.1f} ms/frame avg, {wall:.0f}s wall, "
              f"IDF1 {s['idf1']*100:.2f}  MOTA {s['mota']*100:.2f}  "
              f"IDSW {s['idsw']}  id_inflation {s['id_inflation']:.2f}")

        rows.append({
            "tracker": label, "tracker_config": cfg, "sequence": "OVERALL",
            "ms_per_frame_avg": float(np.mean(all_ms)) if all_ms else float("nan"),
            **s,
        })

    common = {
        "model": os.path.basename(args.model),
        "model_sha256_16": model_hash,
        "imgsz": args.imgsz,
        "conf_thres": args.conf,
        "nms_iou": args.nms_iou,
        "max_det": args.max_det,
        "half": args.half,
        "iou_thresh": args.iou_thresh,
        "class_policy": args.class_policy,
        "ignore_policy": args.ignore_policy,
        "n_sequences": len(sequences),
        "tag": args.tag,
    }
    for r in rows:
        r.update(common)

    print(f"\n[ok] appended {append_rows(args.out, rows)} row(s) -> {args.out}")

    json_out = os.path.splitext(args.out)[0] + ".json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"[ok] {json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
