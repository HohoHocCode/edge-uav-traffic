#!/usr/bin/env python3
"""Score saved MOTChallenge hypotheses against VisDrone-MOT ground truth.

``bench_tracking.py`` writes every tracker's per-frame output to
``results/mot/<tracker>/<sequence>.txt``. This scores those files, so a metric
can be recomputed -- or a scoring convention changed -- without touching the GPU
again. It is also how a partially finished sweep is read: the hypothesis files
land per sequence, while the CSV is only written when the whole run ends.

    python 4-bench/score_mot.py                       # every tracker on disk
    python 4-bench/score_mot.py --trackers botsort botsort_nogmc
    python 4-bench/score_mot.py --class-policy all10  # rescore, no re-run

One caveat on ``--ignore-policy``: the saved files already had the ignore filter
applied if the run that produced them used ``mask``, and re-applying it is a
no-op. Going the other way is not possible -- hypotheses dropped at run time are
not on disk -- so scoring ``keep`` is only meaningful for files produced by a
``keep`` run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from mot_data import (  # noqa: E402
    MOT_EVAL_CLASSES,
    filter_classes,
    list_sequences,
    load_sequence,
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


def read_mot_file(path: str) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """``{frame_index: (ids, xyxy)}`` from a MOTChallenge results file."""
    per_frame: dict[int, tuple[list, list]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = line.split(",")
            if len(p) < 6:
                continue
            frame, tid = int(float(p[0])), int(float(p[1]))
            x, y, w, h = (float(v) for v in p[2:6])
            ids, boxes = per_frame.setdefault(frame, ([], []))
            ids.append(tid)
            boxes.append([x, y, x + w, y + h])
    return {
        k: (np.asarray(v[0], np.int64), np.asarray(v[1], np.float32).reshape(-1, 4))
        for k, v in per_frame.items()
    }


def score_tracker(label: str, hyp_dir: str, sequences, args) -> list[dict]:
    keep_classes = MOT_EVAL_CLASSES if args.class_policy == "mot5" else None
    empty = (np.zeros((0,), np.int64), np.zeros((0, 4), np.float32))

    print(f"\n[score] tracker={label}")
    print(FORMAT_HEADER)

    rows: list[dict] = []
    overall = MotCounts()
    for seq in sequences:
        path = os.path.join(hyp_dir, seq.name + ".txt")
        if not os.path.isfile(path):
            print(f"{seq.name:24s}   (no hypotheses on disk -- skipped)")
            continue
        hyps = read_mot_file(path)

        acc = MotAccumulator(iou_thresh=args.iou_thresh)
        for frame in seq.frames:
            gt = filter_classes(frame, keep_classes)
            hyp_ids, hyp_xyxy = hyps.get(frame.index, empty)
            if args.ignore_policy == "mask" and len(hyp_xyxy):
                keep = keep_mask(hyp_xyxy, gt.ignore_boxes, args.ignore_overlap)
                hyp_ids, hyp_xyxy = hyp_ids[keep], hyp_xyxy[keep]
            acc.update(gt.ids, gt.boxes, hyp_ids, hyp_xyxy)

        counts = acc.finalize()
        overall = overall + counts
        s = summarize(counts)
        print(format_row(seq.name, s))
        rows.append({"tracker": label, "sequence": seq.name, **s})

    if rows:
        s = summarize(overall)
        print("-" * len(FORMAT_HEADER))
        print(format_row("OVERALL", s))
        rows.append({"tracker": label, "sequence": "OVERALL", **s})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(
        ROOT, "test", "VisDrone2019-MOT-test-dev"))
    ap.add_argument("--results", default=os.path.join(ROOT, "results", "mot"),
                    help="directory of <tracker>/<sequence>.txt hypothesis files")
    ap.add_argument("--trackers", nargs="*", default=None,
                    help="subdirectory names; default is all of them")
    ap.add_argument("--sequences", nargs="*", default=None)
    ap.add_argument("--iou-thresh", type=float, default=0.5)
    ap.add_argument("--class-policy", default="mot5", choices=["mot5", "all10"])
    ap.add_argument("--ignore-policy", default="mask", choices=["mask", "keep"])
    ap.add_argument("--ignore-overlap", type=float, default=0.5)
    ap.add_argument("--out", default="", help="CSV to write; empty prints only")
    args = ap.parse_args(argv)

    if not os.path.isdir(args.results):
        raise SystemExit(f"[fatal] no hypothesis directory: {args.results}")
    labels = args.trackers or sorted(
        d for d in os.listdir(args.results)
        if os.path.isdir(os.path.join(args.results, d))
    )
    if not labels:
        raise SystemExit(f"[fatal] no tracker subdirectories under {args.results}")

    names = list_sequences(args.data)
    if args.sequences:
        names = [n for n in names if n in args.sequences]
    sequences = [load_sequence(args.data, n) for n in names]
    print(f"[info] {len(sequences)} sequences, {sum(len(s) for s in sequences)} "
          f"frames | class_policy={args.class_policy} "
          f"ignore_policy={args.ignore_policy} iou_thresh={args.iou_thresh}")

    rows: list[dict] = []
    for label in labels:
        rows.extend(score_tracker(
            label, os.path.join(args.results, label), sequences, args
        ))

    if args.out and rows:
        common = {
            "class_policy": args.class_policy,
            "ignore_policy": args.ignore_policy,
            "iou_thresh": args.iou_thresh,
            "n_sequences": len(sequences),
            "source": "rescored from saved hypotheses",
        }
        for r in rows:
            r.update(common)
        print(f"\n[ok] appended {append_rows(args.out, rows)} row(s) -> {args.out}")
        with open(os.path.splitext(args.out)[0] + ".json", "w",
                  encoding="utf-8") as f:
            json.dump(rows, f, indent=2, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
