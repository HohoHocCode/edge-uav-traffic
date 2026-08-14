#!/usr/bin/env python3
"""Score raw MOTChallenge trajectories with the official HOTA algorithm.

This is a small, dependency-light adaptation of TrackEval's HOTA metric:
https://github.com/JonathonLuiten/TrackEval/blob/master/trackeval/metrics/hota.py

It intentionally follows this repository's tracking convention: association
is class-agnostic after the upstream class policy and ignore-region filtering.
For VID, use the all-10-class raw trajectories emitted by ``bench_tracking``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from metrics import iou_xyxy  # noqa: E402
from mot_data import list_sequences, load_sequence  # noqa: E402


ALPHAS = np.arange(0.05, 0.99, 0.05)


def load_tracker(path: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    per_frame: dict[int, tuple[list[int], list[list[float]]]] = {}
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_number, raw in enumerate(handle, start=1):
            fields = raw.strip().rstrip(",").split(",")
            if fields == [""]:
                continue
            if len(fields) < 6:
                raise ValueError(f"{path}:{line_number}: expected >=6 fields")
            frame = int(float(fields[0]))
            track_id = int(float(fields[1]))
            x, y, width, height = (float(fields[i]) for i in range(2, 6))
            if width <= 0 or height <= 0:
                continue
            ids, boxes = per_frame.setdefault(frame, ([], []))
            ids.append(track_id)
            boxes.append([x, y, x + width, y + height])
    return {
        frame: (
            np.asarray(ids, np.int64),
            np.asarray(boxes, np.float64).reshape(-1, 4),
        )
        for frame, (ids, boxes) in per_frame.items()
    }


def dense(values: list[np.ndarray]) -> tuple[list[np.ndarray], int]:
    unique = sorted({int(value) for array in values for value in array})
    index = {value: position for position, value in enumerate(unique)}
    return [
        np.asarray([index[int(value)] for value in array], np.int64)
        for array in values
    ], len(unique)


def evaluate_sequence(sequence, tracker_path: Path) -> dict[str, np.ndarray]:
    tracker = load_tracker(tracker_path)
    gt_ids_raw = [frame.ids for frame in sequence.frames]
    tr_ids_raw = [tracker.get(frame.index, (np.zeros(0, np.int64), None))[0]
                  for frame in sequence.frames]
    gt_ids, n_gt_ids = dense(gt_ids_raw)
    tr_ids, n_tr_ids = dense(tr_ids_raw)

    similarities = []
    for frame in sequence.frames:
        tr_boxes = tracker.get(
            frame.index, (None, np.zeros((0, 4), np.float64))
        )[1]
        similarities.append(iou_xyxy(frame.boxes, tr_boxes))

    n_alpha = len(ALPHAS)
    tp = np.zeros(n_alpha, np.float64)
    fn = np.zeros(n_alpha, np.float64)
    fp = np.zeros(n_alpha, np.float64)
    loc_sum = np.zeros(n_alpha, np.float64)
    potential = np.zeros((n_gt_ids, n_tr_ids), np.float64)
    gt_count = np.zeros((n_gt_ids, 1), np.float64)
    tr_count = np.zeros((1, n_tr_ids), np.float64)

    # TrackEval HOTA, first pass: global alignment score between identities.
    for gids, tids, similarity in zip(gt_ids, tr_ids, similarities):
        if len(gids) and len(tids):
            denominator = (
                similarity.sum(0)[None, :]
                + similarity.sum(1)[:, None]
                - similarity
            )
            sim_iou = np.zeros_like(similarity)
            mask = denominator > np.finfo(float).eps
            sim_iou[mask] = similarity[mask] / denominator[mask]
            potential[gids[:, None], tids[None, :]] += sim_iou
        if len(gids):
            gt_count[gids] += 1
        if len(tids):
            tr_count[0, tids] += 1

    denominator = gt_count + tr_count - potential
    alignment = np.divide(
        potential,
        denominator,
        out=np.zeros_like(potential),
        where=denominator > np.finfo(float).eps,
    )
    match_counts = [np.zeros_like(potential) for _ in ALPHAS]

    # Second pass: one globally-aligned Hungarian assignment per frame, then
    # threshold it at each alpha exactly as TrackEval does.
    for gids, tids, similarity in zip(gt_ids, tr_ids, similarities):
        if len(gids) == 0:
            fp += len(tids)
            continue
        if len(tids) == 0:
            fn += len(gids)
            continue
        score = alignment[gids[:, None], tids[None, :]] * similarity
        rows, cols = linear_sum_assignment(-score)
        for alpha_index, alpha in enumerate(ALPHAS):
            accepted = similarity[rows, cols] >= alpha - np.finfo(float).eps
            match_rows, match_cols = rows[accepted], cols[accepted]
            matches = len(match_rows)
            tp[alpha_index] += matches
            fn[alpha_index] += len(gids) - matches
            fp[alpha_index] += len(tids) - matches
            if matches:
                loc_sum[alpha_index] += similarity[match_rows, match_cols].sum()
                match_counts[alpha_index][
                    gids[match_rows], tids[match_cols]
                ] += 1

    ass_a = np.zeros(n_alpha, np.float64)
    for alpha_index, counts in enumerate(match_counts):
        accuracy = counts / np.maximum(1.0, gt_count + tr_count - counts)
        ass_a[alpha_index] = (counts * accuracy).sum() / max(1.0, tp[alpha_index])
    return {"tp": tp, "fn": fn, "fp": fp, "ass_a": ass_a, "loc_sum": loc_sum}


def combine(sequence_results: list[dict[str, np.ndarray]]) -> dict:
    tp = sum((result["tp"] for result in sequence_results), np.zeros(len(ALPHAS)))
    fn = sum((result["fn"] for result in sequence_results), np.zeros(len(ALPHAS)))
    fp = sum((result["fp"] for result in sequence_results), np.zeros(len(ALPHAS)))
    ass_numerator = sum(
        (result["ass_a"] * result["tp"] for result in sequence_results),
        np.zeros(len(ALPHAS)),
    )
    loc_sum = sum(
        (result["loc_sum"] for result in sequence_results), np.zeros(len(ALPHAS))
    )
    ass_a = ass_numerator / np.maximum(1.0, tp)
    det_a = tp / np.maximum(1.0, tp + fn + fp)
    loc_a = loc_sum / np.maximum(1e-10, tp)
    hota = np.sqrt(det_a * ass_a)
    return {
        "HOTA": float(hota.mean()),
        "DetA": float(det_a.mean()),
        "AssA": float(ass_a.mean()),
        "LocA": float(loc_a.mean()),
        "HOTA_alpha": hota.tolist(),
        "DetA_alpha": det_a.tolist(),
        "AssA_alpha": ass_a.tolist(),
        "LocA_alpha": loc_a.tolist(),
        "alpha": ALPHAS.tolist(),
        "n_sequences": len(sequence_results),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", default=os.path.join(ROOT, "test", "VisDrone2019-VID-test-dev")
    )
    parser.add_argument("--tracker-dir", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", default=os.path.join(ROOT, "results", "hota.csv"))
    args = parser.parse_args(argv)

    results = []
    for name in list_sequences(args.data):
        print(f"[score] {name}", flush=True)
        sequence = load_sequence(args.data, name)
        results.append(
            evaluate_sequence(sequence, Path(args.tracker_dir) / f"{name}.txt")
        )
    summary = combine(results)
    row = {key: value for key, value in summary.items() if not key.endswith("_alpha")}
    row = {"label": args.label, **row}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    exists = out.is_file() and out.stat().st_size > 0
    with out.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    json_path = out.with_suffix(".json")
    existing = json.loads(json_path.read_text(encoding="utf-8")) if json_path.is_file() else []
    existing.append({"label": args.label, **summary})
    json_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(
        f"[ok] {args.label}: HOTA={summary['HOTA']*100:.2f} "
        f"DetA={summary['DetA']*100:.2f} AssA={summary['AssA']*100:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
