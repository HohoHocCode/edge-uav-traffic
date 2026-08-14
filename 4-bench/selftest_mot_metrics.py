#!/usr/bin/env python3
"""Sanity checks for the hand-rolled CLEAR MOT / IDF1 implementation.

    python 4-bench/selftest_mot_metrics.py

``mot_metrics`` computes published metrics without ``motmetrics`` or
``TrackEval`` to lean on, so the definitions are checked against cases whose
answers are arithmetic rather than opinion: a perfect track, a single id switch,
a detection gap, a pure false positive, a consistent relabelling, and a
hypothesis just outside the IoU gate. Exits non-zero on any mismatch.

Case 2 and case 5 are the ones that matter for tracker selection. Case 2 shows
MOTA moving by 1/N for an id switch while IDF1 halves -- which is why a
fragmenting tracker has to be judged on IDF1. Case 5 shows that renaming every
object consistently is *not* an error: IDF1 solves a global assignment, so a
tracker is not punished for choosing different id numbers than the annotator.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from mot_metrics import MotAccumulator, summarize


def box(x, y, w=10, h=10):
    return [x, y, x + w, y + h]


def run(frames, thresh=0.5):
    acc = MotAccumulator(iou_thresh=thresh)
    for gt_ids, gt_boxes, h_ids, h_boxes in frames:
        acc.update(np.array(gt_ids), np.array(gt_boxes, np.float32).reshape(-1, 4),
                   np.array(h_ids), np.array(h_boxes, np.float32).reshape(-1, 4))
    return summarize(acc.finalize())


N = 20
ok = True


def check(name, got, want, tol=1e-6):
    global ok
    good = abs(got - want) <= tol
    ok &= good
    print(f"{'PASS' if good else 'FAIL'}  {name:38s} got {got:.4f}  want {want:.4f}")


# 1. Perfect tracking: one object, hypothesis is identical every frame.
f = [([1], [box(i * 2, 0)], [7], [box(i * 2, 0)]) for i in range(N)]
s = run(f)
check("perfect: idf1", s["idf1"], 1.0)
check("perfect: mota", s["mota"], 1.0)
check("perfect: idsw", s["idsw"], 0)
check("perfect: motp_iou", s["motp_iou"], 1.0)
check("perfect: id_inflation", s["id_inflation"], 1.0)

# 2. One id switch at the halfway point. MOTA charges it once; IDF1 charges
#    every frame the second identity was wrong.
f = [([1], [box(i * 2, 0)], [7 if i < N // 2 else 8], [box(i * 2, 0)])
     for i in range(N)]
s = run(f)
check("switch: idsw", s["idsw"], 1)
check("switch: mota", s["mota"], 1.0 - 1 / N)
check("switch: idf1", s["idf1"], 2 * (N // 2) / (2 * (N // 2) + 10 + 10))
check("switch: recall", s["recall"], 1.0)
check("switch: id_inflation", s["id_inflation"], 2.0)

# 3. Missed detections in the middle: FN, fragmentation, no id switch.
f = []
for i in range(N):
    gone = 8 <= i < 12
    f.append(([1], [box(i * 2, 0)], [] if gone else [7], [] if gone else [box(i * 2, 0)]))
s = run(f)
check("gap: fn", s["fn"], 4)
check("gap: fp", s["fp"], 0)
check("gap: idsw", s["idsw"], 0)
check("gap: fm", s["fm"], 1)
check("gap: mota", s["mota"], 1.0 - 4 / N)
check("gap: mt (16/20 = 0.8 tracked)", s["mt"], 1)

# 4. Pure false positives: a ghost track alongside a perfect one.
f = [([1], [box(0, 0)], [7, 9], [box(0, 0), box(500, 500)]) for _ in range(N)]
s = run(f)
check("ghost: fp", s["fp"], N)
check("ghost: mota", s["mota"], 1.0 - N / N)
check("ghost: idp (half the hyps are junk)", s["idp"], 0.5)
check("ghost: idr", s["idr"], 1.0)

# 5. Two objects whose hypotheses are swapped for the whole run: IDF1 must
#    recover the consistent-but-renamed mapping and score it perfect.
f = [([1, 2], [box(0, 0), box(100, 100)], [9, 8], [box(0, 0), box(100, 100)])
     for _ in range(N)]
s = run(f)
check("relabel: idf1", s["idf1"], 1.0)
check("relabel: idsw", s["idsw"], 0)

# 6. Below the IoU gate: a hypothesis offset far enough is FP + FN, not a match.
f = [([1], [box(0, 0)], [7], [box(9, 0)]) for _ in range(N)]
s = run(f)
check("below gate: tp", s["tp"], 0)
check("below gate: fp", s["fp"], N)
check("below gate: fn", s["fn"], N)
check("below gate: mota (FP+FN over N gt)", s["mota"], 1.0 - 2 * N / N)

print("\nALL PASS" if ok else "\nFAILURES PRESENT")
sys.exit(0 if ok else 1)
