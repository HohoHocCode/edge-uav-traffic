"""CLEAR MOT and identity metrics, computed here rather than pulled in.

``motmetrics`` and ``TrackEval`` are not installed in this environment and both
would be a heavy dependency for the four numbers this repository actually
quotes. The definitions below are the published ones:

* CLEAR MOT -- Bernardin & Stiefelhagen 2008. MOTA, MOTP, FP, FN, IDSW, FM,
  MT/PT/ML.
* Identity metrics -- Ristani et al. 2016. IDF1, IDP, IDR.

Two implementation details decide whether the numbers match other tools:

1. **Match continuity.** A frame's association is not a fresh Hungarian solve.
   Pairs matched in the previous frame that still clear the IoU gate are kept
   first, and only the remainder is solved. Without this, two overlapping
   objects trade hypotheses whenever the optimum is a tie, and IDSW counts
   arbitration noise instead of tracker failures.
2. **IDF1 is global, not per frame.** It comes from one assignment between
   ground-truth ids and hypothesis ids over the whole sequence, maximising the
   number of co-located frames. That is what makes it sensitive to identity
   fragmentation in a way MOTA structurally is not: MOTA charges an id switch
   once, IDF1 charges every frame the identity was wrong.

Which is why both are reported. A tracker that fragments has near-unchanged
MOTA and visibly worse IDF1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from metrics import iou_xyxy

_UNMATCHED = 1e6          # cost for a pair below the IoU gate


@dataclass
class MotCounts:
    """Raw accumulator state. Sums across sequences; ratios never do."""

    n_frames: int = 0
    n_gt: int = 0                     # ground-truth boxes
    n_hyp: int = 0                    # hypothesis boxes surviving ignore filter
    tp: int = 0
    fp: int = 0
    fn: int = 0
    idsw: int = 0
    fm: int = 0
    iou_sum: float = 0.0
    mt: int = 0
    pt: int = 0
    ml: int = 0
    n_gt_ids: int = 0
    n_hyp_ids: int = 0
    idtp: int = 0
    idfp: int = 0
    idfn: int = 0

    def __add__(self, other: "MotCounts") -> "MotCounts":
        out = MotCounts()
        for f in self.__dataclass_fields__:
            setattr(out, f, getattr(self, f) + getattr(other, f))
        return out


class MotAccumulator:
    """Per-frame association bookkeeping for one sequence.

    Ground truth and hypotheses are matched class-agnostically. Class labels are
    the detector's problem and are scored by ``bench_quality.py``; mixing them in
    here would hide tracker fragmentation behind class confusion, which is the
    opposite of what this benchmark is for. Restrict the *ground truth* by class
    upstream (``mot_data.filter_classes``) if a class-conditioned number is
    wanted.
    """

    def __init__(self, iou_thresh: float = 0.5) -> None:
        self.iou_thresh = float(iou_thresh)
        self.c = MotCounts()
        self._last_hyp: dict[int, int] = {}      # gt id -> hyp id last matched to
        self._tracked_prev: set[int] = set()     # gt ids matched in previous frame
        self._gt_present: dict[int, int] = {}     # gt id -> frames present
        self._gt_tracked: dict[int, int] = {}     # gt id -> frames matched
        self._hyp_ids: set[int] = set()
        self._cooc: dict[tuple[int, int], int] = {}   # (gt id, hyp id) -> frames

    # ------------------------------------------------------------------ #
    def update(
        self,
        gt_ids: np.ndarray,
        gt_boxes: np.ndarray,
        hyp_ids: np.ndarray,
        hyp_boxes: np.ndarray,
    ) -> None:
        gt_ids = np.asarray(gt_ids, np.int64).reshape(-1)
        hyp_ids = np.asarray(hyp_ids, np.int64).reshape(-1)
        gt_boxes = np.asarray(gt_boxes, np.float32).reshape(-1, 4)
        hyp_boxes = np.asarray(hyp_boxes, np.float32).reshape(-1, 4)

        self.c.n_frames += 1
        self.c.n_gt += len(gt_ids)
        self.c.n_hyp += len(hyp_ids)
        self._hyp_ids.update(int(h) for h in hyp_ids)
        for g in gt_ids:
            self._gt_present[int(g)] = self._gt_present.get(int(g), 0) + 1

        iou = iou_xyxy(gt_boxes, hyp_boxes)
        valid = iou >= self.iou_thresh

        # Every co-location feeds IDF1, whether or not this frame's association
        # picked it. IDF1's assignment is global, so it must see the full matrix.
        for gi, gj in zip(*np.nonzero(valid)):
            key = (int(gt_ids[gi]), int(hyp_ids[gj]))
            self._cooc[key] = self._cooc.get(key, 0) + 1

        matches = self._associate(gt_ids, hyp_ids, iou, valid)

        tracked_now = set()
        for gi, gj in matches:
            g, h = int(gt_ids[gi]), int(hyp_ids[gj])
            self.c.tp += 1
            self.c.iou_sum += float(iou[gi, gj])
            self._gt_tracked[g] = self._gt_tracked.get(g, 0) + 1
            tracked_now.add(g)

            prev = self._last_hyp.get(g)
            if prev is not None and prev != h:
                self.c.idsw += 1
            self._last_hyp[g] = h

        self.c.fn += len(gt_ids) - len(matches)
        self.c.fp += len(hyp_ids) - len(matches)

        # Fragmentation: a ground-truth object that was tracked, stopped being
        # tracked, and is tracked again. Counted on the resumption, and only for
        # objects that had been tracked at least once before.
        for g in tracked_now - self._tracked_prev:
            if g in self._last_hyp and self._gt_tracked.get(g, 0) > 1:
                self.c.fm += 1
        self._tracked_prev = tracked_now

    # ------------------------------------------------------------------ #
    def _associate(
        self,
        gt_ids: np.ndarray,
        hyp_ids: np.ndarray,
        iou: np.ndarray,
        valid: np.ndarray,
    ) -> list[tuple[int, int]]:
        """Continuity-preferring association. See module docstring, point 1."""
        if len(gt_ids) == 0 or len(hyp_ids) == 0:
            return []

        matches: list[tuple[int, int]] = []
        gt_free = np.ones(len(gt_ids), bool)
        hyp_free = np.ones(len(hyp_ids), bool)

        hyp_pos = {int(h): j for j, h in enumerate(hyp_ids)}
        for gi, g in enumerate(gt_ids):
            j = hyp_pos.get(self._last_hyp.get(int(g), -1))
            if j is not None and valid[gi, j] and hyp_free[j]:
                matches.append((gi, j))
                gt_free[gi] = False
                hyp_free[j] = False

        gi_rest = np.nonzero(gt_free)[0]
        gj_rest = np.nonzero(hyp_free)[0]
        if len(gi_rest) and len(gj_rest):
            sub = iou[np.ix_(gi_rest, gj_rest)]
            sub_valid = valid[np.ix_(gi_rest, gj_rest)]
            cost = np.where(sub_valid, -sub, _UNMATCHED)
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                if sub_valid[r, c]:
                    matches.append((int(gi_rest[r]), int(gj_rest[c])))
        return matches

    # ------------------------------------------------------------------ #
    def finalize(self) -> MotCounts:
        """Close out track-level counts and solve the global identity matching."""
        c = self.c
        c.n_gt_ids = len(self._gt_present)
        c.n_hyp_ids = len(self._hyp_ids)

        for g, present in self._gt_present.items():
            ratio = self._gt_tracked.get(g, 0) / max(present, 1)
            if ratio >= 0.8:
                c.mt += 1
            elif ratio <= 0.2:
                c.ml += 1
            else:
                c.pt += 1

        c.idtp = self._solve_identity()
        c.idfn = c.n_gt - c.idtp
        c.idfp = c.n_hyp - c.idtp
        return c

    def _solve_identity(self) -> int:
        """Max co-located frames over a one-to-one gt id <-> hyp id assignment."""
        if not self._cooc:
            return 0
        gt_index = {g: i for i, g in enumerate(sorted({g for g, _ in self._cooc}))}
        hyp_index = {h: j for j, h in enumerate(sorted({h for _, h in self._cooc}))}
        m = np.zeros((len(gt_index), len(hyp_index)), np.int64)
        for (g, h), n in self._cooc.items():
            m[gt_index[g], hyp_index[h]] = n
        rows, cols = linear_sum_assignment(-m)
        return int(m[rows, cols].sum())


# ---------------------------------------------------------------------- #
def summarize(c: MotCounts) -> dict:
    """Ratios from summed counts. Safe to call on an aggregate of sequences."""
    n_gt = max(c.n_gt, 1)
    n_hyp = max(c.n_hyp, 1)
    id_den = max(2 * c.idtp + c.idfp + c.idfn, 1)
    n_ids = max(c.n_gt_ids, 1)
    return {
        "idf1": 2.0 * c.idtp / id_den,
        "idp": c.idtp / n_hyp,
        "idr": c.idtp / n_gt,
        "mota": 1.0 - (c.fn + c.fp + c.idsw) / n_gt,
        "motp_iou": c.iou_sum / max(c.tp, 1),
        "recall": c.tp / n_gt,
        "precision": c.tp / n_hyp,
        "idsw": c.idsw,
        "fm": c.fm,
        "fp": c.fp,
        "fn": c.fn,
        "tp": c.tp,
        "mt": c.mt,
        "pt": c.pt,
        "ml": c.ml,
        "mt_pct": 100.0 * c.mt / n_ids,
        "ml_pct": 100.0 * c.ml / n_ids,
        "n_frames": c.n_frames,
        "n_gt": c.n_gt,
        "n_hyp": c.n_hyp,
        "n_gt_ids": c.n_gt_ids,
        "n_hyp_ids": c.n_hyp_ids,
        # How many hypothesis identities the tracker spent per real object. 1.0
        # is perfect; 3.0 means every object was renamed twice on average.
        "id_inflation": c.n_hyp_ids / n_ids,
        "idsw_per_100_frames": 100.0 * c.idsw / max(c.n_frames, 1),
    }


HEADER = (
    "sequence", "idf1", "idp", "idr", "mota", "motp_iou", "recall", "precision",
    "idsw", "fm", "fp", "fn", "mt", "pt", "ml", "n_frames", "n_gt", "n_hyp",
    "n_gt_ids", "n_hyp_ids", "id_inflation", "idsw_per_100_frames",
)


def format_row(name: str, s: dict) -> str:
    return (f"{name:24s} {s['idf1']*100:6.2f} {s['mota']*100:7.2f} "
            f"{s['motp_iou']*100:6.2f} {s['idsw']:6d} {s['fm']:6d} "
            f"{s['fp']:7d} {s['fn']:7d} {s['mt']:5d} {s['ml']:5d} "
            f"{s['id_inflation']:6.2f}")


FORMAT_HEADER = (
    f"{'sequence':24s} {'IDF1':>6s} {'MOTA':>7s} {'MOTP':>6s} {'IDSW':>6s} "
    f"{'FM':>6s} {'FP':>7s} {'FN':>7s} {'MT':>5s} {'ML':>5s} {'IDinf':>6s}"
)
