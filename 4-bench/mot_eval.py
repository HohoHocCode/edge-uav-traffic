#!/usr/bin/env python3
"""Standalone multi-object-tracking benchmark for VisDrone-MOT.

Point it at a detector checkpoint and a VisDrone-MOT split, get IDF1 / MOTA /
IDSW. One file, no imports from the repository it came from -- copy it anywhere.

    pip install ultralytics                 # pulls numpy, opencv, scipy, torch
    python mot_eval.py --selftest           # verify the metrics, no data needed
    python mot_eval.py --model best.pt --data VisDrone2019-MOT-test-dev
    python mot_eval.py --model best.pt --trackers botsort tracktrack ocsort

Expected layout of ``--data``::

    VisDrone2019-MOT-test-dev/
        sequences/<name>/0000001.jpg ...
        annotations/<name>.txt

Sequences without both are skipped, loudly. This split ships 17 annotation files
but only 13 image directories.

--------------------------------------------------------------------------- #
WHY THESE METRICS

MOTA charges an identity switch once, at the frame it happens. IDF1 charges
every frame the identity was wrong, because it solves one global assignment
between ground-truth ids and hypothesis ids over the whole sequence. A tracker
that fragments therefore shows a near-unchanged MOTA and a visibly worse IDF1,
so judging tracker quality on MOTA alone will mislead you. Measured on this
split, disabling camera-motion compensation cost 8.03 IDF1 and only 1.29 MOTA
while nearly doubling id switches.

``id_inflation`` is the plain-language one: hypothesis identities spent per real
object. 1.0 is perfect, 3.0 means every object was renamed twice on average.

--------------------------------------------------------------------------- #
CONVENTIONS, DECLARED PER ROW

A MOT number without these is not comparable to anything, so all three are
written into every output row.

``class_policy``   ``mot5`` scores the five categories of the official VisDrone
                   MOT challenge (pedestrian, car, van, truck, bus); ``all10``
                   scores the full taxonomy. Classes outside the scored set
                   become ignore regions, never background -- dropping them
                   would make every correct track on a motorbike a false
                   positive.
``ignore_policy``  ``mask`` drops hypotheses inside a VisDrone ignored region
                   before scoring; ``keep`` counts them as false positives.
                   VisDrone marks 23k ignore regions and 25k score-0 boxes in
                   this split, so the choice moves the numbers materially.
``iou_thresh``     the association gate, 0.5 by default.

Association is class-agnostic in both policies. Class labels are the detector's
problem; mixing them into the association would hide tracker fragmentation
behind class confusion, which is the opposite of what this measures.

--------------------------------------------------------------------------- #
REFERENCE NUMBERS

VisDrone-MOT test-dev, 13 sequences, 5434 frames, 1885 gt tracks.
YOLOv8n fine-tuned on VisDrone, imgsz 640, conf 0.10, class_policy mot5,
ignore_policy mask, iou_thresh 0.5:

    config                   IDF1   MOTA  IDSW   Rcll   Prcn   IDinf
    botsort                 60.96  39.38  1322  69.95  70.14    2.91
    botsort_buffer          60.88  39.29  1353  69.96  70.09    2.90
    tracktrack              59.72  43.79   202  48.21  91.81    0.76
    botsort_reid            58.53  29.96  1850  71.61  63.85    4.25
    deepocsort              54.02  41.17  2428  63.81  75.07    3.35
    ocsort                  53.31  39.55  2582  64.82  73.21    3.55
    botsort_nogmc           52.93  38.09  2421  66.98  70.94    3.58
    deepocsort_gmc_reid     45.66  10.56  3714  58.45  56.14    1.78

Three things that table settles, all of them counter-intuitive:

1. **Camera compensation is most of the tracker's value on drone footage.**
   ``ocsort`` (53.31) lands within 0.4 of ``botsort_nogmc`` (52.93), and
   Ultralytics' ``ocsort.yaml`` has no ``gmc_method`` key at all -- OC-SORT
   never imports GMC. The whole 7.7-point gap to BoT-SORT is one variable.
2. **Appearance ReID makes it worse at these object sizes.** Most VisDrone
   boxes are under 32x32 px; an embedding pooled from such a crop is closer to
   noise than to an identity, and a wrong appearance distance *rejects* IoU
   matches that were correct, spawning new tracks. ``botsort_reid`` is worse on
   every column than ``botsort``.
3. **"Enable everything" is not a superset.** ``deepocsort_gmc_reid`` turns on
   both GMC and ReID and finishes last by 7 points.

If you are training a detector, the number to watch is that a weak checkpoint
dominates everything above: the same TrackTrack config scored IDF1 59.72 with a
converged YOLOv8n and 28.59 with a 3-epoch one. Fix the detector before tuning
a tracker.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from glob import glob

import numpy as np
from scipy.optimize import linear_sum_assignment

# =========================================================================== #
# Taxonomy
# =========================================================================== #

# VisDrone category id (1-based, as written in the annotation files) -> class
# index (0-based, as a detector trained on this taxonomy emits).
CAT_TO_CLS = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9}

CLASS_NAMES = (
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
)

# The official VisDrone MOT challenge scores five categories, not ten.
MOT_EVAL_CATS = (1, 4, 5, 6, 9)          # pedestrian, car, van, truck, bus
MOT_EVAL_CLASSES = tuple(CAT_TO_CLS[c] for c in MOT_EVAL_CATS)


# =========================================================================== #
# Tracker configurations
# =========================================================================== #
#
# Ultralytics resolves a bare "botsort.yaml" to its own shipped default, so a
# tuned value cannot be passed as a flag -- it has to be a file. These presets
# are written to a temp directory on demand, which keeps this script to one
# file. Each changes exactly one thing against its baseline, so a difference in
# the output table attributes to a cause.

BUILTIN_TRACKERS = (
    "botsort", "bytetrack", "ocsort", "deepocsort", "tracktrack", "fasttrack",
)

TRACKER_PRESETS: dict[str, str] = {
    # BoT-SORT without camera compensation. The control: if optical-flow GMC
    # matters on your footage, this row must lose IDF1 against botsort.
    "botsort_nogmc": """
tracker_type: botsort
track_high_thresh: 0.25
track_low_thresh: 0.1
new_track_thresh: 0.25
track_buffer: 30
match_thresh: 0.8
fuse_score: True
gmc_method: none
proximity_thresh: 0.5
appearance_thresh: 0.8
with_reid: False
model: auto
""",
    # BoT-SORT with appearance ReID. model "auto" hooks the Detect layer of a
    # torch .pt and reuses features the forward pass already computed, so no
    # second network runs. Pointed at an ONNX graph it cannot do that and falls
    # back to downloading a separate classifier.
    "botsort_reid": """
tracker_type: botsort
track_high_thresh: 0.25
track_low_thresh: 0.1
new_track_thresh: 0.25
track_buffer: 30
match_thresh: 0.8
fuse_score: True
gmc_method: sparseOptFlow
proximity_thresh: 0.5
appearance_thresh: 0.8
with_reid: True
model: auto
""",
    # Longer lost-track buffer: 30 frames is 1.0 s at 30 fps, which may retire a
    # track before an occlusion ends.
    "botsort_buffer": """
tracker_type: botsort
track_high_thresh: 0.25
track_low_thresh: 0.1
new_track_thresh: 0.25
track_buffer: 60
match_thresh: 0.8
fuse_score: True
gmc_method: sparseOptFlow
proximity_thresh: 0.5
appearance_thresh: 0.8
with_reid: False
model: auto
""",
    # Deep OC-SORT with the two features its shipped default leaves off.
    # ultralytics' deepocsort.yaml defaults to gmc_method: none and
    # with_reid: False, which makes plain --tracker deepocsort *weaker* than
    # BoT-SORT rather than stronger. Only this variant is a fair comparison.
    "deepocsort_gmc_reid": """
tracker_type: deepocsort
track_high_thresh: 0.3
track_low_thresh: 0.1
new_track_thresh: 0.3
track_buffer: 30
match_thresh: 0.8
fuse_score: True
delta_t: 3
inertia: 0.2
use_byte: False
gmc_method: sparseOptFlow
with_reid: True
model: auto
proximity_thresh: 0.5
appearance_thresh: 0.9
alpha_fixed_emb: 0.95
""",
}

_preset_dir: str | None = None


def resolve_tracker(name: str) -> tuple[str, str]:
    """``(label, config path or builtin name)`` for a tracker argument."""
    global _preset_dir

    if name in BUILTIN_TRACKERS:
        return name, f"{name}.yaml"

    if name in TRACKER_PRESETS:
        if _preset_dir is None:
            _preset_dir = tempfile.mkdtemp(prefix="mot_eval_trackers_")
        path = os.path.join(_preset_dir, f"{name}.yaml")
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                f.write(TRACKER_PRESETS[name].lstrip())
        return name, path

    if os.path.isfile(name):
        return os.path.splitext(os.path.basename(name))[0], name

    raise SystemExit(
        f"[fatal] unknown tracker {name!r}\n"
        f"        builtin: {', '.join(BUILTIN_TRACKERS)}\n"
        f"        presets: {', '.join(sorted(TRACKER_PRESETS))}\n"
        f"        or a path to a tracker YAML"
    )


# =========================================================================== #
# Ground truth
# =========================================================================== #


@dataclass
class Frame:
    """One frame of ground truth."""

    index: int                            # 1-based, matches the annotation file
    path: str
    boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), np.float32))
    ids: np.ndarray = field(default_factory=lambda: np.zeros((0,), np.int64))
    classes: np.ndarray = field(default_factory=lambda: np.zeros((0,), np.int32))
    ignore_boxes: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 4), np.float32)
    )


@dataclass
class Sequence:
    name: str
    frames: list[Frame]

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def n_gt_boxes(self) -> int:
        return sum(len(f.boxes) for f in self.frames)

    @property
    def n_gt_ids(self) -> int:
        return len({int(i) for f in self.frames for i in f.ids})


def parse_mot_annotation(path: str) -> dict[int, tuple[list, list, list, list]]:
    """Group one VisDrone-MOT annotation file by frame index.

    Line format -- note it is *not* the VisDrone-DET format::

        frame_index,target_id,bbox_left,bbox_top,bbox_width,bbox_height,
        score,category,truncation,occlusion

    Three kinds of line become *ignore regions* rather than disappearing:

    * ``category == 0``  -- an explicitly masked-out area, not an object.
    * ``category == 11`` -- "others": a real object outside the taxonomy.
    * ``score == 0``     -- a box the official protocol excludes from scoring.

    The distinction matters: a track sitting in an unannotated crowd is not a
    false positive, and deleting the region silently turns it into one.
    """
    per_frame: dict[int, tuple[list, list, list, list]] = {}

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip().rstrip(",")
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 8:
                continue
            try:
                frame = int(float(parts[0]))
                tid = int(float(parts[1]))
                x, y, w, h = (float(parts[i]) for i in range(2, 6))
                score = float(parts[6])
                cat = int(float(parts[7]))
            except ValueError:
                continue
            if w <= 0 or h <= 0:
                continue

            box = [x, y, x + w, y + h]
            boxes, ids, classes, ignores = per_frame.setdefault(
                frame, ([], [], [], [])
            )
            if cat == 0 or cat == 11 or cat not in CAT_TO_CLS or score == 0:
                ignores.append(box)
                continue
            boxes.append(box)
            ids.append(tid)
            classes.append(CAT_TO_CLS[cat])

    return per_frame


def list_sequences(root: str) -> list[str]:
    """Sequence names that have *both* frames and annotations."""
    seq_dir = os.path.join(root, "sequences")
    ann_dir = os.path.join(root, "annotations")
    if not os.path.isdir(seq_dir):
        raise SystemExit(
            f"[fatal] no sequences/ under {root}\n"
            f"        expected {root}/sequences/<name>/*.jpg and "
            f"{root}/annotations/<name>.txt"
        )
    names, orphans = [], []
    for name in sorted(os.listdir(seq_dir)):
        if not os.path.isdir(os.path.join(seq_dir, name)):
            continue
        if os.path.isfile(os.path.join(ann_dir, name + ".txt")):
            names.append(name)
        else:
            orphans.append(name)
    if orphans:
        print(f"[warn] {len(orphans)} sequence(s) have no annotation and are "
              f"skipped: {', '.join(orphans)}")
    return names


def load_sequence(root: str, name: str, limit: int | None = None) -> Sequence:
    """Load one sequence: ordered frame paths plus per-frame ground truth."""
    paths = sorted(glob(os.path.join(root, "sequences", name, "*.jpg")))
    if not paths:
        raise SystemExit(f"[fatal] no .jpg frames for sequence {name}")
    if limit:
        paths = paths[:limit]

    per_frame = parse_mot_annotation(os.path.join(root, "annotations", name + ".txt"))

    empty = ([], [], [], [])
    frames = []
    for i, path in enumerate(paths, start=1):
        boxes, ids, classes, ignores = per_frame.get(i, empty)
        frames.append(Frame(
            index=i,
            path=path,
            boxes=np.asarray(boxes, np.float32).reshape(-1, 4),
            ids=np.asarray(ids, np.int64).reshape(-1),
            classes=np.asarray(classes, np.int32).reshape(-1),
            ignore_boxes=np.asarray(ignores, np.float32).reshape(-1, 4),
        ))
    return Sequence(name, frames)


def filter_classes(frame: Frame, keep: tuple[int, ...] | None) -> Frame:
    """Restrict ground truth to ``keep``, moving the rest into ignore regions."""
    if keep is None or len(frame.classes) == 0:
        return frame
    mask = np.isin(frame.classes, np.asarray(keep, np.int32))
    if mask.all():
        return frame
    return Frame(
        index=frame.index,
        path=frame.path,
        boxes=frame.boxes[mask],
        ids=frame.ids[mask],
        classes=frame.classes[mask],
        ignore_boxes=np.concatenate([frame.ignore_boxes, frame.boxes[~mask]]),
    )


def keep_mask(
    boxes: np.ndarray, ignore_boxes: np.ndarray, overlap_thresh: float = 0.5,
) -> np.ndarray:
    """Which boxes survive the ignore regions. ``True`` = keep.

    Uses intersection-over-*detection-area*, not IoU: a small detection fully
    inside a large ignored region has a tiny IoU but should still be dropped.
    """
    if boxes.shape[0] == 0:
        return np.zeros((0,), bool)
    if ignore_boxes.shape[0] == 0:
        return np.ones((boxes.shape[0],), bool)

    d_area = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * \
             np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    lt = np.maximum(boxes[:, None, :2], ignore_boxes[None, :, :2])
    rb = np.minimum(boxes[:, None, 2:4], ignore_boxes[None, :, 2:4])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]
    return inter.max(axis=1) / np.maximum(d_area, 1e-9) < overlap_thresh


# =========================================================================== #
# Metrics: CLEAR MOT (Bernardin 2008) + identity metrics (Ristani 2016)
# =========================================================================== #

_UNMATCHED = 1e6          # assignment cost for a pair below the IoU gate


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    area_a = np.maximum(0.0, a[:, 2] - a[:, 0]) * np.maximum(0.0, a[:, 3] - a[:, 1])
    area_b = np.maximum(0.0, b[:, 2] - b[:, 0]) * np.maximum(0.0, b[:, 3] - b[:, 1])
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:4], b[None, :, 2:4])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-12)


@dataclass
class MotCounts:
    """Raw accumulator state. Sums across sequences; ratios never do."""

    n_frames: int = 0
    n_gt: int = 0
    n_hyp: int = 0
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

    Two implementation details decide whether these numbers match other tools:

    1. **Match continuity.** A frame's association is not a fresh Hungarian
       solve. Pairs matched in the previous frame that still clear the IoU gate
       are kept first, and only the remainder is solved. Without this, two
       overlapping objects trade hypotheses whenever the optimum is a tie, and
       IDSW counts arbitration noise instead of tracker failures.
    2. **IDF1 is global.** It comes from one assignment between ground-truth
       ids and hypothesis ids over the whole sequence, maximising co-located
       frames -- so a tracker is not punished for choosing different id numbers
       than the annotator, only for being inconsistent.
    """

    def __init__(self, iou_thresh: float = 0.5) -> None:
        self.iou_thresh = float(iou_thresh)
        self.c = MotCounts()
        self._last_hyp: dict[int, int] = {}
        self._tracked_prev: set[int] = set()
        self._gt_present: dict[int, int] = {}
        self._gt_tracked: dict[int, int] = {}
        self._hyp_ids: set[int] = set()
        self._cooc: dict[tuple[int, int], int] = {}

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
        # picked it: IDF1's assignment is global and must see the full matrix.
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

        # Fragmentation: an object that was tracked, stopped being tracked, and
        # is tracked again. Counted on the resumption.
        for g in tracked_now - self._tracked_prev:
            if g in self._last_hyp and self._gt_tracked.get(g, 0) > 1:
                self.c.fm += 1
        self._tracked_prev = tracked_now

    def _associate(
        self,
        gt_ids: np.ndarray,
        hyp_ids: np.ndarray,
        iou: np.ndarray,
        valid: np.ndarray,
    ) -> list[tuple[int, int]]:
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
            rows, cols = linear_sum_assignment(
                np.where(sub_valid, -sub, _UNMATCHED)
            )
            for r, c in zip(rows, cols):
                if sub_valid[r, c]:
                    matches.append((int(gi_rest[r]), int(gj_rest[c])))
        return matches

    def finalize(self) -> MotCounts:
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
        # Hypothesis identities spent per real object. 1.0 is perfect.
        "id_inflation": c.n_hyp_ids / n_ids,
        "idsw_per_100_frames": 100.0 * c.idsw / max(c.n_frames, 1),
    }


FORMAT_HEADER = (
    f"{'sequence':24s} {'IDF1':>6s} {'MOTA':>7s} {'MOTP':>6s} {'IDSW':>6s} "
    f"{'FM':>6s} {'FP':>7s} {'FN':>7s} {'MT':>5s} {'ML':>5s} {'IDinf':>6s}"
)


def format_row(name: str, s: dict) -> str:
    return (f"{name:24s} {s['idf1']*100:6.2f} {s['mota']*100:7.2f} "
            f"{s['motp_iou']*100:6.2f} {s['idsw']:6d} {s['fm']:6d} "
            f"{s['fp']:7d} {s['fn']:7d} {s['mt']:5d} {s['ml']:5d} "
            f"{s['id_inflation']:6.2f}")


# =========================================================================== #
# Output
# =========================================================================== #


def append_rows(path: str, rows: list[dict]) -> int:
    """Append to a CSV, refusing to write under a header that does not match.

    ``csv.DictWriter`` writes values in *its* field order and skips the header
    when the file exists. Point two scripts with different columns at one path
    and the second one's rows land under the first one's header, shifted -- a
    file that parses cleanly and means something else entirely. A benchmark that
    quietly reports the wrong number is worse than one that refuses to run.
    """
    if not rows:
        return 0

    fieldnames = list(rows[0].keys())
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    exists = os.path.exists(path) and os.path.getsize(path) > 0

    if exists:
        with open(path, "r", newline="", encoding="utf-8") as f:
            existing = next(csv.reader(f), [])
        if existing != fieldnames:
            missing = [c for c in existing if c not in fieldnames]
            extra = [c for c in fieldnames if c not in existing]
            raise SystemExit(
                f"[fatal] {path} has a different schema; appending would shift "
                f"every column.\n"
                f"        file has {len(existing)} columns, these rows have "
                f"{len(fieldnames)}\n"
                + (f"        only in file: {missing}\n" if missing else "")
                + (f"        only in rows: {extra}\n" if extra else "")
                + f"        write to a different --out, or delete {path}"
            )

    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerows(rows)
    return len(rows)


def write_mot_results(path: str, rows: list[tuple]) -> None:
    """Write hypotheses in MOTChallenge format, for reuse by other tools."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for frame, tid, x, y, w, h, score, cls in rows:
            f.write(f"{int(frame)},{int(tid)},{x:.2f},{y:.2f},{w:.2f},{h:.2f},"
                    f"{score:.4f},{int(cls)},-1,-1\n")


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


def sha256_short(path: str, n: int = 16) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


# =========================================================================== #
# Running a tracker
# =========================================================================== #


def run_sequence(model, seq: Sequence, tracker_cfg: str, args, keep_classes):
    """Track one sequence end to end and score it.

    The first frame goes in with ``persist=False`` so Ultralytics rebuilds the
    tracker. State carried over from the previous sequence would let ids survive
    a scene cut and score as identity errors the tracker never made.
    """
    import cv2

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
            verbose=False,
            **args.quantize_kw,
        )[0]
        per_frame_ms.append((time.perf_counter() - t0) * 1000.0)

        boxes = result.boxes
        if boxes is not None and boxes.id is not None:
            hyp_xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
            hyp_ids = boxes.id.cpu().numpy().astype(np.int64)
            hyp_cls = boxes.cls.cpu().numpy().astype(np.int32)
            hyp_conf = boxes.conf.cpu().numpy().astype(np.float32)
        else:
            # No ids means no confirmed tracks -- normal for the first frames of
            # a sequence, not an error.
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


def _row(label: str, cfg: str, sequence: str, ms: float, s: dict) -> dict:
    """One output row, in one column order.

    Both the tracking run and the rescore path go through here. They otherwise
    produce different columns, and a CSV written by one would be refused -- or
    worse, silently shifted -- by the other.
    """
    return {"tracker": label, "tracker_config": cfg, "sequence": sequence,
            "ms_per_frame_avg": ms, **s}


def score_saved(label: str, hyp_dir: str, sequences, args) -> list[dict]:
    """Score hypotheses already on disk -- no model, no GPU."""
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
        rows.append(_row(label, "", seq.name, float("nan"), s))

    if rows:
        s = summarize(overall)
        print("-" * len(FORMAT_HEADER))
        print(format_row("OVERALL", s))
        rows.append(_row(label, "", "OVERALL", float("nan"), s))
    return rows


# =========================================================================== #
# Self-test
# =========================================================================== #


def selftest() -> int:
    """Check the metric definitions on cases whose answers are arithmetic.

    Worth running once on a new machine: these definitions are implemented here
    rather than pulled from ``motmetrics`` or ``TrackEval``, and a silently
    wrong metric is the most expensive kind of bug in a benchmark.
    """
    def box(x, y, w=10, h=10):
        return [x, y, x + w, y + h]

    def run(frames, thresh=0.5):
        acc = MotAccumulator(iou_thresh=thresh)
        for gt_ids, gt_boxes, h_ids, h_boxes in frames:
            acc.update(np.array(gt_ids),
                       np.array(gt_boxes, np.float32).reshape(-1, 4),
                       np.array(h_ids),
                       np.array(h_boxes, np.float32).reshape(-1, 4))
        return summarize(acc.finalize())

    N = 20
    ok = True

    def check(name, got, want, tol=1e-6):
        nonlocal ok
        good = abs(got - want) <= tol
        ok &= good
        print(f"{'PASS' if good else 'FAIL'}  {name:38s} "
              f"got {got:.4f}  want {want:.4f}")

    # 1. Perfect tracking.
    f = [([1], [box(i * 2, 0)], [7], [box(i * 2, 0)]) for i in range(N)]
    s = run(f)
    check("perfect: idf1", s["idf1"], 1.0)
    check("perfect: mota", s["mota"], 1.0)
    check("perfect: idsw", s["idsw"], 0)
    check("perfect: motp_iou", s["motp_iou"], 1.0)
    check("perfect: id_inflation", s["id_inflation"], 1.0)

    # 2. One id switch halfway. MOTA charges it once, IDF1 charges every frame
    #    the second identity was wrong -- the reason both are reported.
    f = [([1], [box(i * 2, 0)], [7 if i < N // 2 else 8], [box(i * 2, 0)])
         for i in range(N)]
    s = run(f)
    check("switch: idsw", s["idsw"], 1)
    check("switch: mota", s["mota"], 1.0 - 1 / N)
    check("switch: idf1", s["idf1"], 2 * (N // 2) / (2 * (N // 2) + 10 + 10))
    check("switch: recall", s["recall"], 1.0)
    check("switch: id_inflation", s["id_inflation"], 2.0)

    # 3. A detection gap: FN and fragmentation, no id switch.
    f = []
    for i in range(N):
        gone = 8 <= i < 12
        f.append(([1], [box(i * 2, 0)],
                  [] if gone else [7], [] if gone else [box(i * 2, 0)]))
    s = run(f)
    check("gap: fn", s["fn"], 4)
    check("gap: fp", s["fp"], 0)
    check("gap: idsw", s["idsw"], 0)
    check("gap: fm", s["fm"], 1)
    check("gap: mota", s["mota"], 1.0 - 4 / N)
    check("gap: mt (16/20 tracked)", s["mt"], 1)

    # 4. A ghost track alongside a perfect one.
    f = [([1], [box(0, 0)], [7, 9], [box(0, 0), box(500, 500)]) for _ in range(N)]
    s = run(f)
    check("ghost: fp", s["fp"], N)
    check("ghost: mota", s["mota"], 0.0)
    check("ghost: idp", s["idp"], 0.5)
    check("ghost: idr", s["idr"], 1.0)

    # 5. Consistent relabelling is not an error: IDF1 solves a global
    #    assignment, so different id *numbers* score perfect.
    f = [([1, 2], [box(0, 0), box(100, 100)], [9, 8], [box(0, 0), box(100, 100)])
         for _ in range(N)]
    s = run(f)
    check("relabel: idf1", s["idf1"], 1.0)
    check("relabel: idsw", s["idsw"], 0)

    # 6. Just outside the IoU gate: FP + FN, not a match.
    f = [([1], [box(0, 0)], [7], [box(9, 0)]) for _ in range(N)]
    s = run(f)
    check("below gate: tp", s["tp"], 0)
    check("below gate: fp", s["fp"], N)
    check("below gate: fn", s["fn"], N)
    check("below gate: mota", s["mota"], 1.0 - 2 * N / N)

    print("\nALL PASS" if ok else "\nFAILURES PRESENT")
    return 0 if ok else 1


# =========================================================================== #
# CLI
# =========================================================================== #


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="VisDrone-MOT tracking benchmark (IDF1 / MOTA / IDSW).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="presets: " + ", ".join(sorted(TRACKER_PRESETS)),
    )
    ap.add_argument("--selftest", action="store_true",
                    help="verify the metric implementation and exit")
    ap.add_argument("--score-only", default="",
                    help="score saved hypotheses in this directory "
                         "(<tracker>/<sequence>.txt); no model needed")

    ap.add_argument("--data", default="test/VisDrone2019-MOT-test-dev",
                    help="split root holding sequences/ and annotations/")
    ap.add_argument("--model", default="",
                    help="detector weights. Prefer a torch .pt: with_reid=auto "
                         "needs its Detect features, which an ONNX graph cannot "
                         "supply")
    ap.add_argument("--trackers", nargs="+", default=["botsort"],
                    help=f"{' '.join(BUILTIN_TRACKERS)}, a preset name, or a "
                         f"path to a tracker YAML")
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
                    help="0 = all frames; a small value smoke-tests the setup")
    ap.add_argument("--out", default="results/tracking_quality.csv")
    ap.add_argument("--save-mot", default="results/mot",
                    help="directory for MOTChallenge hypotheses; '' skips")
    ap.add_argument("--progress-every", type=int, default=200)
    ap.add_argument("--tag", default="", help="free-text label for this run")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()

    # ---------------------------------------------------------------- data
    names = list_sequences(args.data)
    if args.sequences:
        unknown = set(args.sequences) - set(names)
        if unknown:
            raise SystemExit(f"[fatal] not scorable here: {sorted(unknown)}\n"
                             f"        available: {names}")
        names = [n for n in names if n in args.sequences]
    if not names:
        raise SystemExit(f"[fatal] no scorable sequences under {args.data}")

    sequences = [
        load_sequence(args.data, n, limit=args.limit_frames or None) for n in names
    ]
    args.n_sequences = len(sequences)
    print(f"[info] {len(sequences)} sequences, "
          f"{sum(len(s) for s in sequences)} frames, "
          f"{sum(s.n_gt_ids for s in sequences)} gt ids, "
          f"{sum(s.n_gt_boxes for s in sequences)} gt boxes "
          f"(before the {args.class_policy} class policy)")

    rows: list[dict] = []

    # ------------------------------------------------- score-only shortcut
    if args.score_only:
        labels = sorted(
            d for d in os.listdir(args.score_only)
            if os.path.isdir(os.path.join(args.score_only, d))
        )
        if not labels:
            raise SystemExit(f"[fatal] no tracker subdirectories under "
                             f"{args.score_only}")
        for label in labels:
            rows.extend(score_saved(
                label, os.path.join(args.score_only, label), sequences, args
            ))
        _stamp(rows, args, model="", model_hash="",
               tag=args.tag or "rescored from saved hypotheses")
        _write(args, rows)
        return 0

    # ------------------------------------------------------------- model
    if not args.model:
        raise SystemExit("[fatal] --model is required (or use --score-only)")
    if not os.path.isfile(args.model):
        raise SystemExit(f"[fatal] weights not found: {args.model}")

    import cv2
    from ultralytics import YOLO

    model = YOLO(args.model)
    print(f"[info] {os.path.basename(args.model)} "
          f"({sha256_short(args.model)}) | {len(model.names)} classes | "
          f"imgsz {args.imgsz} | conf {args.conf}")
    if len(model.names) != len(CLASS_NAMES) or \
            tuple(model.names[i] for i in sorted(model.names)) != CLASS_NAMES:
        print(f"[warn] model classes differ from the VisDrone taxonomy this "
              f"script maps ground truth to:\n"
              f"       model: {model.names}\n"
              f"       expected: {dict(enumerate(CLASS_NAMES))}\n"
              f"       association is class-agnostic so the totals still mean "
              f"something, but --class-policy mot5 selects ground-truth classes "
              f"by index and will select the wrong ones.")

    # ultralytics 8.4 replaced the boolean `half` with `quantize`, which names
    # the bit width. Support both so this runs on either.
    import inspect
    sig = inspect.signature(model.track)
    params = set(sig.parameters)
    if "quantize" in params or any(p.kind == p.VAR_KEYWORD
                                   for p in sig.parameters.values()):
        try:
            from ultralytics.cfg import DEFAULT_CFG_DICT
            use_quantize = "quantize" in DEFAULT_CFG_DICT
        except Exception:
            use_quantize = True
    else:
        use_quantize = False
    args.quantize_kw = ({"quantize": 16 if args.half else None}
                        if use_quantize else {"half": args.half})

    # Warm up before any timing: the first forward pass pays CUDA context setup
    # and cuDNN autotuning, which would otherwise land on whichever tracker runs
    # first and make it look several times slower than the rest.
    warm = cv2.imread(sequences[0].frames[0].path)
    for _ in range(3):
        model.predict(warm, imgsz=args.imgsz, conf=args.conf, iou=args.nms_iou,
                      max_det=args.max_det, device=args.device, verbose=False,
                      **args.quantize_kw)

    keep_classes = MOT_EVAL_CLASSES if args.class_policy == "mot5" else None
    model_hash = sha256_short(args.model)

    # --------------------------------------------------------------- run
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

            rows.append(_row(label, cfg, seq.name,
                             float(np.mean(ms)) if ms else float("nan"), s))

        wall = time.perf_counter() - t_wall
        s = summarize(overall)
        print("-" * len(FORMAT_HEADER))
        print(format_row("OVERALL", s))
        print(f"    {np.mean(all_ms):.1f} ms/frame avg, {wall:.0f}s wall")
        rows.append(_row(label, cfg, "OVERALL",
                         float(np.mean(all_ms)) if all_ms else float("nan"), s))

    print("\n[note] the latency column is a by-product, not a benchmark: a long "
          "sweep on a laptop varies by several times as clocks throttle. "
          "Measure speed separately on an idle machine.")

    _stamp(rows, args, model=os.path.basename(args.model),
           model_hash=model_hash, tag=args.tag)
    _write(args, rows)
    return 0


def _stamp(rows: list[dict], args, model: str, model_hash: str, tag: str) -> None:
    """Stamp every row with the run conditions.

    A MOT number is not comparable to another one without them, so they travel
    in the row rather than in a filename or a lab notebook.
    """
    for r in rows:
        r.update({
            "model": model,
            "model_sha256_16": model_hash,
            "imgsz": args.imgsz,
            "conf_thres": args.conf,
            "nms_iou": args.nms_iou,
            "max_det": args.max_det,
            "half": args.half,
            "iou_thresh": args.iou_thresh,
            "class_policy": args.class_policy,
            "ignore_policy": args.ignore_policy,
            "n_sequences": args.n_sequences,
            "tag": tag,
        })


def _write(args, rows: list[dict]) -> None:
    if not rows:
        return
    print(f"\n[ok] appended {append_rows(args.out, rows)} row(s) -> {args.out}")
    json_out = os.path.splitext(args.out)[0] + ".json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"[ok] {json_out}")


if __name__ == "__main__":
    raise SystemExit(main())
