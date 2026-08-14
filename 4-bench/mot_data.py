"""VisDrone-MOT loader.

Annotation format, one object per line -- note it is *not* the DET format:

    frame_index,target_id,bbox_left,bbox_top,bbox_width,bbox_height,
    score,category,truncation,occlusion

``frame_index`` is 1-based and indexes ``sequences/<name>/0000001.jpg`` onwards.
``target_id`` is unique within a sequence, not across the split.

The exclusion rules follow ``visdrone_data.py`` so DET and MOT numbers in this
repository mean the same thing:

* ``category == 0``  -- ignored region: a masked-out area, not an object.
* ``category == 11`` -- "others": a real object outside the 10-class taxonomy.
* ``score == 0``     -- a box the official protocol excludes from scoring.

All three become *ignore regions* rather than disappearing. That distinction is
the whole point: a track sitting in an unannotated crowd is not a false
positive, and deleting the region silently turns it into one. 23,472 ignored
regions and 25,901 score-0 boxes are present in this split, so the choice moves
the numbers materially.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from glob import glob

import numpy as np

from visdrone_data import CAT_TO_CLS, CLASS_NAMES  # noqa: F401  (re-exported)

# The official VisDrone MOT challenge scores five categories, not ten. Quoting
# a number against published results requires saying which set it used.
MOT_EVAL_CATS = (1, 4, 5, 6, 9)          # pedestrian, car, van, truck, bus
MOT_EVAL_CLASSES = tuple(CAT_TO_CLS[c] for c in MOT_EVAL_CATS)


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

    def __repr__(self) -> str:  # pragma: no cover
        return (f"Sequence({self.name}, {len(self.frames)} frames, "
                f"{self.n_gt_ids} ids, {self.n_gt_boxes} boxes)")


def parse_mot_annotation(path: str) -> dict[int, tuple[list, list, list, list]]:
    """Group one annotation file by frame index.

    Returns ``{frame_index: (boxes, ids, classes, ignore_boxes)}`` with boxes in
    xyxy. Frames absent from the file are absent from the dict; the caller fills
    them in as empty, because "no annotation for this frame" and "no objects in
    this frame" are the same thing in this format.
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
    """Sequence names that have *both* frames and annotations.

    This split ships 17 annotation files but only 13 image directories. A
    sequence without frames cannot be run, and silently scoring it as all-misses
    would poison the aggregate, so it is left out here rather than downstream.
    """
    seq_dir = os.path.join(root, "sequences")
    ann_dir = os.path.join(root, "annotations")
    if not os.path.isdir(seq_dir):
        raise FileNotFoundError(f"no sequences/ under {root}")
    names = []
    for name in sorted(os.listdir(seq_dir)):
        if not os.path.isdir(os.path.join(seq_dir, name)):
            continue
        if os.path.isfile(os.path.join(ann_dir, name + ".txt")):
            names.append(name)
    return names


def load_sequence(root: str, name: str, limit: int | None = None) -> Sequence:
    """Load one sequence: ordered frame paths plus per-frame ground truth."""
    paths = sorted(glob(os.path.join(root, "sequences", name, "*.jpg")))
    if not paths:
        raise FileNotFoundError(f"no frames for sequence {name}")
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
    """Restrict ground truth to ``keep``, moving the rest into ignore regions.

    Dropping the other classes outright would make every correct track on a
    motorbike a false positive under the 5-class protocol. Treating them as
    ignored is what makes the two class policies comparable.
    """
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


def write_mot_results(path: str, rows: list[tuple]) -> None:
    """Write tracker output in MOTChallenge format, for reuse by other tools.

    ``rows`` are ``(frame, id, x, y, w, h, score, class_id)``. Keeping the raw
    per-frame hypotheses on disk means a metric can be recomputed, or a third
    party's evaluator pointed at it, without re-running the GPU.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for frame, tid, x, y, w, h, score, cls in rows:
            f.write(f"{int(frame)},{int(tid)},{x:.2f},{y:.2f},{w:.2f},{h:.2f},"
                    f"{score:.4f},{int(cls)},-1,-1\n")
