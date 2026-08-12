"""VisDrone-DET loader.

Annotation format, one object per line:

    bbox_left,bbox_top,bbox_width,bbox_height,score,category,truncation,occlusion

``category`` is 1-indexed with two special values:

    0   ignored region  — a masked-out area, not an object
    11  others          — an object that is not in the 10-class taxonomy

Both are excluded from the 10 evaluated classes, but they are *not* the same
thing and must not be handled the same way. A detection that lands inside an
ignored region is not a false positive; the region is simply not annotated.
The standard Ultralytics conversion script drops both silently, which makes
every detection in a crowd marked "ignored" count against you and depresses AP
by a margin that varies per image.

This loader keeps the ignore regions so the benchmark can report AP *both*
ways and say which one it is quoting.

``score`` is 0 for boxes that the official protocol excludes; we honour it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from glob import glob

import numpy as np

# VisDrone category id (1-based) -> our class index (0-based)
CAT_TO_CLS = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9}

CLASS_NAMES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]


@dataclass
class Sample:
    image_id: str
    image_path: str
    boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), np.float32))
    classes: np.ndarray = field(default_factory=lambda: np.zeros((0,), np.int32))
    ignore_boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), np.float32))

    def __repr__(self) -> str:  # pragma: no cover
        return (f"Sample({self.image_id}, {len(self.boxes)} objs, "
                f"{len(self.ignore_boxes)} ignore)")


def parse_annotation(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boxes, classes, ignores = [], [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip().rstrip(",")
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 6:
                continue
            try:
                x, y, w, h = (float(parts[i]) for i in range(4))
                score = float(parts[4])
                cat = int(float(parts[5]))
            except ValueError:
                continue
            if w <= 0 or h <= 0:
                continue
            box = [x, y, x + w, y + h]

            if cat == 0:
                ignores.append(box)
                continue
            if cat == 11 or cat not in CAT_TO_CLS:
                # "others": not an evaluated class, but it *is* a real object,
                # so treat its area as ignored rather than as background.
                ignores.append(box)
                continue
            if score == 0:
                ignores.append(box)
                continue
            boxes.append(box)
            classes.append(CAT_TO_CLS[cat])

    return (
        np.asarray(boxes, np.float32).reshape(-1, 4),
        np.asarray(classes, np.int32).reshape(-1),
        np.asarray(ignores, np.float32).reshape(-1, 4),
    )


def load_split(root: str, limit: int | None = None) -> list[Sample]:
    """Load a VisDrone-DET split directory (``images/`` + ``annotations/``)."""
    img_dir = os.path.join(root, "images")
    ann_dir = os.path.join(root, "annotations")
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"no images/ under {root}")

    paths = sorted(glob(os.path.join(img_dir, "*.jpg")))
    if limit:
        paths = paths[:limit]

    samples = []
    for p in paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        ann = os.path.join(ann_dir, stem + ".txt")
        if os.path.exists(ann):
            b, c, ig = parse_annotation(ann)
        else:
            b = np.zeros((0, 4), np.float32)
            c = np.zeros((0,), np.int32)
            ig = np.zeros((0, 4), np.float32)
        samples.append(Sample(stem, p, b, c, ig))
    return samples


def filter_ignored(
    boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray,
    ignore_boxes: np.ndarray, overlap_thresh: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drop detections that sit mostly inside an ignored region.

    Uses intersection-over-*detection-area*, not IoU: a small detection fully
    inside a large ignored region has a tiny IoU but should still be dropped.
    """
    if boxes.shape[0] == 0 or ignore_boxes.shape[0] == 0:
        return boxes, scores, classes

    d_area = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * \
             np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    lt = np.maximum(boxes[:, None, :2], ignore_boxes[None, :, :2])
    rb = np.minimum(boxes[:, None, 2:4], ignore_boxes[None, :, 2:4])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]
    frac = inter.max(axis=1) / np.maximum(d_area, 1e-9)

    keep = frac < overlap_thresh
    return boxes[keep], scores[keep], classes[keep]


def dataset_stats(samples: list[Sample]) -> dict:
    n_obj = sum(len(s.boxes) for s in samples)
    n_ign = sum(len(s.ignore_boxes) for s in samples)
    per_cls = np.zeros(10, dtype=np.int64)
    areas = []
    for s in samples:
        for c in s.classes:
            per_cls[int(c)] += 1
        if len(s.boxes):
            w = s.boxes[:, 2] - s.boxes[:, 0]
            h = s.boxes[:, 3] - s.boxes[:, 1]
            areas.append(w * h)
    areas = np.concatenate(areas) if areas else np.zeros(0)
    return {
        "n_images": len(samples),
        "n_objects": n_obj,
        "n_ignore_regions": n_ign,
        "objects_per_image": n_obj / max(len(samples), 1),
        "per_class": {CLASS_NAMES[i]: int(per_cls[i]) for i in range(10)},
        "pct_small": float((areas < 32 ** 2).mean() * 100) if areas.size else 0.0,
        "pct_medium": float((((areas >= 32 ** 2) & (areas < 96 ** 2)).mean()) * 100)
        if areas.size else 0.0,
        "pct_large": float((areas >= 96 ** 2).mean() * 100) if areas.size else 0.0,
    }
