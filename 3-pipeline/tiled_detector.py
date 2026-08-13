#!/usr/bin/env python3
"""Tiled inference: run the detector on overlapping crops, merge the results.

Why this exists. A 1920x1080 frame letterboxed to 640 is scaled by 0.33, so a
15 px pedestrian arrives at the network as 5 px, and mosaic augmentation halves
it again during training. A stride-8 feature map cannot represent an object
that small -- the label is asking for something the architecture cannot express.

Cutting the frame into a 2x2 grid with 20% overlap gives 1067x600 tiles, which
letterbox to 640 at a scale of 0.60 instead of 0.33. Every object arrives
**1.80x larger**, independent of its size. The cost is exactly four forward
passes per frame instead of one, plus the merge.

Three details decide whether the merge is correct, and all three are silent
when wrong:

**Overlap must exceed the largest object.** Otherwise an object can be cut by
every tile that sees it, and no tile holds it whole. At 20% the overlap band is
214 px on a 1920-wide frame; objects larger than that need a bigger overlap.

**Truncated detections must be dropped, not merged.** A car cut in half by a
tile edge produces a confident half-car box. NMS will not remove it, because
its IoU against the whole-car box from the neighbouring tile is around 0.5 --
below any sane threshold. So detections touching an *interior* tile edge are
dropped outright; the overlap guarantees a neighbouring tile saw the object
whole. Detections touching the *image* edge are kept, because there is no
neighbour and truncation there is real.

**Merging must be per class.** Class-agnostic NMS across tiles would delete a
motorbike that legitimately overlaps a car.

    from detector import Yolov8Detector
    from tiled_detector import TiledDetector

    det = TiledDetector(Yolov8Detector(session), grid=2, overlap=0.20)
    d = det(frame)          # same Detections contract as the plain detector
"""

from __future__ import annotations

import time

import numpy as np

from detector import Detections, nms_numpy


def tile_rects(w: int, h: int, grid: int = 2, overlap: float = 0.20
               ) -> list[tuple[int, int, int, int]]:
    """Tile origins and size for a grid x grid split with fractional overlap.

    ``tw * (grid - overlap) == w`` places the tiles so that the union covers the
    frame exactly, with neighbours sharing ``overlap * tw`` pixels. The last
    tile is pinned to the right/bottom edge so rounding never leaves a seam.
    """
    if grid < 1:
        raise ValueError("grid must be >= 1")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")
    if grid == 1:
        return [(0, 0, w, h)]

    tw = int(round(w / (grid - overlap)))
    th = int(round(h / (grid - overlap)))
    xs = [int(round((w - tw) * i / (grid - 1))) for i in range(grid)]
    ys = [int(round((h - th) * i / (grid - 1))) for i in range(grid)]
    return [(x, y, tw, th) for y in ys for x in xs]


class TiledDetector:
    """Wraps any detector exposing ``__call__(img) -> Detections``."""

    def __init__(
        self,
        detector,
        grid: int = 2,
        overlap: float = 0.20,
        merge_iou: float = 0.55,
        edge_margin_px: int = 2,
        drop_truncated: bool = True,
        max_det: int = 500,
    ) -> None:
        self.detector = detector
        self.grid = int(grid)
        self.overlap = float(overlap)
        self.merge_iou = float(merge_iou)
        self.edge_margin_px = int(edge_margin_px)
        self.drop_truncated = bool(drop_truncated)
        self.max_det = int(max_det)

    # ------------------------------------------------------------------ #
    def _keep_mask(self, xyxy, tx, ty, tw, th, W, H):
        """False for boxes touching an interior tile edge.

        The image border is exempt: an object cut off by the frame itself is
        genuinely cut off, and dropping it would lose every detection along the
        outside of the picture.
        """
        m = self.edge_margin_px
        keep = np.ones(len(xyxy), dtype=bool)
        if not self.drop_truncated:
            return keep

        touch_l = xyxy[:, 0] <= tx + m
        touch_t = xyxy[:, 1] <= ty + m
        touch_r = xyxy[:, 2] >= tx + tw - m
        touch_b = xyxy[:, 3] >= ty + th - m

        # An edge is "interior" when the tile does not sit against the frame.
        if tx > 0:
            keep &= ~touch_l
        if ty > 0:
            keep &= ~touch_t
        if tx + tw < W:
            keep &= ~touch_r
        if ty + th < H:
            keep &= ~touch_b
        return keep

    # ------------------------------------------------------------------ #
    def __call__(self, img: np.ndarray) -> Detections:
        H, W = img.shape[:2]
        rects = tile_rects(W, H, self.grid, self.overlap)

        boxes, confs, clses = [], [], []
        pre = inf = post = 0.0
        t0 = time.perf_counter()

        for (tx, ty, tw, th) in rects:
            crop = img[ty:ty + th, tx:tx + tw]
            d = self.detector(crop)
            pre += d.pre_ms
            inf += d.infer_ms
            post += d.post_ms
            if len(d) == 0:
                continue

            xyxy = d.xyxy.copy()
            xyxy[:, [0, 2]] += tx           # tile coords -> frame coords
            xyxy[:, [1, 3]] += ty

            keep = self._keep_mask(xyxy, tx, ty, tw, th, W, H)
            if keep.any():
                boxes.append(xyxy[keep])
                confs.append(d.conf[keep])
                clses.append(d.cls[keep])

        if not boxes:
            z = np.zeros((0, 4), np.float32)
            return Detections(z, np.zeros((0,), np.float32),
                              np.zeros((0,), np.int32), pre, inf, post)

        xyxy = np.concatenate(boxes).astype(np.float32)
        conf = np.concatenate(confs).astype(np.float32)
        cls = np.concatenate(clses).astype(np.int32)

        # Per-class merge via the offset trick: shifting each class into its own
        # coordinate band makes one NMS pass behave as one pass per class.
        t1 = time.perf_counter()
        offset = cls.astype(np.float32) * (max(W, H) * 4.0)
        keep = nms_numpy(xyxy + offset[:, None], conf, self.merge_iou)
        keep = keep[: self.max_det]
        merge_ms = (time.perf_counter() - t1) * 1000.0

        return Detections(
            xyxy[keep], conf[keep], cls[keep],
            pre_ms=pre, infer_ms=inf, post_ms=post + merge_ms,
        )

    # ------------------------------------------------------------------ #
    @property
    def n_tiles(self) -> int:
        return self.grid * self.grid

    def scale_gain(self, w: int, h: int, imgsz: int = 640) -> float:
        """How much larger an object arrives at the network, versus untiled."""
        _, _, tw, th = tile_rects(w, h, self.grid, self.overlap)[0]
        return (imgsz / max(tw, th)) / (imgsz / max(w, h))
