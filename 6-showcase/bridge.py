"""Glue between supervision's detection container and the repository's tracker.

**Why not ``sv.ByteTrack``.** supervision deprecated it in 0.28 and removes it in
0.31; the installed 0.30 emits a FutureWarning on construction. Writing new code
against an API with one release left is not worth it when
``3-pipeline/tracker.py`` is right there, and that tracker is the better fit
anyway:

* its constructor arguments map one-for-one onto the ``tracker:`` block of
  ``configs/pipeline.yaml``, so the frozen config fully determines behaviour
* it returns objects that already carry ``.track_id`` / ``.cls`` / ``.centroid``
  / ``.box`` / ``.history`` -- exactly what ``TrafficAnalytics`` was written
  against, so no adapter is needed in either direction
* it is what the board runs, so the tracking video shows the deployed behaviour
  rather than a second tracker's approximation of it

supervision keeps the jobs it is best at here: ``sv.Detections`` as the
detection container, ``from_ultralytics`` to fill it, the annotators behind
``--no-fast-draw``, and ``ColorPalette``.

What is left for this module is the detection->tracker handoff and the tracker
health statistics Task 4 reports.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
def as_arrays(dets) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``sv.Detections`` -> the ``(xyxy, conf, cls)`` triple the tracker takes."""
    n = len(dets)
    if n == 0:
        return (np.zeros((0, 4), np.float32), np.zeros((0,), np.float32),
                np.zeros((0,), np.int32))
    xyxy = np.asarray(dets.xyxy, dtype=np.float32).reshape(-1, 4)
    conf = (
        np.asarray(dets.confidence, dtype=np.float32).reshape(-1)
        if dets.confidence is not None else np.ones(n, np.float32)
    )
    cls = (
        np.asarray(dets.class_id).astype(np.int32).reshape(-1)
        if dets.class_id is not None else np.zeros(n, np.int32)
    )
    return xyxy, conf, cls


def build_tracker(tcfg: dict):
    """``3-pipeline/tracker.py``'s ByteTrack from the ``tracker:`` YAML block.

    The keys map one-for-one, but they are listed explicitly rather than
    splatted: a typo in the YAML would otherwise be swallowed as an unexpected
    keyword and the tracker would silently run on its defaults.

    ``Track.reset_ids()`` matters when several renderers run in one process
    (``render_all.py``): the id counter is class-global, so without a reset the
    second task would start numbering at whatever the first one reached and its
    "unique ids" figure would be meaningless.
    """
    from tracker import ByteTrack, Track       # 3-pipeline/tracker.py

    Track.reset_ids()
    return ByteTrack(
        track_high_thresh=float(tcfg.get("track_high_thresh", 0.50)),
        track_low_thresh=float(tcfg.get("track_low_thresh", 0.10)),
        new_track_thresh=float(tcfg.get("new_track_thresh", 0.60)),
        match_iou=float(tcfg.get("match_iou", 0.20)),
        match_iou_low=float(tcfg.get("match_iou_low", 0.50)),
        track_buffer=int(tcfg.get("track_buffer", 30)),
        min_box_area=float(tcfg.get("min_box_area", 4.0)),
        min_hits=int(tcfg.get("min_hits", 3)),
    )


# --------------------------------------------------------------------------- #
class TrackStats:
    """Id churn and track age -- the tracker-quality signals available here.

    This sequence ships no MOT ground truth, so MOTA and IDF1 cannot be computed
    and are not claimed. What can be measured without ground truth is how long
    identities survive and how many new ones appear per frame. A tracker that
    fragments shows it as a high, spiky new-id rate together with an age
    histogram piled into the lowest bucket -- which is the failure this readout
    exists to make visible.
    """

    def __init__(self) -> None:
        self._first_seen: dict[int, int] = {}
        self._prev_ids: set[int] = set()
        self.unique_total = 0
        self.n_new = 0
        self.n_lost = 0

    def update(self, tracks, frame_id: int) -> None:
        live = {int(t.track_id) for t in tracks}
        self.n_new = len(live - self._prev_ids)
        self.n_lost = len(self._prev_ids - live)
        for tid in live - self._prev_ids:
            if tid not in self._first_seen:
                self._first_seen[tid] = frame_id
                self.unique_total += 1
        self._prev_ids = live

        # A retired id will never be looked up again; dropping it keeps the dict
        # bounded over a long run.
        if len(self._first_seen) > 100_000:
            self._first_seen = {t: self._first_seen[t] for t in live}

    def age_histogram(self, frame_id: int) -> dict[str, int]:
        buckets = {"1_5": 0, "6_15": 0, "16_30": 0, "31_60": 0, "60p": 0}
        for tid in self._prev_ids:
            age = frame_id - self._first_seen.get(tid, frame_id) + 1
            if age <= 5:
                buckets["1_5"] += 1
            elif age <= 15:
                buckets["6_15"] += 1
            elif age <= 30:
                buckets["16_30"] += 1
            elif age <= 60:
                buckets["31_60"] += 1
            else:
                buckets["60p"] += 1
        return buckets

    def mean_age(self, frame_id: int) -> float:
        if not self._prev_ids:
            return 0.0
        return float(np.mean([
            frame_id - self._first_seen.get(t, frame_id) + 1 for t in self._prev_ids
        ]))


# --------------------------------------------------------------------------- #
def track_arrays(tracks) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Confirmed tracks -> ``(xyxy, track_ids, class_ids)`` for the drawing code."""
    if not tracks:
        return (np.zeros((0, 4), np.float32), np.zeros((0,), np.int64),
                np.zeros((0,), np.int64))
    xyxy = np.asarray([t.box for t in tracks], dtype=np.float32).reshape(-1, 4)
    ids = np.asarray([t.track_id for t in tracks], dtype=np.int64)
    cls = np.asarray([t.cls for t in tracks], dtype=np.int64)
    return xyxy, ids, cls


def traces_of(tracks, max_len: int = 60, min_travel: float = 8.0) -> dict[int, list]:
    """Trails, from the history the tracker already maintains (capped at 64).

    Tracks that have not actually gone anywhere are dropped. A parked car still
    accumulates a history, because the Kalman centroid jitters by a pixel or two
    every frame, and drawing it produces a scribble on top of a stationary
    object -- which reads as tracking failure when it is the opposite. Requiring
    ``min_travel`` pixels of end-to-end displacement means a trail on screen
    means motion.
    """
    out: dict[int, list] = {}
    for t in tracks:
        h = t.history[-max_len:] if max_len else t.history
        if len(h) < 2:
            continue
        (x0, y0), (x1, y1) = h[0], h[-1]
        if (x1 - x0) ** 2 + (y1 - y0) ** 2 >= min_travel ** 2:
            out[int(t.track_id)] = h
    return out


def size_split(xyxy: np.ndarray, small_max: float, medium_max: float) -> dict[str, int]:
    """Count boxes by the COCO area convention frozen in ``configs/visdrone.yaml``.

    This is the split the headline detection count hides: most VisDrone objects
    are small, so a change that only moves the total says nothing about whether
    the small ones were found.
    """
    if len(xyxy) == 0:
        return {"small": 0, "medium": 0, "large": 0}
    a = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    return {
        "small": int((a <= small_max).sum()),
        "medium": int(((a > small_max) & (a <= medium_max)).sum()),
        "large": int((a > medium_max).sum()),
    }
