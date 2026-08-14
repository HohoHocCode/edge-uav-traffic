"""Small adapters between detection, tracking, analytics, and drawing.

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

Task 4 now uses Ultralytics' maintained trackers directly. The repository
tracker adapters remain here for Task 5, whose analytics consumes its native
``Track`` objects. The Task 4 helpers below only read tracker ids returned by
Ultralytics; they do not perform association or motion prediction.
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
        self._history: dict[int, list[tuple[float, float]]] = {}
        self.unique_total = 0
        self.n_new = 0
        self.n_lost = 0

    def update(self, tracks, frame_id: int) -> None:
        ids = np.asarray([t.track_id for t in tracks], dtype=np.int64)
        boxes = np.asarray([t.box for t in tracks], dtype=np.float32).reshape(-1, 4)
        self.update_arrays(ids, boxes, frame_id)

    def update_arrays(
        self, ids: np.ndarray, xyxy: np.ndarray, frame_id: int
    ) -> None:
        """Update display statistics from tracker outputs, without tracking."""
        live = {int(tid) for tid in ids}
        self.n_new = len(live - self._prev_ids)
        self.n_lost = len(self._prev_ids - live)
        for tid in live - self._prev_ids:
            if tid not in self._first_seen:
                self._first_seen[tid] = frame_id
                self.unique_total += 1

        for tid, box in zip(ids, xyxy):
            key = int(tid)
            x1, y1, x2, y2 = map(float, box)
            history = self._history.setdefault(key, [])
            history.append(((x1 + x2) * 0.5, (y1 + y2) * 0.5))
            if len(history) > 120:
                del history[:-120]
        self._prev_ids = live

        # A retired id will never be looked up again; dropping it keeps the dict
        # bounded over a long run.
        if len(self._first_seen) > 100_000:
            self._first_seen = {t: self._first_seen[t] for t in live}
            self._history = {t: self._history[t] for t in live if t in self._history}

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

    def warp_history(self, A: np.ndarray) -> float:
        """Carry every stored trail point into the current frame's coordinates.

        Call once per frame, *before* ``update_arrays`` appends this frame's
        centroids, so the whole trail and the new point share one coordinate
        frame.

        Two separate things are wrong without this, and both read as tracker
        failure when the tracker is innocent:

        * A car driving straight renders as a curve bent by the drone's flight
          path, because the older points of its trail are pinned to screen
          positions that no longer line up with this frame.
        * ``traces`` gates on end-to-end displacement to avoid scribbling on
          parked cars -- but measured on screen, a *stationary* object drifts a
          median 208 px over a 60-frame window on this footage (measured on
          video/uav0000076_00241_s_30fps, mean 271 px, p95 826 px). 99.7% of
          windows clear the 8 px gate, so the filter passes everything and every
          parked car gets a trail. Compensated, a stationary object's trail
          collapses to a few px of Kalman jitter and the gate works as intended.

        ``A`` is the 2x3 affine that maps previous-frame coordinates to this
        frame's, as returned by Ultralytics' ``GMC.apply``. Returns the camera
        translation in px, for the on-screen readout.
        """
        A = np.asarray(A, dtype=np.float32)
        M, t = A[:, :2], A[:, 2]

        # One matmul for every trail in the frame, not one per track. At 22
        # tracks x 120 points a per-track loop with a Python tuple rebuild
        # measured 1.9 ms/frame; batched, with tolist() doing the conversion in
        # C, it is a fraction of that -- and this runs on every frame of the
        # render, where draw and encode are already the budget.
        keys = [tid for tid, pts in self._history.items() if pts]
        if not keys:
            return float(np.hypot(t[0], t[1]))

        lengths = [len(self._history[tid]) for tid in keys]
        flat = np.concatenate([
            np.asarray(self._history[tid], dtype=np.float32) for tid in keys
        ])
        flat = flat @ M.T + t
        for tid, chunk in zip(keys, np.split(flat, np.cumsum(lengths[:-1]))):
            self._history[tid] = chunk.tolist()
        return float(np.hypot(t[0], t[1]))

    def traces(self, max_len: int = 60, min_travel: float = 8.0) -> dict[int, list]:
        """Return visible trails for currently active ids."""
        out: dict[int, list] = {}
        for tid in self._prev_ids:
            points = self._history.get(tid, [])[-max_len:]
            if len(points) < 2:
                continue
            (x0, y0), (x1, y1) = points[0], points[-1]
            if (x1 - x0) ** 2 + (y1 - y0) ** 2 >= min_travel ** 2:
                out[tid] = points
        return out


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


def tracking_arrays(dets) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ultralytics-tracked ``sv.Detections`` -> boxes, ids, classes.

    Detections that have not yet been accepted by the tracker have no id and
    are intentionally omitted from a tracking render.
    """
    ids = getattr(dets, "tracker_id", None)
    if len(dets) == 0 or ids is None:
        return (np.zeros((0, 4), np.float32), np.zeros((0,), np.int64),
                np.zeros((0,), np.int64))
    ids = np.asarray(ids, dtype=np.int64).reshape(-1)
    keep = ids >= 0
    xyxy = np.asarray(dets.xyxy, dtype=np.float32).reshape(-1, 4)[keep]
    cls = (
        np.asarray(dets.class_id, dtype=np.int64).reshape(-1)[keep]
        if dets.class_id is not None else np.zeros(int(keep.sum()), np.int64)
    )
    return xyxy, ids[keep], cls


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
