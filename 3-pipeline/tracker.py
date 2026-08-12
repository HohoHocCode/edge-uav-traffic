"""ByteTrack — dependency-free implementation (numpy only).

Written out rather than pulled from Ultralytics for two reasons that matter on
this project:

* It runs on the board without dragging torch onto an aarch64 device.
* Every threshold is an explicit constructor argument, so the frozen config in
  ``configs/pipeline.yaml`` fully determines behaviour. Ultralytics' defaults
  for ``botsort.yaml`` have changed between releases, which silently
  invalidates any benchmark that cites the file name instead of the values.

The association is the standard two-stage ByteTrack cascade: high-confidence
detections first against all tracks, then low-confidence leftovers against the
tracks that are still unmatched. The second stage is what recovers small,
partially-occluded objects — which on aerial footage is most of them.
"""

from __future__ import annotations

import numpy as np

from kalman import KalmanBoxTracker


# --------------------------------------------------------------------------- #
def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of xyxy boxes -> (len(a), len(b))."""
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    area_a = np.maximum(0.0, a[:, 2] - a[:, 0]) * np.maximum(0.0, a[:, 3] - a[:, 1])
    area_b = np.maximum(0.0, b[:, 2] - b[:, 0]) * np.maximum(0.0, b[:, 3] - b[:, 1])

    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:4], b[None, :, 2:4])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]
    return (inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)).astype(np.float32)


def greedy_match(
    cost: np.ndarray, thresh: float
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Greedy assignment on an IoU matrix.

    Greedy rather than Hungarian: on aerial scenes with many similar small
    boxes the two agree almost always, and greedy has no scipy dependency and
    a predictable cost on the Kryo cores. Returns (matches, unmatched_rows,
    unmatched_cols).
    """
    matches: list[tuple[int, int]] = []
    if cost.size == 0:
        return matches, list(range(cost.shape[0])), list(range(cost.shape[1]))

    rows_free = set(range(cost.shape[0]))
    cols_free = set(range(cost.shape[1]))

    order = np.dstack(np.unravel_index(np.argsort(cost, axis=None)[::-1], cost.shape))[0]
    for r, c in order:
        r, c = int(r), int(c)
        if cost[r, c] < thresh:
            break
        if r in rows_free and c in cols_free:
            matches.append((r, c))
            rows_free.discard(r)
            cols_free.discard(c)
    return matches, sorted(rows_free), sorted(cols_free)


# --------------------------------------------------------------------------- #
class Track:
    _next_id = 1

    __slots__ = (
        "kf", "track_id", "cls", "conf", "hits", "age",
        "time_since_update", "state", "history",
    )

    def __init__(self, box: np.ndarray, cls: int, conf: float) -> None:
        self.kf = KalmanBoxTracker(box)
        self.track_id = Track._next_id
        Track._next_id += 1
        self.cls = int(cls)
        self.conf = float(conf)
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.state = "tentative"           # tentative | tracked | lost
        self.history: list[tuple[float, float]] = []

    @staticmethod
    def reset_ids() -> None:
        """Reset the global ID counter — call between independent sequences."""
        Track._next_id = 1

    @property
    def box(self) -> np.ndarray:
        return self.kf.get_state()

    @property
    def centroid(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def predict(self) -> None:
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self, box: np.ndarray, cls: int, conf: float) -> None:
        self.kf.update(box)
        self.cls = int(cls)
        self.conf = float(conf)
        self.hits += 1
        self.time_since_update = 0
        self.history.append(self.centroid)
        if len(self.history) > 64:
            self.history.pop(0)


class PseudoTrack:
    """A detection wearing a track's interface, for tracker-disabled runs.

    With ``tracker.enabled: false`` the analytics layer still has to receive
    something track-shaped, otherwise every count silently reads zero and the
    run looks like an empty scene rather than a detection-only scene.

    ``track_id`` is **negative and per-frame**: it is the object's index in
    this frame's detection list, not an identity. Index 1 in consecutive
    frames is generally two unrelated objects. The negative sign is the
    contract that tells :class:`~analytics.TrafficAnalytics` to exclude these
    from line crossing, which is only meaningful for a persistent identity.
    Per-frame and per-ROI counts remain valid.
    """

    __slots__ = ("track_id", "cls", "conf", "_box", "history", "state",
                 "hits", "time_since_update")

    def __init__(self, box: np.ndarray, cls: int, conf: float, idx: int) -> None:
        self._box = np.asarray(box, dtype=np.float32)
        self.cls = int(cls)
        self.conf = float(conf)
        self.track_id = -(idx + 1)          # negative: not a real identity
        self.history: list[tuple[float, float]] = []
        self.state = "tracked"
        self.hits = 1
        self.time_since_update = 0

    @property
    def box(self) -> np.ndarray:
        return self._box

    @property
    def centroid(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self._box
        return (float(x1 + x2) / 2.0, float(y1 + y2) / 2.0)


def detections_as_tracks(xyxy: np.ndarray, conf: np.ndarray,
                         cls: np.ndarray) -> list[PseudoTrack]:
    return [PseudoTrack(xyxy[i], cls[i], conf[i], i) for i in range(len(xyxy))]


class ByteTrack:
    """Two-stage IoU tracker.

    Parameters mirror ``configs/pipeline.yaml`` one-for-one.
    """

    def __init__(
        self,
        track_high_thresh: float = 0.50,
        track_low_thresh: float = 0.10,
        new_track_thresh: float = 0.60,
        match_iou: float = 0.20,
        match_iou_low: float = 0.50,
        track_buffer: int = 30,
        min_box_area: float = 4.0,
        min_hits: int = 3,
    ) -> None:
        # NOTE ON SEMANTICS: these are *IoU* thresholds (higher = stricter).
        # Upstream ByteTrack names its equivalent `match_thresh: 0.8`, but that
        # value is a threshold on the cost 1-IoU, i.e. IoU >= 0.2. Copying 0.8
        # across as an IoU bar rejects almost every real association. The
        # defaults here are the upstream behaviour, written the way they read.
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.match_iou = match_iou
        self.match_iou_low = match_iou_low
        self.track_buffer = track_buffer
        self.min_box_area = min_box_area
        self.min_hits = min_hits

        self.tracks: list[Track] = []
        self.frame_id = 0

    # ------------------------------------------------------------------ #
    def update(
        self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray
    ) -> list[Track]:
        """Advance one frame. Returns the currently confirmed tracks."""
        self.frame_id += 1

        # Drop degenerate boxes before they can spawn tracks.
        if len(xyxy):
            area = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
            keep = area >= self.min_box_area
            xyxy, conf, cls = xyxy[keep], conf[keep], cls[keep]

        hi = conf >= self.track_high_thresh
        lo = (conf >= self.track_low_thresh) & ~hi

        for t in self.tracks:
            t.predict()

        # ---- stage 1: high-confidence detections vs all tracks ----------
        pred = np.array([t.box for t in self.tracks], dtype=np.float32) \
            if self.tracks else np.zeros((0, 4), np.float32)

        m1, un_tracks, un_dets_hi = greedy_match(
            iou_matrix(pred, xyxy[hi]), self.match_iou
        )
        hi_idx = np.flatnonzero(hi)
        for ti, di in m1:
            d = hi_idx[di]
            self.tracks[ti].update(xyxy[d], cls[d], conf[d])

        # ---- stage 2: low-confidence detections vs still-unmatched ------
        # A lower IoU bar here: these are the boxes the detector was unsure
        # about, and demanding a tight overlap defeats the purpose.
        if un_tracks and lo.any():
            rem = [self.tracks[i] for i in un_tracks]
            rem_boxes = np.array([t.box for t in rem], dtype=np.float32)
            m2, _, _ = greedy_match(
                iou_matrix(rem_boxes, xyxy[lo]), self.match_iou_low
            )
            lo_idx = np.flatnonzero(lo)
            matched_local = set()
            for ti, di in m2:
                d = lo_idx[di]
                rem[ti].update(xyxy[d], cls[d], conf[d])
                matched_local.add(ti)
            un_tracks = [un_tracks[i] for i in range(len(rem)) if i not in matched_local]

        # ---- age out ----------------------------------------------------
        for i in un_tracks:
            t = self.tracks[i]
            if t.state == "tracked":
                t.state = "lost"

        self.tracks = [
            t for t in self.tracks if t.time_since_update <= self.track_buffer
        ]

        # ---- spawn ------------------------------------------------------
        for di in un_dets_hi:
            d = hi_idx[di]
            if conf[d] >= self.new_track_thresh:
                self.tracks.append(Track(xyxy[d], cls[d], conf[d]))

        # ---- promote ----------------------------------------------------
        for t in self.tracks:
            if t.time_since_update == 0 and t.hits >= self.min_hits:
                t.state = "tracked"

        return self.confirmed()

    # ------------------------------------------------------------------ #
    def confirmed(self) -> list[Track]:
        """Tracks that are mature and were seen on the current frame."""
        return [
            t for t in self.tracks
            if t.state == "tracked" and t.time_since_update == 0
        ]

    def reset(self) -> None:
        self.tracks.clear()
        self.frame_id = 0
        Track.reset_ids()
