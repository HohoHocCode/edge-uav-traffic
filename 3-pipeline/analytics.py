"""Traffic-order analytics on top of tracks.

This is the layer that turns "boxes on a frame" into something a command post
can act on:

* per-class and per-group counts (VisDrone Task 5, crowd/vehicle counting)
* per-ROI counts, so "the intersection" is a separate number from "the frame"
* directional line-crossing counts, using the track history rather than the
  current box — a box that teleports across the line because of a detector
  flicker must not be counted
* a congestion level with hysteresis, so the alert does not chatter

Counting rules, stated once so the report can cite them:

- Only *confirmed* tracks are counted. A detection that has been seen fewer
  than ``min_hits`` times is not an object yet.
- A line crossing requires the track's motion segment to intersect the drawn
  *segment*, not merely to change sides of the infinite line it lies on.
  Traffic passing beyond either end of the painted line is not counted.
- A crossing is attributed to a track once per direction change, keyed on
  track id, so re-crossing back and forth counts twice (once each way) and
  jitter on the line counts zero.
- A track whose centroid jumps more than ``max_step_norm`` of the frame in one
  frame is an association error rather than a vehicle; its implied motion
  segment would sweep every line in its path, so no crossing is counted for
  that step.
- ROI membership uses the box centroid, not overlap. Overlap makes a vehicle
  straddling the ROI edge belong to two regions at once, and then the regional
  counts no longer sum to the frame count.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# --------------------------------------------------------------------------- #
def _side_of_line(px: float, py: float, seg: tuple[float, float, float, float]) -> float:
    """Signed side of point (px, py) relative to the directed segment's line."""
    x1, y1, x2, y2 = seg
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


def _segments_cross(
    p1: tuple[float, float], p2: tuple[float, float],
    seg: tuple[float, float, float, float],
) -> bool:
    """True if the motion segment p1->p2 properly crosses the counting segment.

    A sign change of :func:`_side_of_line` alone is **not** a crossing: it only
    says the track moved from one side of the *infinite line* to the other. A
    vehicle passing far beyond either end of the painted line produces exactly
    that sign change, and counting it inflates the tally with traffic that
    never went through the counted gate.

    Solving for both parameters and requiring each to lie in [0, 1] restricts
    the test to the drawn segment.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3, x4, y4 = seg

    rx, ry = x2 - x1, y2 - y1
    sx, sy = x4 - x3, y4 - y3
    denom = rx * sy - ry * sx
    if abs(denom) < 1e-12:
        return False                      # parallel or degenerate

    qpx, qpy = x3 - x1, y3 - y1
    t = (qpx * sy - qpy * sx) / denom     # along the motion segment
    u = (qpx * ry - qpy * rx) / denom     # along the counting segment
    return 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0


@dataclass
class RoiSpec:
    name: str
    box: tuple[float, float, float, float]   # normalised x1,y1,x2,y2


@dataclass
class LineSpec:
    name: str
    seg: tuple[float, float, float, float]   # normalised x1,y1,x2,y2


@dataclass
class FrameReport:
    frame_id: int
    timestamp_ms: float
    n_tracks: int
    counts_by_class: dict[str, int] = field(default_factory=dict)
    counts_by_group: dict[str, int] = field(default_factory=dict)
    counts_by_roi: dict[str, int] = field(default_factory=dict)
    line_crossings: dict[str, dict[str, int]] = field(default_factory=dict)
    congestion_level: str = "normal"          # normal | warn | alert
    vehicle_count: int = 0
    person_count: int = 0

    def to_row(self) -> dict:
        """Flatten for CSV / SQLite."""
        row = {
            "frame_id": self.frame_id,
            "timestamp_ms": round(self.timestamp_ms, 2),
            "n_tracks": self.n_tracks,
            "vehicle_count": self.vehicle_count,
            "person_count": self.person_count,
            "congestion_level": self.congestion_level,
        }
        for k, v in self.counts_by_class.items():
            row[f"cls_{k}"] = v
        for k, v in self.counts_by_roi.items():
            row[f"roi_{k}"] = v
        for lname, d in self.line_crossings.items():
            row[f"line_{lname}_fwd"] = d["forward"]
            row[f"line_{lname}_bwd"] = d["backward"]
        return row


# --------------------------------------------------------------------------- #
class TrafficAnalytics:
    def __init__(
        self,
        class_names: dict[int, str],
        groups: dict[str, list[str]] | None = None,
        rois: list[RoiSpec] | None = None,
        lines: list[LineSpec] | None = None,
        vehicle_classes: list[str] | None = None,
        warn_count: int = 25,
        alert_count: int = 45,
        hysteresis: int = 5,
        max_step_norm: float = 0.25,
    ) -> None:
        self.class_names = dict(class_names)
        self.groups = groups or {}
        self.rois = rois or []
        self.lines = lines or []
        self.vehicle_classes = set(
            vehicle_classes or ["car", "van", "truck", "bus", "motor"]
        )
        self.person_classes = set(self.groups.get("person", ["pedestrian", "people"]))
        self.warn_count = warn_count
        self.alert_count = alert_count
        self.hysteresis = hysteresis
        self.max_step_norm = max_step_norm

        # track_id -> {line_name: last signed side}
        self._last_side: dict[int, dict[str, float]] = {}
        # track_id -> last normalised centroid, for the segment-crossing test
        self._last_pos: dict[int, tuple[float, float]] = {}
        # cumulative crossings
        self._crossings: dict[str, dict[str, int]] = {
            ln.name: {"forward": 0, "backward": 0} for ln in self.lines
        }
        self._level = "normal"
        self._level_streak = 0
        self._pending_level: str | None = None

    # ------------------------------------------------------------------ #
    def _class_name(self, cls_id: int) -> str:
        return self.class_names.get(int(cls_id), f"class_{int(cls_id)}")

    # ------------------------------------------------------------------ #
    def update(self, tracks, frame_shape: tuple[int, int], frame_id: int,
               timestamp_ms: float) -> FrameReport:
        h, w = frame_shape
        rep = FrameReport(frame_id=frame_id, timestamp_ms=timestamp_ms,
                          n_tracks=len(tracks))

        # Seed every known class at zero. Without this the key set of
        # counts_by_class varies frame to frame, and any consumer that locks
        # its schema on the first frame -- the telemetry CSV writer does --
        # permanently loses the columns for classes absent from frame 0.
        counts_cls: dict[str, int] = {n: 0 for n in self.class_names.values()}
        counts_roi: dict[str, int] = {r.name: 0 for r in self.rois}
        live_ids = set()

        for t in tracks:
            live_ids.add(t.track_id)
            name = self._class_name(t.cls)
            counts_cls[name] = counts_cls.get(name, 0) + 1

            cx, cy = t.centroid
            nx, ny = cx / max(w, 1), cy / max(h, 1)

            for r in self.rois:
                x1, y1, x2, y2 = r.box
                if x1 <= nx <= x2 and y1 <= ny <= y2:
                    counts_roi[r.name] += 1

            # Line crossing needs a persistent identity. A negative track_id
            # marks a detection wrapped as a track (tracker disabled): those
            # ids are per-frame positions in the detection list, so id -1 in
            # consecutive frames is two unrelated objects, and treating them
            # as one manufactures a crossing on almost every frame.
            if t.track_id < 0:
                continue

            # Line crossing uses the track's previous *recorded* position, so a
            # track that appears already past the line never fires.
            prev_pos = self._last_pos.get(t.track_id)
            sides = self._last_side.setdefault(t.track_id, {})

            # Guard against detector flicker. An identity that jumps most of
            # the way across the frame in one frame is an association error,
            # not a vehicle, and the segment it implies would sweep across
            # every counting line in its path.
            step_ok = prev_pos is not None and (
                (nx - prev_pos[0]) ** 2 + (ny - prev_pos[1]) ** 2
            ) <= self.max_step_norm ** 2

            for ln in self.lines:
                s = _side_of_line(nx, ny, ln.seg)
                prev = sides.get(ln.name)
                if (prev_pos is not None and step_ok and prev is not None
                        and prev != 0.0 and s != 0.0
                        and _segments_cross(prev_pos, (nx, ny), ln.seg)):
                    if prev < 0 < s:
                        self._crossings[ln.name]["forward"] += 1
                    elif s < 0 < prev:
                        self._crossings[ln.name]["backward"] += 1
                sides[ln.name] = s

            self._last_pos[t.track_id] = (nx, ny)

        # Forget tracks that are gone, so the dict cannot grow without bound
        # over a long deployment.
        for tid in list(self._last_side):
            if tid not in live_ids:
                self._last_side.pop(tid, None)
                self._last_pos.pop(tid, None)

        rep.counts_by_class = counts_cls
        rep.counts_by_roi = counts_roi
        rep.line_crossings = {k: dict(v) for k, v in self._crossings.items()}

        for gname, members in self.groups.items():
            rep.counts_by_group[gname] = sum(counts_cls.get(m, 0) for m in members)

        rep.vehicle_count = sum(
            c for n, c in counts_cls.items() if n in self.vehicle_classes
        )
        rep.person_count = sum(
            c for n, c in counts_cls.items() if n in self.person_classes
        )
        rep.congestion_level = self._update_level(rep.vehicle_count)
        return rep

    # ------------------------------------------------------------------ #
    def _update_level(self, vehicle_count: int) -> str:
        """Congestion level with hysteresis.

        A raw threshold on a per-frame count flickers between states whenever
        the detector drops one vehicle, which produces an alert stream nobody
        will trust. The level only moves after ``hysteresis`` consecutive
        frames agree on *the same* new level.

        That last word is the whole point. Counting any frame that merely
        disagrees with the current level lets an alternating warn/alert/warn
        sequence accumulate a streak and flip the state, which is precisely
        the flapping the hysteresis exists to prevent. The candidate level is
        therefore tracked explicitly and the streak restarts whenever it
        changes.
        """
        if vehicle_count >= self.alert_count:
            target = "alert"
        elif vehicle_count >= self.warn_count:
            target = "warn"
        else:
            target = "normal"

        if target == self._level:
            self._pending_level = None
            self._level_streak = 0
            return self._level

        if target != self._pending_level:
            self._pending_level = target
            self._level_streak = 1
        else:
            self._level_streak += 1

        if self._level_streak >= self.hysteresis:
            self._level = target
            self._pending_level = None
            self._level_streak = 0
        return self._level

    # ------------------------------------------------------------------ #
    @property
    def total_crossings(self) -> dict[str, dict[str, int]]:
        return {k: dict(v) for k, v in self._crossings.items()}

    def reset(self) -> None:
        self._last_side.clear()
        self._last_pos.clear()
        for d in self._crossings.values():
            d["forward"] = d["backward"] = 0
        self._level = "normal"
        self._level_streak = 0
        self._pending_level = None


# --------------------------------------------------------------------------- #
def build_from_config(cfg: dict, class_names: dict[int, str],
                      groups: dict[str, list[str]] | None = None) -> TrafficAnalytics:
    """Construct from the ``analytics:`` block of pipeline.yaml."""
    rois = [RoiSpec(r["name"], tuple(r["box"])) for r in cfg.get("rois", [])]
    lines = [LineSpec(l["name"], tuple(l["seg"])) for l in cfg.get("lines", [])]
    cong = cfg.get("congestion", {})
    return TrafficAnalytics(
        class_names=class_names,
        groups=groups or {},
        rois=rois,
        lines=lines,
        vehicle_classes=cong.get("vehicle_classes"),
        warn_count=cong.get("warn_count", 25),
        alert_count=cong.get("alert_count", 45),
        hysteresis=cong.get("hysteresis", 5),
        max_step_norm=cfg.get("max_step_norm", 0.25),
    )
