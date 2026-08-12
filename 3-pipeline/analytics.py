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
- A line crossing is attributed to a track exactly once per direction change,
  keyed on track id, so re-crossing the same line back and forth counts twice
  (once each way) and jitter on the line counts zero.
- ROI membership uses the box centroid, not overlap. Overlap makes a vehicle
  straddling the ROI edge belong to two regions at once, and then the regional
  counts no longer sum to the frame count.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# --------------------------------------------------------------------------- #
def _side_of_line(px: float, py: float, seg: tuple[float, float, float, float]) -> float:
    """Signed side of point (px, py) relative to the directed segment."""
    x1, y1, x2, y2 = seg
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


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

        # track_id -> {line_name: last signed side}
        self._last_side: dict[int, dict[str, float]] = {}
        # cumulative crossings
        self._crossings: dict[str, dict[str, int]] = {
            ln.name: {"forward": 0, "backward": 0} for ln in self.lines
        }
        self._level = "normal"
        self._level_streak = 0

    # ------------------------------------------------------------------ #
    def _class_name(self, cls_id: int) -> str:
        return self.class_names.get(int(cls_id), f"class_{int(cls_id)}")

    # ------------------------------------------------------------------ #
    def update(self, tracks, frame_shape: tuple[int, int], frame_id: int,
               timestamp_ms: float) -> FrameReport:
        h, w = frame_shape
        rep = FrameReport(frame_id=frame_id, timestamp_ms=timestamp_ms,
                          n_tracks=len(tracks))

        counts_cls: dict[str, int] = {}
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

            # Line crossing uses the previous *recorded* side for this track,
            # so a track that appears already past the line never fires.
            sides = self._last_side.setdefault(t.track_id, {})
            for ln in self.lines:
                s = _side_of_line(nx, ny, ln.seg)
                prev = sides.get(ln.name)
                if prev is not None and prev != 0.0 and s != 0.0:
                    if prev < 0 < s:
                        self._crossings[ln.name]["forward"] += 1
                    elif s < 0 < prev:
                        self._crossings[ln.name]["backward"] += 1
                sides[ln.name] = s

        # Forget tracks that are gone, so the dict cannot grow without bound
        # over a long deployment.
        for tid in list(self._last_side):
            if tid not in live_ids:
                self._last_side.pop(tid, None)

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
        frames agree on the new level.
        """
        if vehicle_count >= self.alert_count:
            target = "alert"
        elif vehicle_count >= self.warn_count:
            target = "warn"
        else:
            target = "normal"

        if target == self._level:
            self._level_streak = 0
            return self._level

        self._level_streak += 1
        if self._level_streak >= self.hysteresis:
            self._level = target
            self._level_streak = 0
        return self._level

    # ------------------------------------------------------------------ #
    @property
    def total_crossings(self) -> dict[str, dict[str, int]]:
        return {k: dict(v) for k, v in self._crossings.items()}

    def reset(self) -> None:
        self._last_side.clear()
        for d in self._crossings.values():
            d["forward"] = d["backward"] = 0
        self._level = "normal"
        self._level_streak = 0


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
    )
