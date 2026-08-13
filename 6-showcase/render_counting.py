#!/usr/bin/env python3
"""VisDrone Task 5 -- crowd and vehicle counting.

Tracking as in Task 4, then the repository's own ``TrafficAnalytics`` on top:
per-class and per-group counts, per-ROI counts, directional line crossings, and
a congestion level with hysteresis. Those rules are reused rather than
re-derived so the video reports exactly what the pipeline reports -- see
``bridge.py`` for why ``sv.LineZone`` is not used for the counting.

Two corrections matter here and both come from the camera moving:

* **Crossings are counted twice.** ``ta`` sees every track and produces the
  per-class, per-ROI and congestion figures, which are screen quantities and
  correct as such. ``ta_move`` sees only tracks still moving once the camera is
  compensated out, and that is the line-crossing figure worth quoting -- a
  screen-fixed gate on a panning camera counts parked vehicles.
* **The density map slides with the ground** (``Heatmap.warp``), so a cluster
  stays a blob instead of painting a streak along the flight path. It decays
  with a ~0.8 s half-life, which makes it "where is the crowd now" rather than
  a motion history.

Read the congestion badge with the limitation from the README in mind: the
monitor inherits the detector's blindness. If the detector stops seeing
vehicles, the count falls and this badge reads NORMAL. Nothing here
distinguishes an empty road from a road it cannot see.
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "3-pipeline"))

import common                                  # noqa: E402
import draw                                    # noqa: E402
import video_io                                # noqa: E402
from analytics import build_from_config        # noqa: E402  (3-pipeline)
from bridge import as_arrays, build_tracker, track_arrays   # noqa: E402
from gmc import GlobalMotion, moving_mask, warp_tracks      # noqa: E402
from record import Recorder                    # noqa: E402
from viz import draw_regions                   # noqa: E402  (3-pipeline)


def main(argv=None) -> int:
    p = common.base_parser(__doc__.splitlines()[0], "task5_counting.mp4")
    p.add_argument("--heatmap", dest="heatmap", action="store_true", default=True)
    p.add_argument("--no-heatmap", dest="heatmap", action="store_false")
    p.add_argument("--heat-radius", type=int, default=60,
                   help="density kernel radius, in source pixels")
    # 0.97 is a half-life of about 23 frames (0.8 s). Slower decay turns the map
    # into a motion history -- long streaks behind every mover -- which answers
    # a different question than "where is the crowd".
    p.add_argument("--heat-decay", type=float, default=0.97,
                   help="per-frame multiplier; 1.0 = never forget")
    args = p.parse_args(argv)

    pcfg, vcfg, meta, engine, class_names, run_meta = common.setup(args, "task5_counting")
    tcfg = pcfg.get("tracker", {})
    acfg = pcfg.get("analytics", {})
    groups = vcfg.get("groups", {})

    tracker = build_tracker(tcfg)
    gmc = GlobalMotion(scale=args.gmc_scale) if args.gmc else None

    # Two analytics instances over the same frozen config.
    #
    #   ta      every track -> per-class / per-ROI counts, congestion. These are
    #           screen quantities and are correct as such: "how many vehicles
    #           are in view" does not care whether they are parked.
    #   ta_move only tracks moving after camera compensation -> line crossings.
    #
    # The split exists because a counting line means "traffic passed this gate",
    # and on a moving camera a screen-fixed line sweeps across parked vehicles
    # and counts every one of them. Measured on this clip the camera translates
    # 4.9 px/frame on average, and only about a quarter of tracks are actually
    # moving -- so the uncorrected figure is mostly drone, not traffic.
    ta = build_from_config(acfg, class_names, groups)
    ta_move = build_from_config(acfg, class_names, groups)
    person_ids = {cid for cid, n in class_names.items()
                  if n in set(groups.get("person", ["pedestrian", "people"]))}

    run_meta["analytics"] = {
        "rois": [{"name": r.name, "box": list(r.box)} for r in ta.rois],
        "lines": [{"name": ln.name, "seg": list(ln.seg)} for ln in ta.lines],
        "warn_count": ta.warn_count, "alert_count": ta.alert_count,
        "hysteresis": ta.hysteresis,
        "vehicle_classes": sorted(ta.vehicle_classes),
    }
    run_meta["groups"] = {k: list(v) for k, v in groups.items()}
    run_meta["gmc"] = {"enabled": bool(args.gmc), "scale": args.gmc_scale,
                       "min_speed_px": args.min_speed}

    heat = draw.Heatmap((meta.height, meta.width), radius=args.heat_radius,
                        decay=args.heat_decay) if args.heatmap else None

    rec = Recorder(args.out, run_meta)
    timings_all: list = []

    reader = video_io.ThreadedReader(meta)
    writer = None if args.benchmark else video_io.ThreadedWriter(
        args.out, meta.width, meta.height, meta.fps
    )
    tick = common.Ticker(meta.n_frames)

    with reader:
        with (writer if writer is not None else common.null_ctx()):
            for idxs, frames in reader.batches(args.batch):
                dets_batch, timings = engine.infer_batch(frames)
                timings_all.extend(timings)

                for i, frame, dets, t in zip(idxs, frames, dets_batch, timings):
                    A = gmc.update(frame) if gmc is not None else None
                    if A is not None:
                        warp_tracks(tracker.tracks, A)

                    tracks = tracker.update(*as_arrays(dets))
                    mask = moving_mask(tracks, args.min_speed)
                    movers = [tr for tr, m in zip(tracks, mask) if m]

                    ts = i * 1000.0 / meta.fps
                    report = ta.update(tracks, (meta.height, meta.width), i, ts)
                    rep_move = ta_move.update(movers, (meta.height, meta.width), i, ts)

                    xyxy, ids, cls = track_arrays(tracks)
                    cents = (
                        np.column_stack([(xyxy[:, 0] + xyxy[:, 2]) * 0.5,
                                         (xyxy[:, 1] + xyxy[:, 3]) * 0.5])
                        if len(xyxy) else np.zeros((0, 2), np.float32)
                    )
                    if heat is not None:
                        # Slide the accumulator with the ground first, then add
                        # this frame's objects at their current positions.
                        if A is not None:
                            heat.warp(A)
                        heat.update(cents)

                    if writer is not None:
                        if heat is not None:
                            frame = heat.render(frame)
                        draw_regions(frame, ta.rois, ta.lines)
                        draw.draw_boxes(frame, xyxy, cls)
                        # People are the crowd half of this task and the
                        # smallest boxes in the frame; a dot reads at zoom
                        # levels where a 6-pixel rectangle does not.
                        if len(cents) and person_ids:
                            mask = np.isin(cls, list(person_ids))
                            if mask.any():
                                draw.draw_dots(frame, cents[mask], (110, 220, 110), 3)

                        if args.stats:
                            xs = " ".join(
                                f"{ln} {d['forward']}>/{d['backward']}<"
                                for ln, d in rep_move.line_crossings.items()
                            ) or "-"
                            rois = " ".join(
                                f"{k}:{v}" for k, v in report.counts_by_roi.items()
                            ) or "-"
                            draw.draw_stats(frame, [
                                "TASK 5  CROWD / VEHICLE COUNTING",
                                f"people       {report.person_count:4d}",
                                f"vehicles     {report.vehicle_count:4d}",
                                f"tracks       {report.n_tracks:4d}"
                                f"   moving {len(movers)}",
                                f"crossings    {xs}",
                                f"roi          {rois}",
                                f"camera       {gmc.translation_px:5.1f} px/frame"
                                if gmc is not None else "camera       (gmc off)",
                                f"pre {t.pre_ms:5.1f} infer {t.infer_ms:5.1f} post {t.post_ms:5.1f} ms",
                            ])
                            draw.draw_badge(frame, report.congestion_level.upper(),
                                            report.congestion_level)
                        draw.draw_progress(frame, (i + 1) / meta.n_frames)
                        writer.write(frame)
                        common.maybe_save_frame(args, frame, i)

                    row = report.to_row()
                    row.update({
                        f"grp_{g}": int(v)
                        for g, v in report.counts_by_group.items()
                    })
                    # Crossings appear twice on purpose. `line_*` is what a
                    # screen-fixed gate sees and includes the drone's own
                    # sweep; `xmov_*` counts only camera-compensated movers and
                    # is the traffic figure. Keeping both lets the web app show
                    # the correction rather than assert it.
                    for ln, d in rep_move.line_crossings.items():
                        row[f"xmov_{ln}_fwd"] = d["forward"]
                        row[f"xmov_{ln}_bwd"] = d["backward"]
                    row["n_moving"] = len(movers)
                    row["n_static"] = len(tracks) - len(movers)
                    if gmc is not None:
                        row.update(gmc.stats())
                    row.update({
                        "pre_ms": round(t.pre_ms, 3),
                        "infer_ms": round(t.infer_ms, 3),
                        "post_ms": round(t.post_ms, 3),
                        "total_ms": round(t.total_ms, 3),
                    })
                    rec.add(row)
                tick.tick(len(frames))

    elapsed, _ = tick.finish()
    common.report_speed(timings_all, elapsed, rec.n)
    for name, d in ta_move.total_crossings.items():
        raw = ta.total_crossings[name]
        print(f"[info] line '{name}': {d['forward']} forward, {d['backward']} backward "
              f"(moving objects only; uncorrected screen count would be "
              f"{raw['forward']}/{raw['backward']})")
    if gmc is not None:
        cam = sum(r.get("cam_px", 0.0) for r in rec.frames) / max(rec.n, 1)
        mov = sum(r["n_moving"] for r in rec.frames) / max(rec.n, 1)
        tot = sum(r["n_tracks"] for r in rec.frames) / max(rec.n, 1)
        print(f"[info] camera    {cam:.2f} px/frame mean"
              f"{f', {gmc.n_failed} frames unestimated' if gmc.n_failed else ''}")
        print(f"[info] motion    {mov:.1f} of {tot:.1f} tracks/frame moving")
    levels = [r["congestion_level"] for r in rec.frames]
    for lvl in ("normal", "warn", "alert"):
        n = levels.count(lvl)
        if n:
            print(f"[info] congestion {lvl:6s} {n:5d} frames "
                  f"({100.0 * n / max(len(levels), 1):.1f}%)")
    print("[note] the congestion level inherits the detector's blindness: a count "
          "that falls because detection failed reads the same as light traffic.")

    rec.check_alignment(writer.frames_written if writer else rec.n)
    rec.close(class_names)
    if writer is not None:
        common.finish_video(args, writer.frames_written)
    return 0 if rec.n else 1


if __name__ == "__main__":
    raise SystemExit(main())
