#!/usr/bin/env python3
"""VisDrone Task 4 -- multi-object tracking.

Detection still runs in batches, but the tracker is updated one frame at a time
in decode order. That ordering is a correctness requirement, not a style choice:
ByteTrack carries state, and feeding it frames out of order corrupts identity
silently instead of raising. ``ThreadedReader`` preserves the order, and the
row/frame count check in ``record.py`` is the test for it.

Boxes are coloured by *track id* rather than class, because identity is what
this task is about -- a car that keeps one colour for 300 frames is the result
being shown.

No MOTA or IDF1 is reported. This sequence ships no ground truth, so those
cannot be computed, and inventing them would be worse than omitting them. Track
age and id churn *can* be measured without ground truth, and they are what the
readout shows instead.
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "3-pipeline"))

import common                                              # noqa: E402
import draw                                                # noqa: E402
import video_io                                            # noqa: E402
from bridge import (TrackStats, as_arrays, build_tracker,   # noqa: E402
                    track_arrays, traces_of)
from gmc import GlobalMotion, moving_mask, warp_tracks      # noqa: E402
from record import Recorder                                # noqa: E402


def main(argv=None) -> int:
    p = common.base_parser(__doc__.splitlines()[0], "task4_tracking.mp4")
    p.add_argument("--trace", type=int, default=60, help="trail length, in frames")
    args = p.parse_args(argv)

    pcfg, vcfg, meta, engine, class_names, run_meta = common.setup(args, "task4_tracking")
    tcfg = pcfg.get("tracker", {})
    tracker = build_tracker(tcfg)
    stats = TrackStats()
    gmc = GlobalMotion(scale=args.gmc_scale) if args.gmc else None
    run_meta["gmc"] = {"enabled": bool(args.gmc), "scale": args.gmc_scale,
                       "min_speed_px": args.min_speed}
    run_meta["tracker"] = {
        k: tcfg.get(k) for k in
        ("type", "track_high_thresh", "track_low_thresh", "new_track_thresh",
         "match_iou", "match_iou_low", "track_buffer", "min_hits")
    }

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
                    # Camera motion first, and before anything draws on the
                    # frame: the tracks must be moved into this frame's
                    # coordinates before the tracker predicts from them.
                    A = gmc.update(frame) if gmc is not None else None
                    if A is not None:
                        warp_tracks(tracker.tracks, A)

                    tracks = tracker.update(*as_arrays(dets))
                    stats.update(tracks, i)
                    mask = moving_mask(tracks, args.min_speed)
                    n_moving = int(mask.sum())

                    xyxy, ids, cls = track_arrays(tracks)
                    per_class = {
                        f"trk_{name}": int((cls == cid).sum())
                        for cid, name in sorted(class_names.items())
                    }
                    ages = stats.age_histogram(i)

                    if writer is not None:
                        # Trails only for tracks that are genuinely moving.
                        # Drawing one on a parked car -- which is what happens
                        # without camera compensation -- says the tracker is
                        # broken when it is the drone that moved.
                        movers = [tr for tr, m in zip(tracks, mask) if m]
                        draw.draw_traces(frame, traces_of(movers, args.trace))
                        draw.draw_boxes(frame, xyxy, ids)
                        draw.draw_labels(
                            frame, xyxy,
                            [f"#{int(tid)} {class_names.get(int(c), c)}"
                             for tid, c in zip(ids, cls)],
                            ids,
                        )
                        if args.stats:
                            draw.draw_stats(frame, [
                                "TASK 4  MULTI-OBJECT TRACKING",
                                f"active       {len(ids):4d}"
                                f"   moving {n_moving}  static {len(ids) - n_moving}",
                                f"unique ids   {stats.unique_total:4d}"
                                f"   (+{stats.n_new} -{stats.n_lost})",
                                f"mean age     {stats.mean_age(i):5.1f} frames",
                                "age 1-5/6-15/16-30/31-60/60+  "
                                f"{ages['1_5']}/{ages['6_15']}/{ages['16_30']}"
                                f"/{ages['31_60']}/{ages['60p']}",
                                f"camera       {gmc.translation_px:5.1f} px/frame"
                                if gmc is not None else "camera       (gmc off)",
                                f"pre {t.pre_ms:5.1f} infer {t.infer_ms:5.1f}"
                                f" post {t.post_ms:5.1f} ms",
                            ])
                        draw.draw_progress(frame, (i + 1) / meta.n_frames)
                        writer.write(frame)
                        common.maybe_save_frame(args, frame, i)

                    rec.add({
                        "frame_id": i,
                        "timestamp_ms": round(i * 1000.0 / meta.fps, 2),
                        "n_active": len(ids),
                        "n_moving": n_moving,
                        "n_static": len(ids) - n_moving,
                        "n_new": stats.n_new,
                        "n_lost": stats.n_lost,
                        "unique_total": stats.unique_total,
                        "mean_age": round(stats.mean_age(i), 2),
                        **{f"age_{k}": v for k, v in ages.items()},
                        **per_class,
                        **(gmc.stats() if gmc is not None else {}),
                        "pre_ms": round(t.pre_ms, 3),
                        "infer_ms": round(t.infer_ms, 3),
                        "post_ms": round(t.post_ms, 3),
                        "total_ms": round(t.total_ms, 3),
                    })
                tick.tick(len(frames))

    elapsed, _ = tick.finish()
    common.report_speed(timings_all, elapsed, rec.n)
    churn = sum(r["n_new"] for r in rec.frames) / max(rec.n, 1)
    mean_active = sum(r["n_active"] for r in rec.frames) / max(rec.n, 1)
    print(f"[info] tracks    {stats.unique_total} unique ids, "
          f"{mean_active:.1f} active/frame, {churn:.2f} new ids/frame")
    if gmc is not None:
        cam = sum(r.get("cam_px", 0.0) for r in rec.frames) / max(rec.n, 1)
        mov = sum(r["n_moving"] for r in rec.frames) / max(rec.n, 1)
        print(f"[info] camera    {cam:.2f} px/frame mean"
              f"{f', {gmc.n_failed} frames unestimated' if gmc.n_failed else ''}")
        print(f"[info] motion    {mov:.1f} of {mean_active:.1f} tracks/frame "
              f"actually moving (camera-compensated)")
    print("[note] no MOT ground truth for this sequence: no MOTA/IDF1 is claimed.")

    rec.check_alignment(writer.frames_written if writer else rec.n)
    rec.close(class_names)
    if writer is not None:
        common.finish_video(args, writer.frames_written)
    return 0 if rec.n else 1


if __name__ == "__main__":
    raise SystemExit(main())
