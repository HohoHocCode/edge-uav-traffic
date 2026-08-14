#!/usr/bin/env python3
"""VisDrone Task 4 -- multi-object tracking with Ultralytics trackers.

Frames are decoded in order and passed directly to ``YOLO.track`` with
``persist=True``. Ultralytics owns association, motion prediction, lost-track
recovery, and (where supported by the selected tracker) camera-motion
compensation. This renderer only keeps display histories and health statistics.

Those display histories are camera-compensated too, using the warp the selected
tracker already computed rather than a second optical-flow estimate. The
association was compensated from the start; the *display* was not, and on this
footage that alone accounts for most of what reads as tracker failure -- trails
bending along the flight path, and a moving-object filter that passes parked
cars because they slide across the screen. See ``TrackStats.warp_history``.

No MOTA or IDF1 is reported because this sequence has no MOT ground truth.
Track age and id churn are measurable without ground truth and are recorded as
operational diagnostics, not claimed as accuracy metrics.
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import common                                      # noqa: E402
import draw                                        # noqa: E402
import video_io                                    # noqa: E402
from bridge import TrackStats, tracking_arrays     # noqa: E402
from record import Recorder                        # noqa: E402


TRACKERS = {
    "botsort": {
        "config": "botsort.yaml",
        "description": "Ultralytics default; Kalman + IoU/score + GMC",
    },
    "bytetrack": {
        "config": "bytetrack.yaml",
        "description": "lighter baseline; two-stage confidence association",
    },
    "tracktrack": {
        "config": "tracktrack.yaml",
        "description": "strongest installed option; multi-cue matching + GMC",
    },
}


def main(argv=None) -> int:
    p = common.base_parser(__doc__.splitlines()[0], "task4_tracking.mp4")
    p.add_argument(
        "--tracker",
        choices=tuple(TRACKERS),
        default="botsort",
        help="Ultralytics tracker configuration",
    )
    p.add_argument("--trace", type=int, default=60, help="trail length, in frames")
    p.set_defaults(batch=1, conf=0.10)
    args = p.parse_args(argv)

    if args.engine != "ultralytics":
        raise SystemExit(
            "[fatal] Task 4 uses Ultralytics trackers; choose --engine ultralytics"
        )
    # Stateful MOT is frame-sequential. A low detector threshold is required so
    # BoT-SORT/ByteTrack can use their second-stage low-confidence association.
    if args.batch != 1:
        print(f"[warn] --batch {args.batch} ignored: tracking is frame-sequential")
        args.batch = 1

    _pcfg, _vcfg, meta, engine, class_names, run_meta = common.setup(
        args, "task4_tracking"
    )
    tracker_info = TRACKERS[args.tracker]
    tracker_config = tracker_info["config"]
    run_meta["tracker"] = {"name": args.tracker, **tracker_info}
    # Not a second optical-flow pass: this reuses the warp the tracker computes
    # for its own association. --no-gmc leaves the association compensated and
    # the display uncompensated, which is the state this renderer shipped in and
    # the only reason to keep the flag.
    run_meta["gmc"] = {"source": "tracker", "display_compensated": bool(args.gmc)}

    stats = TrackStats()
    camera_px = 0.0
    gmc_frames = 0
    sv_annotators = None
    if not args.fast_draw:
        import supervision as sv

        palette = draw.sv_palette()
        sv_annotators = (
            sv.BoxAnnotator(
                color=palette, thickness=1, color_lookup=sv.ColorLookup.TRACK
            ),
            sv.LabelAnnotator(
                color=palette,
                color_lookup=sv.ColorLookup.TRACK,
                text_scale=0.35,
                text_thickness=1,
            ),
        )
    rec = Recorder(args.out, run_meta)
    timings_all: list = []

    reader = video_io.ThreadedReader(meta)
    writer = None if args.benchmark else video_io.ThreadedWriter(
        args.out, meta.width, meta.height, meta.fps
    )
    tick = common.Ticker(meta.n_frames)

    with reader:
        with (writer if writer is not None else common.null_ctx()):
            for i, frame in reader:
                dets, timing = engine.track_frame(frame, tracker_config)
                timings_all.append(timing)

                # The tracker has already compensated its own association for
                # camera motion; the display had not. Reuse that same warp for
                # the trails before appending this frame's centroids, so both
                # live in one coordinate frame. See TrackStats.warp_history.
                warp = engine.last_warp if args.gmc else None
                if warp is not None:
                    camera_px = stats.warp_history(warp)
                    gmc_frames += 1

                xyxy, ids, cls = tracking_arrays(dets)
                stats.update_arrays(ids, xyxy, i)
                ages = stats.age_histogram(i)
                per_class = {
                    f"trk_{name}": int((cls == cid).sum())
                    for cid, name in sorted(class_names.items())
                }
                # Recorded, not just drawn. A trail means "this object moved
                # across the ground", so the count is the readout that says
                # whether the camera compensation is doing its job: uncompensated
                # it approaches n_active, because every parked car qualifies.
                trails = stats.traces(args.trace)

                if writer is not None:
                    draw.draw_traces(frame, trails)
                    labels = [
                        f"#{int(tid)} {class_names.get(int(class_id), class_id)}"
                        for tid, class_id in zip(ids, cls)
                    ]
                    if args.fast_draw:
                        draw.draw_boxes(frame, xyxy, ids)
                        draw.draw_labels(frame, xyxy, labels, ids)
                    else:
                        box_annotator, label_annotator = sv_annotators
                        frame = box_annotator.annotate(frame, dets)
                        frame = label_annotator.annotate(frame, dets, labels=labels)
                    if args.stats:
                        draw.draw_stats(frame, [
                            "TASK 4  MULTI-OBJECT TRACKING",
                            f"tracker      {args.tracker}",
                            f"active       {len(ids):4d}",
                            f"unique ids   {stats.unique_total:4d}"
                            f"   (+{stats.n_new} -{stats.n_lost})",
                            f"mean age     {stats.mean_age(i):5.1f} frames",
                            f"camera       {camera_px:5.1f} px/frame"
                            if warp is not None else
                            f"camera       {'(gmc off)' if not args.gmc else '(tracker has no gmc)'}",
                            "age 1-5/6-15/16-30/31-60/60+  "
                            f"{ages['1_5']}/{ages['6_15']}/{ages['16_30']}"
                            f"/{ages['31_60']}/{ages['60p']}",
                            f"pre {timing.pre_ms:5.1f} infer {timing.infer_ms:5.1f}"
                            f" post {timing.post_ms:5.1f} track {timing.track_ms:5.1f} ms",
                        ])
                    draw.draw_progress(frame, (i + 1) / meta.n_frames)
                    writer.write(frame)
                    common.maybe_save_frame(args, frame, i)

                rec.add({
                    "frame_id": i,
                    "timestamp_ms": round(i * 1000.0 / meta.fps, 2),
                    "tracker": args.tracker,
                    "n_active": len(ids),
                    "n_traces": len(trails),
                    "n_new": stats.n_new,
                    "n_lost": stats.n_lost,
                    "unique_total": stats.unique_total,
                    "mean_age": round(stats.mean_age(i), 2),
                    "camera_px": round(camera_px, 3) if warp is not None else "",
                    **{f"age_{key}": value for key, value in ages.items()},
                    **per_class,
                    "pre_ms": round(timing.pre_ms, 3),
                    "infer_ms": round(timing.infer_ms, 3),
                    "post_ms": round(timing.post_ms, 3),
                    "track_ms": round(timing.track_ms, 3),
                    "total_ms": round(timing.total_ms, 3),
                })
                tick.tick()

    elapsed, _ = tick.finish()
    common.report_speed(timings_all, elapsed, rec.n)
    churn = sum(row["n_new"] for row in rec.frames) / max(rec.n, 1)
    mean_active = sum(row["n_active"] for row in rec.frames) / max(rec.n, 1)
    mean_traces = sum(row["n_traces"] for row in rec.frames) / max(rec.n, 1)
    print(
        f"[info] tracks    {stats.unique_total} unique ids, "
        f"{mean_active:.1f} active/frame, {churn:.2f} new ids/frame"
    )
    print(f"[info] trails    {mean_traces:.1f} of {mean_active:.1f} active tracks "
          f"drawn a trail ({100.0 * mean_traces / max(mean_active, 1e-9):.0f}% "
          f"judged to be moving)")
    if not args.gmc:
        print("[warn] display not camera-compensated (--no-gmc): trails bend "
              "with the flight path and the parked-car filter passes everything")
    elif gmc_frames:
        cam = [row["camera_px"] for row in rec.frames if row["camera_px"] != ""]
        print(f"[info] camera    {gmc_frames}/{rec.n} frames compensated, "
              f"{sum(cam) / max(len(cam), 1):.2f} px/frame mean translation")
    else:
        print(f"[warn] tracker '{args.tracker}' computes no camera warp; trails "
              f"are drawn in raw image coordinates")
    print("[note] no MOT ground truth for this sequence: no MOTA/IDF1 is claimed.")

    rec.check_alignment(writer.frames_written if writer else rec.n)
    rec.close(class_names)
    if writer is not None:
        common.finish_video(args, writer.frames_written)
    return 0 if rec.n else 1


if __name__ == "__main__":
    raise SystemExit(main())
