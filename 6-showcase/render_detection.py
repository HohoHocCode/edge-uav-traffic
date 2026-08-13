#!/usr/bin/env python3
"""VisDrone Task 2 -- object detection in video.

Per-frame detection with no tracker. That absence is the point: Task 2 asks what
the detector sees in each frame independently, and adding temporal smoothing
would answer Task 4's question instead.

It is also the task that benefits most from batching, because nothing here is
sequential -- frames can be fed to the GPU in groups of eight with no state
carried between them.
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import common                      # noqa: E402
import draw                        # noqa: E402
import video_io                    # noqa: E402
from bridge import size_split      # noqa: E402
from record import Recorder        # noqa: E402


def main(argv=None) -> int:
    p = common.base_parser(__doc__.splitlines()[0], "task2_detection.mp4")
    args = p.parse_args(argv)

    pcfg, vcfg, meta, engine, class_names, run_meta = common.setup(args, "task2_detection")
    small_max = float(vcfg.get("area_small_max", 1024))
    medium_max = float(vcfg.get("area_medium_max", 9216))

    sv_annotators = None
    if not args.fast_draw:
        import supervision as sv
        pal = draw.sv_palette()
        sv_annotators = (
            sv.BoxAnnotator(color=pal, thickness=1),
            sv.LabelAnnotator(color=pal, text_scale=0.35, text_thickness=1),
        )

    rec = Recorder(args.out, run_meta)
    timings_all: list = []
    total_det = 0

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
                    n = len(dets)
                    total_det += n
                    cls = (
                        np.asarray(dets.class_id).astype(int)
                        if dets.class_id is not None else np.zeros(n, int)
                    )
                    conf = (
                        np.asarray(dets.confidence, dtype=np.float32)
                        if dets.confidence is not None else np.zeros(n, np.float32)
                    )
                    sizes = size_split(np.asarray(dets.xyxy, np.float32),
                                       small_max, medium_max)
                    per_class = {
                        f"cls_{name}": int((cls == cid).sum())
                        for cid, name in sorted(class_names.items())
                    }

                    if writer is not None:
                        if args.fast_draw:
                            draw.draw_boxes(frame, np.asarray(dets.xyxy, np.float32), cls)
                            labels = [
                                f"{class_names.get(int(c), c)} {v:.2f}"
                                for c, v in zip(cls, conf)
                            ]
                            draw.draw_labels(frame, np.asarray(dets.xyxy, np.float32),
                                             labels, cls)
                        else:
                            box_ann, lab_ann = sv_annotators
                            frame = box_ann.annotate(frame, dets)
                            frame = lab_ann.annotate(
                                frame, dets,
                                labels=[f"{class_names.get(int(c), c)} {v:.2f}"
                                        for c, v in zip(cls, conf)],
                            )

                        if args.stats:
                            draw.draw_stats(frame, [
                                "TASK 2  DETECTION",
                                f"detections   {n:4d}   (total {total_det})",
                                f"mean conf    {float(conf.mean()) if n else 0.0:.3f}",
                                f"size  s/m/l  {sizes['small']}/{sizes['medium']}/{sizes['large']}",
                                f"pre {t.pre_ms:5.1f} infer {t.infer_ms:5.1f} post {t.post_ms:5.1f} ms",
                                f"model fps    {t.fps:5.1f}",
                            ])
                        draw.draw_progress(frame, (i + 1) / meta.n_frames)
                        writer.write(frame)
                        common.maybe_save_frame(args, frame, i)

                    rec.add({
                        "frame_id": i,
                        "timestamp_ms": round(i * 1000.0 / meta.fps, 2),
                        "n_det": n,
                        "mean_conf": round(float(conf.mean()) if n else 0.0, 4),
                        "det_small": sizes["small"],
                        "det_medium": sizes["medium"],
                        "det_large": sizes["large"],
                        **per_class,
                        "pre_ms": round(t.pre_ms, 3),
                        "infer_ms": round(t.infer_ms, 3),
                        "post_ms": round(t.post_ms, 3),
                        "total_ms": round(t.total_ms, 3),
                    })
                tick.tick(len(frames))

    elapsed, _ = tick.finish()
    common.report_speed(timings_all, elapsed, rec.n)
    print(f"[info] detections {total_det} total, "
          f"{total_det / max(rec.n, 1):.1f} per frame")

    rec.check_alignment(writer.frames_written if writer else rec.n)
    rec.close(class_names)
    if writer is not None:
        common.finish_video(args, writer.frames_written)
    return 0 if rec.n else 1


if __name__ == "__main__":
    raise SystemExit(main())
