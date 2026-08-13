"""Shared CLI, config loading and run scaffolding for the three renderers.

The three tasks differ only in what they compute and what they draw. Argument
parsing, config resolution, engine construction, threaded I/O and the progress
readout are identical, so they live here once.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "3-pipeline"))

import video_io                       # noqa: E402
from engine import build_engine       # noqa: E402

DEFAULT_SOURCE = os.path.join(ROOT, "video", "uav0000076_00241_s_30fps.mp4")


# --------------------------------------------------------------------------- #
def base_parser(description: str, default_out: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--source", default=DEFAULT_SOURCE, help="input video")
    p.add_argument("--out", default=os.path.join(ROOT, "results", default_out))
    p.add_argument("--start", type=float, default=0.0, help="start offset, seconds")
    p.add_argument("--duration", type=float, default=0.0,
                   help="seconds to process; 0 = to the end")

    p.add_argument("--engine", choices=("ultralytics", "onnx"), default="ultralytics")
    p.add_argument("--model", default=None, help="override the weights path")
    # 640, not 1280. Measured on this clip, 1280 costs 3x the time for no extra
    # detections at conf 0.25 (and only ~15% more at conf 0.10). The model was
    # fine-tuned at 640 on VisDrone stills that were themselves downscaled to
    # it, so running at 1280 presents objects roughly twice the pixel size it
    # learned -- upsampling moves the input away from the training distribution
    # rather than revealing more of it. It also keeps these runs comparable to
    # docs/RESULTS.md, which is a 640 protocol.
    p.add_argument("--imgsz", type=int, default=640,
                   help="inference size; ignored by --engine onnx (graph is fixed)")
    p.add_argument("--batch", type=int, default=8, help="frames per forward pass")
    p.add_argument("--device", default="0", help="CUDA index, or 'cpu'")
    p.add_argument("--half", dest="half", action="store_true", default=True)
    p.add_argument("--no-half", dest="half", action="store_false")
    p.add_argument("--conf", type=float, default=None, help="override pipeline.yaml")
    p.add_argument("--iou", type=float, default=None, help="override pipeline.yaml")

    p.add_argument("--fast-draw", dest="fast_draw", action="store_true", default=True,
                   help="grouped cv2 drawing (default)")
    p.add_argument("--no-fast-draw", dest="fast_draw", action="store_false",
                   help="use supervision's annotators instead")
    p.add_argument("--stats", dest="stats", action="store_true", default=True,
                   help="burn the temporary readout block into the frames")
    p.add_argument("--no-stats", dest="stats", action="store_false")
    p.add_argument("--save-frames", type=int, default=0,
                   help="also write a PNG every N frames (0 = off)")

    # Global motion compensation. On by default because this footage is shot
    # from a moving UAV, and without it a parked car crosses the counting line
    # whenever the drone pans over it.
    p.add_argument("--gmc", dest="gmc", action="store_true", default=True)
    p.add_argument("--no-gmc", dest="gmc", action="store_false")
    p.add_argument("--gmc-scale", type=float, default=0.25,
                   help="optical-flow works at this fraction of the frame")
    p.add_argument("--min-speed", type=float, default=1.2,
                   help="px/frame, camera-compensated, above which a track counts as moving")

    p.add_argument("--config", default=os.path.join(ROOT, "configs", "pipeline.yaml"))
    p.add_argument("--classes", default=os.path.join(ROOT, "configs", "visdrone.yaml"))
    p.add_argument("--h264", action="store_true", help="transcode with ffmpeg when done")
    # 20, not 18. Annotation adds hard edges and text, which are expensive to
    # encode; at crf 18 a 52 s render lands near 170 MB, and 20 costs roughly
    # a third of that for no visible difference on these overlays.
    p.add_argument("--crf", type=int, default=20, help="x264 quality; lower = larger")
    p.add_argument("--cv-threads", type=int, default=4,
                   help="cv2.setNumThreads; left unbounded OpenCV competes with torch")
    p.add_argument("--benchmark", type=int, default=0,
                   help="time N frames and exit without writing a video")
    return p


def load_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_configs(args) -> tuple[dict, dict]:
    for path in (args.config, args.classes):
        if not os.path.isfile(path):
            raise SystemExit(f"[fatal] config not found: {path}")
    return load_yaml(args.config), load_yaml(args.classes)


# --------------------------------------------------------------------------- #
def setup(args, task: str):
    """Resolve config, open the source, build the engine. Shared by all three."""
    import cv2

    cv2.setNumThreads(max(1, args.cv_threads))

    pcfg, vcfg = load_configs(args)
    meta = video_io.probe(args.source, args.start, args.duration)
    if args.benchmark:
        meta.n_frames = min(meta.n_frames, args.benchmark)
        # Divert the record too. A timing run must not overwrite the CSV/JSON
        # from a real render that shares the default output name.
        base, ext = os.path.splitext(args.out)
        args.out = f"{base}.bench{ext}"

    engine = build_engine(args, pcfg.get("model", {}))
    class_names = engine.class_names or {
        int(k): str(v) for k, v in vcfg["names"].items()
    }

    print(f"[info] task      {task}")
    print(f"[info] source    {os.path.basename(meta.path)} "
          f"{meta.width}x{meta.height} @ {meta.fps:.2f} fps, "
          f"{meta.n_frames} frames ({meta.duration_s:.1f}s)")
    print(f"[info] engine    {engine.info.summary()}")

    print("[info] warming up")
    engine.warmup((meta.height, meta.width), n=2)

    run_meta = {"task": task, **meta.as_dict(), **engine.info.as_dict(),
                "fast_draw": bool(args.fast_draw)}
    return pcfg, vcfg, meta, engine, class_names, run_meta


# --------------------------------------------------------------------------- #
class null_ctx:
    """Stand-in for the writer when ``--benchmark`` suppresses video output."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class Ticker:
    """Progress and throughput. Reports end-to-end fps, not just the model's."""

    def __init__(self, total: int, every: int = 60) -> None:
        self.total = max(total, 1)
        self.every = every
        self.t0 = time.perf_counter()
        self.n = 0

    def tick(self, n: int = 1) -> None:
        self.n += n
        if self.n % self.every < n or self.n >= self.total:
            el = time.perf_counter() - self.t0
            fps = self.n / max(el, 1e-6)
            eta = (self.total - self.n) / max(fps, 1e-6)
            print(f"[run] {self.n:5d}/{self.total}  {fps:6.1f} fps  eta {eta:5.1f}s",
                  flush=True)

    def finish(self) -> tuple[float, float]:
        el = time.perf_counter() - self.t0
        fps = self.n / max(el, 1e-6)
        print(f"[done] {self.n} frames in {el:.1f}s  ({fps:.1f} fps end-to-end)")
        return el, fps


def mean_timing(timings) -> dict:
    """Average the latency split over a run. Reported split, never as one number."""
    if not timings:
        return {"pre_ms": 0.0, "infer_ms": 0.0, "post_ms": 0.0, "total_ms": 0.0}
    pre = float(np.mean([t.pre_ms for t in timings]))
    inf = float(np.mean([t.infer_ms for t in timings]))
    post = float(np.mean([t.post_ms for t in timings]))
    return {"pre_ms": pre, "infer_ms": inf, "post_ms": post,
            "total_ms": pre + inf + post}


def report_speed(all_timings, elapsed: float, n: int) -> None:
    m = mean_timing(all_timings)
    print(f"[info] model     pre {m['pre_ms']:5.2f} | infer {m['infer_ms']:6.2f} | "
          f"post {m['post_ms']:5.2f} = {m['total_ms']:6.2f} ms "
          f"({1000.0 / max(m['total_ms'], 1e-6):.1f} fps model-only)")
    if n:
        print(f"[info] wall      {elapsed * 1000.0 / n:6.2f} ms/frame "
              f"({n / max(elapsed, 1e-6):.1f} fps end-to-end incl. decode/draw/encode)")


def maybe_save_frame(args, frame, idx: int) -> None:
    if not args.save_frames or idx % args.save_frames:
        return
    import cv2

    d = os.path.join(os.path.dirname(os.path.abspath(args.out)), "frames")
    os.makedirs(d, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.out))[0]
    cv2.imwrite(os.path.join(d, f"{base}_{idx:06d}.png"), frame)


def finish_video(args, writer_frames: int) -> None:
    print(f"[ok] {args.out}  ({writer_frames} frames)")
    if args.h264:
        video_io.transcode_h264(args.out, crf=args.crf)
