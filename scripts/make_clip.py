#!/usr/bin/env python3
"""Synthesise an aerial video clip from VisDrone-DET still images.

VisDrone-DET is a still-image split, but the demo and the tracker need video.
Rather than pull the separate multi-gigabyte VID/MOT archives, this builds a
clip by flying a virtual camera over the high-resolution stills: a slow pan
with a gentle zoom, which is exactly the motion a survey UAV produces.

The result is honest for what we use it for. Object motion within the scene is
not real, so this clip is *not* used for any tracking accuracy claim; it is
used to exercise the pipeline, to record the demo, and to measure end-to-end
latency under a realistic detection load. The report says so.

    python scripts/make_clip.py --data data/VisDrone2019-DET-val \
        --out assets/demo_clip.mp4 --seconds 20
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "4-bench"))
sys.path.insert(0, os.path.join(ROOT, "2-augment"))

import cv2  # noqa: E402


def ease(t: float) -> float:
    """Smoothstep — a linear pan starts and stops with a visible jerk."""
    return t * t * (3.0 - 2.0 * t)


def fly_over(
    img: np.ndarray, n_frames: int, out_w: int, out_h: int, rng: np.random.Generator
) -> list[np.ndarray]:
    h, w = img.shape[:2]
    aspect = out_w / out_h

    # Crop window: start wide, end slightly tighter (a descent).
    z0 = rng.uniform(0.72, 0.85)
    z1 = z0 * rng.uniform(0.80, 0.92)

    def crop_size(z):
        cw = w * z
        ch = cw / aspect
        if ch > h:
            ch = h * z
            cw = ch * aspect
        return cw, ch

    cw0, ch0 = crop_size(z0)
    cw1, ch1 = crop_size(z1)

    x0 = rng.uniform(0, max(w - cw0, 1))
    y0 = rng.uniform(0, max(h - ch0, 1))
    x1 = rng.uniform(0, max(w - cw1, 1))
    y1 = rng.uniform(0, max(h - ch1, 1))

    frames = []
    for i in range(n_frames):
        t = ease(i / max(n_frames - 1, 1))
        cw = cw0 + (cw1 - cw0) * t
        ch = ch0 + (ch1 - ch0) * t
        cx = x0 + (x1 - x0) * t
        cy = y0 + (y1 - y0) * t

        xa, ya = int(round(cx)), int(round(cy))
        xb, yb = int(round(cx + cw)), int(round(cy + ch))
        xa, ya = max(0, xa), max(0, ya)
        xb, yb = min(w, xb), min(h, yb)
        if xb - xa < 16 or yb - ya < 16:
            continue
        frames.append(cv2.resize(img[ya:yb, xa:xb], (out_w, out_h),
                                 interpolation=cv2.INTER_LINEAR))
    return frames


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "VisDrone2019-DET-val"))
    ap.add_argument("--out", default=os.path.join(ROOT, "assets", "demo_clip.mp4"))
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--shots", type=int, default=5, help="how many source images")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--degrade", default=None,
                    help="apply a degradation condition to the whole clip")
    ap.add_argument("--min-objects", type=int, default=60,
                    help="prefer source images with at least this many objects")
    args = ap.parse_args()

    from visdrone_data import load_split

    samples = load_split(args.data)
    if not samples:
        print(f"[fatal] no images under {args.data}", file=sys.stderr)
        return 2

    busy = [s for s in samples if len(s.boxes) >= args.min_objects]
    pool = busy or samples
    print(f"[info] {len(pool)}/{len(samples)} images have >= {args.min_objects} objects")

    rng = np.random.default_rng(args.seed)
    picks = rng.choice(len(pool), size=min(args.shots, len(pool)), replace=False)

    total_frames = int(args.seconds * args.fps)
    per_shot = max(total_frames // len(picks), 2)

    degrade_fn = None
    if args.degrade:
        import degradations as D
        if args.degrade not in D.CONDITION_IDS:
            print(f"[fatal] unknown condition {args.degrade!r}; "
                  f"known: {D.CONDITION_IDS}", file=sys.stderr)
            return 2
        degrade_fn = lambda im, i: D.apply_condition(im, args.degrade, seed=i)  # noqa: E731

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (args.width, args.height))
    if not writer.isOpened():
        print("[fatal] cannot open the video writer (missing codec?)", file=sys.stderr)
        return 2

    n_written = 0
    for k, pi in enumerate(picks):
        s = pool[int(pi)]
        img = cv2.imread(s.image_path)
        if img is None:
            continue
        frames = fly_over(img, per_shot, args.width, args.height, rng)
        for f in frames:
            if degrade_fn is not None:
                f = degrade_fn(f, n_written)
            writer.write(f)
            n_written += 1
        print(f"  shot {k + 1}/{len(picks)}: {s.image_id} "
              f"({len(s.boxes)} objs) -> {len(frames)} frames")

    writer.release()
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"[ok] {args.out}  {n_written} frames, "
          f"{n_written / args.fps:.1f}s, {size_mb:.1f} MB")
    print("[note] camera motion is synthetic; do not quote tracking accuracy "
          "from this clip.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
