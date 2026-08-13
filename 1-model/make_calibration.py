#!/usr/bin/env python3
"""Build the calibration tensor set for post-training quantization.

    python 1-model/make_calibration.py --data data/VisDrone2019-DET-train \
        --n 256 --out results/calib

Post-training quantization picks per-tensor scales from the activations a
handful of real images produce. Two things therefore decide whether the
quantized model works, and both are easy to get silently wrong:

**The preprocessing must be byte-identical to the runtime's.** If calibration
sees ``img/255`` in RGB and the deployed pipeline feeds BGR, or pads with 0
where the pipeline pads with 114, the scales are fitted to a distribution the
model never sees again. Accuracy drops, nothing errors, and the cause is
invisible in every log. This script therefore imports the *same*
``letterbox`` and the same constants the detector uses rather than
reimplementing them.

**The images must be representative.** Calibrating only on bright daytime
frames sets ranges that clip at dusk. Sampling is spread across the split
with a fixed seed, and ``--conditions`` can mix in degraded frames so the
ranges cover the weather the model is expected to survive — which matters
here, because the whole robustness result says clean-only data is not what
this model will meet.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "3-pipeline"))
sys.path.insert(0, os.path.join(ROOT, "2-augment"))
sys.path.insert(0, os.path.join(ROOT, "4-bench"))

import cv2  # noqa: E402

import degradations as D  # noqa: E402
from detector import letterbox  # noqa: E402  <- the runtime's own function
from tiled_detector import tile_rects  # noqa: E402
from visdrone_data import load_split  # noqa: E402


def preprocess(img: np.ndarray, imgsz: int, pad_value: int) -> np.ndarray:
    """Exactly what Yolov8Detector.preprocess does, minus the batch axis."""
    lb, _, _ = letterbox(img, imgsz, pad_value)
    x = lb[:, :, ::-1]                                   # BGR -> RGB
    x = np.ascontiguousarray(x.transpose(2, 0, 1), dtype=np.float32)
    x /= 255.0
    return x


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data",
                                                   "VisDrone2019-DET-train"))
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--pad-value", type=int, default=114)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "calib"))
    ap.add_argument("--input-name", default="images",
                    help="ONNX input tensor name; must match the graph")
    ap.add_argument("--conditions", nargs="*", default=None,
                    help="mix degraded frames into the calibration set, e.g. "
                         "rain_medium blur_light. Ranges then cover the "
                         "conditions the model is expected to survive")
    args = ap.parse_args()

    samples = load_split(args.data)
    if not samples:
        print(f"[fatal] no images under {args.data}", file=sys.stderr)
        return 2

    rng = random.Random(args.seed)
    # Spread across the split rather than taking the first N: VisDrone is
    # ordered by sequence, so the head of the list is a handful of scenes.
    idx = sorted(rng.sample(range(len(samples)), min(args.n, len(samples))))

    conds = args.conditions or []
    for c in conds:
        if c not in D.CONDITION_IDS:
            print(f"[fatal] unknown condition {c!r}", file=sys.stderr)
            return 2

    os.makedirs(args.out, exist_ok=True)
    batch = []
    used = []
    for k, i in enumerate(idx):
        s = samples[i]
        img = cv2.imread(s.image_path)
        if img is None:
            continue
        cond = "clean"
        if conds:
            # Deterministic assignment so the set is reproducible.
            pick = rng.random()
            if pick < 0.5:
                cond = conds[rng.randrange(len(conds))]
                img = D.apply_condition(img, cond, seed=i)
        batch.append(preprocess(img, args.imgsz, args.pad_value))
        used.append({"image": s.image_id, "condition": cond})
        if (k + 1) % 64 == 0:
            print(f"  {k + 1}/{len(idx)}", flush=True)

    if not batch:
        print("[fatal] no images read", file=sys.stderr)
        return 2

    arr = np.stack(batch).astype(np.float32)     # (N, 3, H, W)
    npy = os.path.join(args.out, "calibration.npy")
    np.save(npy, arr)

    h = hashlib.sha256(arr.tobytes()).hexdigest()[:16]
    meta = {
        "n": int(arr.shape[0]),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "imgsz": args.imgsz,
        "pad_value": args.pad_value,
        "seed": args.seed,
        "conditions": conds or ["clean"],
        "input_name": args.input_name,
        "sha256_16": h,
        "source": os.path.abspath(args.data),
        "value_range": [float(arr.min()), float(arr.max())],
        "mean": float(arr.mean()),
        "preprocessing": "letterbox -> BGR2RGB -> HWC2CHW -> float32 /255 "
                         "(imported from 3-pipeline/detector.py)",
        "images": used[:32],
    }
    with open(os.path.join(args.out, "calibration.meta.json"), "w",
              encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[ok] {npy}  {arr.shape}  {arr.nbytes / 1e6:.1f} MB")
    print(f"[ok] range [{arr.min():.3f}, {arr.max():.3f}]  mean {arr.mean():.4f}")
    print(f"[ok] calib sha256 {h}  -- record this alongside every quantized row")
    if conds:
        from collections import Counter
        print(f"[ok] mix: {dict(Counter(u['condition'] for u in used))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
