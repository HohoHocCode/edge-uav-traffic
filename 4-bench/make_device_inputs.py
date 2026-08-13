#!/usr/bin/env python3
"""Preprocess a VisDrone split once, into raw tensors the board can execute.

    python 4-bench/make_device_inputs.py --data data/VisDrone2019-DET-val \
        --out results/device_in --limit 200

Why this exists: the board runs Android 13 with toybox and no Python, so it
cannot letterbox anything. But for an *offline* evaluation it does not need
to -- the images never change, so preprocessing is done once here and the
board only ever executes the graph.

That split is the whole point. The accuracy tables in docs/QUANTIZATION.md
sections 3, 4 and 6 are simulated on the host with ONNX Runtime. Feeding the
same tensors to the real Hexagon and scoring the outputs is what turns them
into device measurements.

The manifest is the part that must not be got wrong. Every image has its own
``gain`` and ``pad`` because VisDrone val holds at least three resolutions
(960x540, 1360x765, 1920x1080), and a detection decoded with another image's
letterbox parameters lands in the wrong place with no error anywhere. So the
manifest records them per image, in the same order as the input list, and
``eval_device_outputs.py`` reads them back by index.

Sizing, before you run it on the full split: float32 input is 4.9 MB per
image, so 548 images is 2.7 GB pushed and ~258 MB of outputs pulled back.
``--dtype float16`` halves the push for fp16 graphs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "3-pipeline"))
sys.path.insert(0, os.path.join(ROOT, "2-augment"))

import cv2  # noqa: E402

import degradations as D  # noqa: E402
from detector import letterbox  # noqa: E402  <- the runtime's own function
from visdrone_data import load_split  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data",
                                                   "VisDrone2019-DET-val"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "device_in"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--pad-value", type=int, default=114)
    ap.add_argument("--limit", type=int, default=0, help="0 = all images")
    ap.add_argument("--condition", default="clean",
                    help="degradation to apply before letterboxing; the board "
                         "then measures AP under that condition")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    ap.add_argument("--device-dir", default="/data/local/tmp/sky/eval",
                    help="where the files will live on the board; the input "
                         "list must hold paths as the board will see them")
    args = ap.parse_args()

    samples = load_split(args.data)
    if not samples:
        print(f"[fatal] no images under {args.data}", file=sys.stderr)
        return 2
    if args.limit:
        samples = samples[: args.limit]

    if args.condition != "clean" and args.condition not in D.CONDITION_IDS:
        print(f"[fatal] unknown condition {args.condition!r}", file=sys.stderr)
        return 2

    os.makedirs(args.out, exist_ok=True)
    dt = np.float32 if args.dtype == "float32" else np.float16

    manifest = []
    listing = []
    total_bytes = 0

    for i, s in enumerate(samples):
        img = cv2.imread(s.image_path)
        if img is None:
            print(f"[warn] unreadable, skipped: {s.image_path}")
            continue
        if args.condition != "clean":
            img = D.apply_condition(img, args.condition, seed=i)

        h, w = img.shape[:2]
        lb, gain, pad = letterbox(img, args.imgsz, args.pad_value)
        x = lb[:, :, ::-1]
        x = np.ascontiguousarray(x.transpose(2, 0, 1), dtype=np.float32)
        x /= 255.0

        name = f"{len(manifest):05d}.raw"
        x.astype(dt).tofile(os.path.join(args.out, name))
        total_bytes += x.astype(dt).nbytes

        listing.append(f"{args.device_dir}/raw/{name}")
        manifest.append({
            "index": len(manifest),
            "file": name,
            "image_id": s.image_id,
            "src_h": int(h),
            "src_w": int(w),
            "gain": float(gain),
            "pad": [float(pad[0]), float(pad[1])],
        })

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(samples)}", flush=True)

    if not manifest:
        print("[fatal] nothing preprocessed", file=sys.stderr)
        return 2

    with open(os.path.join(args.out, "input_list.txt"), "w",
              newline="\n", encoding="utf-8") as f:
        f.write("\n".join(listing) + "\n")

    meta = {
        "n": len(manifest),
        "imgsz": args.imgsz,
        "pad_value": args.pad_value,
        "dtype": args.dtype,
        "condition": args.condition,
        "data": os.path.abspath(args.data),
        "device_dir": args.device_dir,
        "items": manifest,
    }
    with open(os.path.join(args.out, "manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump(meta, f)

    print(f"[ok] {len(manifest)} tensors in {args.out}  "
          f"({total_bytes / 1e6:.0f} MB, {args.dtype})")
    print(f"[ok] input_list.txt uses board paths under {args.device_dir}/raw/")
    print("\nNext:")
    print(f"  adb push {args.out}/  {args.device_dir}/raw")
    print(f"  adb shell qnn-net-run --retrieve_context ... "
          f"--input_list {args.device_dir}/raw/input_list.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
