#!/usr/bin/env python3
"""Build an offline weather-augmented VisDrone dataset in Ultralytics format.

Writes real image files so the result can be uploaded to Colab and trained
without this repository being present.

    python 2-augment/build_dataset.py \
        --train data/VisDrone2019-DET-train \
        --val   data/VisDrone2019-DET-val \
        --out   D:/visdrone_weather

Why the labels are simply copied
--------------------------------
Every condition here is **photometric**: rain, exposure and motion blur change
pixel values but not geometry. No box moves, so the augmented twin of an image
carries byte-identical labels. This is the reason offline augmentation is safe
here and would *not* be safe for flips, crops or rotations.

Composition
-----------
``--mode pair`` (default) writes each source image twice: once clean, once
degraded with a sampled condition. The model keeps its clean-weather accuracy
because it still sees clean data, and learns the degraded case from the twin.

``--mode mixed`` writes each source image once, degrading a ``--degrade-frac``
share of them. Same file count as the original dataset, so roughly half the
upload — at the cost of having no clean/degraded pairs.

Reproducibility
---------------
Every assignment is derived from ``--seed`` and the image index, and the full
mapping is written to ``manifest.csv``. Regenerating with the same seed gives
byte-identical output.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "4-bench"))

import cv2  # noqa: E402

import degradations as D  # noqa: E402
from visdrone_data import CAT_TO_CLS, CLASS_NAMES  # noqa: E402

# The conditions requested for training, with sampling weights.
# Deliberately excludes fog and bright_up: the benchmark showed the clean
# model already retains 94% and 97% of AP under those, so spending training
# capacity on them buys nothing.
DEFAULT_CONDITIONS = [
    ("rain_light", 0.20),
    ("rain_medium", 0.22),
    ("rain_heavy", 0.18),
    ("bright_down", 0.15),
    ("bright_up", 0.10),
    ("blur_light", 0.15),
]


def parse_visdrone_ann(path: str) -> tuple[list[tuple[int, float, float, float, float]], int, int]:
    """Return (objects, n_ignored, n_others) with boxes as absolute xyxy."""
    objs: list[tuple[int, float, float, float, float]] = []
    n_ign = n_oth = 0
    if not os.path.exists(path):
        return objs, n_ign, n_oth
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip().rstrip(",")
            if not line:
                continue
            p = line.split(",")
            if len(p) < 6:
                continue
            try:
                x, y, w, h = (float(p[i]) for i in range(4))
                score = float(p[4])
                cat = int(float(p[5]))
            except ValueError:
                continue
            if w <= 0 or h <= 0:
                continue
            if cat == 0:
                n_ign += 1
                continue
            if cat == 11 or cat not in CAT_TO_CLS:
                n_oth += 1
                continue
            if score == 0:
                n_ign += 1
                continue
            objs.append((CAT_TO_CLS[cat], x, y, x + w, y + h))
    return objs, n_ign, n_oth


def to_yolo_lines(objs, img_w: int, img_h: int) -> list[str]:
    """Absolute xyxy -> normalised YOLO `cls cx cy w h`, clipped to the frame."""
    out = []
    for cls, x1, y1, x2, y2 in objs:
        x1 = max(0.0, min(float(x1), img_w))
        y1 = max(0.0, min(float(y1), img_h))
        x2 = max(0.0, min(float(x2), img_w))
        y2 = max(0.0, min(float(y2), img_h))
        w = x2 - x1
        h = y2 - y1
        if w <= 1.0 or h <= 1.0:      # degenerate after clipping
            continue
        cx = (x1 + w / 2.0) / img_w
        cy = (y1 + h / 2.0) / img_h
        out.append(f"{cls} {cx:.6f} {cy:.6f} {w / img_w:.6f} {h / img_h:.6f}")
    return out


def sample_condition(rng: random.Random, conds) -> str:
    r = rng.random() * sum(w for _, w in conds)
    acc = 0.0
    for name, w in conds:
        acc += w
        if r <= acc:
            return name
    return conds[-1][0]


# --------------------------------------------------------------------------- #
def process_split(
    src_root: str, out_root: str, split: str, mode: str,
    degrade_frac: float, conds, seed: int, jpeg_quality: int,
    augment: bool, limit: int | None, writer_rows: list,
) -> dict:
    img_dir = os.path.join(src_root, "images")
    ann_dir = os.path.join(src_root, "annotations")
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"missing {img_dir}")

    out_img = os.path.join(out_root, "images", split)
    out_lbl = os.path.join(out_root, "labels", split)
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)

    names = sorted(n for n in os.listdir(img_dir) if n.lower().endswith(".jpg"))
    if limit:
        names = names[:limit]

    rng = random.Random(seed)
    stats = {"src": len(names), "written": 0, "degraded": 0,
             "objects": 0, "ignored": 0, "others": 0, "skipped": 0}
    t0 = time.perf_counter()

    for i, name in enumerate(names):
        stem = os.path.splitext(name)[0]
        src_img = os.path.join(img_dir, name)
        img = cv2.imread(src_img)
        if img is None:
            stats["skipped"] += 1
            continue
        h, w = img.shape[:2]

        objs, n_ign, n_oth = parse_visdrone_ann(os.path.join(ann_dir, stem + ".txt"))
        lines = to_yolo_lines(objs, w, h)
        stats["objects"] += len(lines)
        stats["ignored"] += n_ign
        stats["others"] += n_oth

        def emit(out_stem: str, image, condition: str) -> None:
            cv2.imwrite(os.path.join(out_img, out_stem + ".jpg"), image,
                        [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
            with open(os.path.join(out_lbl, out_stem + ".txt"), "w",
                      encoding="utf-8") as f:
                f.write("\n".join(lines))
                if lines:
                    f.write("\n")
            stats["written"] += 1
            writer_rows.append([split, out_stem, name, condition, len(lines)])

        if not augment:
            emit(stem, img, "clean")
        elif mode == "pair":
            emit(stem, img, "clean")
            cond = sample_condition(rng, conds)
            emit(f"{stem}__{cond}",
                 D.apply_condition(img, cond, seed=rng.randrange(1 << 30)), cond)
            stats["degraded"] += 1
        else:  # mixed
            if rng.random() < degrade_frac:
                cond = sample_condition(rng, conds)
                emit(f"{stem}__{cond}",
                     D.apply_condition(img, cond, seed=rng.randrange(1 << 30)), cond)
                stats["degraded"] += 1
            else:
                emit(stem, img, "clean")

        if (i + 1) % 250 == 0:
            el = time.perf_counter() - t0
            rate = (i + 1) / max(el, 1e-9)
            eta = (len(names) - i - 1) / max(rate, 1e-9)
            print(f"    [{split}] {i + 1}/{len(names)}  "
                  f"{rate:.1f} img/s  ETA {eta / 60:.1f} min", flush=True)

    stats["seconds"] = time.perf_counter() - t0
    return stats


def dir_size_gb(path: str) -> float:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total / 1e9


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="VisDrone2019-DET-train dir")
    ap.add_argument("--val", default=None, help="VisDrone2019-DET-val dir")
    ap.add_argument("--out", required=True, help="output dataset root")
    ap.add_argument("--mode", default="pair", choices=["pair", "mixed"])
    ap.add_argument("--degrade-frac", type=float, default=0.55,
                    help="mixed mode only")
    ap.add_argument("--augment-val", action="store_true",
                    help="also degrade the val split (default: keep val clean "
                         "so training metrics stay comparable to the baseline)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jpeg-quality", type=int, default=92)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap images/split")
    ap.add_argument("--clean", action="store_true", help="wipe the output dir first")
    args = ap.parse_args()

    out = os.path.abspath(args.out)
    if args.clean and os.path.isdir(out):
        print(f"[info] removing {out}")
        shutil.rmtree(out)
    os.makedirs(out, exist_ok=True)

    conds = DEFAULT_CONDITIONS
    print(f"[info] mode={args.mode}  seed={args.seed}  jpeg_q={args.jpeg_quality}")
    print(f"[info] conditions: {[c for c, _ in conds]}")
    print(f"[info] out: {out}")

    rows: list[list] = []
    limit = args.limit or None

    print("[run] train split")
    st_train = process_split(args.train, out, "train", args.mode,
                             args.degrade_frac, conds, args.seed,
                             args.jpeg_quality, True, limit, rows)

    st_val = None
    if args.val:
        print("[run] val split")
        st_val = process_split(args.val, out, "val", args.mode,
                               args.degrade_frac, conds, args.seed + 1,
                               args.jpeg_quality, args.augment_val, limit, rows)

    # -- manifest ------------------------------------------------------
    with open(os.path.join(out, "manifest.csv"), "w", newline="",
              encoding="utf-8") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["split", "out_stem", "source_image", "condition", "n_objects"])
        wcsv.writerows(rows)

    # -- data.yaml -----------------------------------------------------
    yaml_path = os.path.join(out, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("# VisDrone + offline weather augmentation\n")
        f.write("# Generated by 2-augment/build_dataset.py\n")
        f.write("# Class order is frozen and matches configs/visdrone.yaml --\n")
        f.write("# do not reorder, the fine-tuned checkpoint depends on it.\n")
        f.write(f"path: {out.replace(os.sep, '/')}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n" if args.val else "val: images/train\n")
        f.write(f"nc: {len(CLASS_NAMES)}\n")
        f.write("names:\n")
        for i, n in enumerate(CLASS_NAMES):
            f.write(f"  {i}: {n}\n")

    # -- summary -------------------------------------------------------
    print("\n=== summary ===")
    for nm, st in (("train", st_train), ("val", st_val)):
        if st is None:
            continue
        print(f"  {nm}: {st['src']} source -> {st['written']} images "
              f"({st['degraded']} degraded), {st['objects']} boxes, "
              f"{st['ignored']} ignored + {st['others']} others dropped, "
              f"{st['seconds'] / 60:.1f} min")
    size = dir_size_gb(out)
    print(f"  total on disk: {size:.2f} GB")
    print(f"  data.yaml: {yaml_path}")
    print(f"  manifest:  {os.path.join(out, 'manifest.csv')}")
    print("\nOn Colab:")
    print("  yolo train model=best.pt data=<path>/data.yaml imgsz=640 epochs=60")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
