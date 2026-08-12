#!/usr/bin/env python3
"""Fetch the VisDrone-DET validation split (548 images, ~78 MB).

Pulled from the Ultralytics assets mirror, which is the same archive the
official VisDrone.yaml uses and does not require a Google Drive login.

    python scripts/fetch_data.py
    python scripts/fetch_data.py --split train      # 6471 images, ~1.4 GB
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

URLS = {
    "val": "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip",
    "train": "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-train.zip",
    "test-dev": "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-test-dev.zip",
}


def _progress(done: int, total: int) -> None:
    if total <= 0:
        sys.stdout.write(f"\r  {done / 1e6:.1f} MB")
    else:
        pct = 100.0 * done / total
        bar = "#" * int(pct // 3)
        sys.stdout.write(f"\r  [{bar:<33}] {pct:5.1f}%  "
                         f"{done / 1e6:.1f}/{total / 1e6:.1f} MB")
    sys.stdout.flush()


def download(url: str, dest: str) -> None:
    with urllib.request.urlopen(url) as r:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        with open(dest, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                _progress(done, total)
    print()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=sorted(URLS))
    ap.add_argument("--out", default=os.path.join(ROOT, "data"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    target = os.path.join(args.out, f"VisDrone2019-DET-{args.split}")

    if os.path.isdir(target) and not args.force:
        n = len(os.listdir(os.path.join(target, "images"))) \
            if os.path.isdir(os.path.join(target, "images")) else 0
        print(f"[skip] {target} already exists ({n} images). Use --force to refetch.")
        return 0

    zip_path = target + ".zip"
    print(f"[get] {URLS[args.split]}")
    try:
        download(URLS[args.split], zip_path)
    except OSError as exc:
        print(f"[fatal] download failed: {exc}", file=sys.stderr)
        return 1

    print("[unzip] ...")
    if os.path.isdir(target) and args.force:
        shutil.rmtree(target)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(args.out)
    os.remove(zip_path)

    img_dir = os.path.join(target, "images")
    ann_dir = os.path.join(target, "annotations")
    n_img = len(os.listdir(img_dir)) if os.path.isdir(img_dir) else 0
    n_ann = len(os.listdir(ann_dir)) if os.path.isdir(ann_dir) else 0
    print(f"[ok] {target}: {n_img} images, {n_ann} annotation files")
    if n_img != n_ann:
        print(f"[warn] image/annotation count mismatch ({n_img} vs {n_ann})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
