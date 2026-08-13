#!/usr/bin/env python3
"""Download the quantized models and check they still detect anything.

    python 1-model/check_quant_output.py --fp32 models/new/x.onnx \
        --jobs w8a8=jpxxwyljp w8a16=j5m8j30yp --image <path>

A latency table is not a result until this has run. On the previous model the
fastest precision by a wide margin produced zero detections -- every class
score exactly 0.0, while the box branch stayed healthy, so nothing downstream
raised an error and the profile job reported a perfectly good 1.993 ms.

The comparison is against the fp32 graph on the same frame, so "it detects
something" is measured rather than eyeballed: the count above the deployment
threshold, the score range, and the correlation of the confidence column.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
import zipfile

import numpy as np

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

import onnxruntime as ort  # noqa: E402
import qai_hub as hub  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "3-pipeline"))
sys.path.insert(0, os.path.join(ROOT, "4-bench"))

import cv2  # noqa: E402

from detector import letterbox  # noqa: E402
from tiled_detector import tile_rects  # noqa: E402
from visdrone_data import load_split  # noqa: E402


def fetch(job_id: str, dest: str) -> str | None:
    """Download a quantize job's QDQ onnx; unzip if it arrives as an archive."""
    os.makedirs(dest, exist_ok=True)
    m = hub.get_job(job_id).get_target_model()
    if m is None:
        return None
    p = m.download(dest)
    p = str(p)
    if p.endswith(".zip") or zipfile.is_zipfile(p):
        with zipfile.ZipFile(p) as z:
            z.extractall(dest)
        hits = glob.glob(os.path.join(dest, "**", "*.onnx"), recursive=True)
        return hits[0] if hits else None
    return p


def run(path: str, x: np.ndarray) -> np.ndarray:
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return s.run(None, {s.get_inputs()[0].name: x})[0]


def summarise(out: np.ndarray, conf_thres: float) -> dict:
    """Confidence *and* box geometry. Both, because either one alone lies.

    Two different collapses have now been observed, and a check that looks at
    only one of them calls the other a pass:

    * confidence collapses to exactly 0 while boxes stay healthy (w8a8 on the
      anchor-based head), and
    * boxes collapse to a single degenerate point while confidence tracks fp32
      at a correlation of 0.9995 (every precision on the end-to-end head).

    So a model counts as usable only if detections clear the threshold *and*
    their boxes have non-zero extent.
    """
    a = out[0]
    if a.ndim == 2 and a.shape[1] == 6:          # end-to-end (N, 6)
        conf, box = a[:, 4], a[:, :4]
    else:                                        # (4+nc, A) -> cxcywh
        conf = a[4:].max(axis=0)
        cxcywh = a[:4].T
        box = np.stack([cxcywh[:, 0] - cxcywh[:, 2] / 2,
                        cxcywh[:, 1] - cxcywh[:, 3] / 2,
                        cxcywh[:, 0] + cxcywh[:, 2] / 2,
                        cxcywh[:, 1] + cxcywh[:, 3] / 2], 1)

    m = conf > conf_thres
    n = int(m.sum())
    if n:
        w = box[m, 2] - box[m, 0]
        h = box[m, 3] - box[m, 1]
        n_valid = int(((w > 1) & (h > 1)).sum())
        wh = f"{w.mean():.1f}x{h.mean():.1f}"
    else:
        n_valid, wh = 0, "-"

    return {
        "n_above_thres": n,
        "n_box_valid": n_valid,
        "mean_wh": wh,
        "conf_max": float(conf.max()),
        "box_min": float(box.min()),
        "box_max": float(box.max()),
        "_conf": conf,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32", required=True)
    ap.add_argument("--jobs", nargs="+", required=True, metavar="NAME=JOBID")
    ap.add_argument("--data", default=os.path.join(ROOT, "data",
                                                   "VisDrone2019-DET-val"))
    ap.add_argument("--conf", type=float, default=0.25,
                    help="deployment threshold; the AP protocol uses 0.001 but "
                         "a model is unusable if nothing clears this")
    ap.add_argument("--dest", default="models/quant")
    ap.add_argument("--out", default="results/quant_output_check.csv")
    args = ap.parse_args()

    # One real tile, preprocessed exactly as the runtime does it.
    s = load_split(args.data)[0]
    img = cv2.imread(s.image_path)
    H, W = img.shape[:2]
    tx, ty, tw, th = tile_rects(W, H, 2, 0.20)[0]
    lb, _, _ = letterbox(img[ty:ty + th, tx:tx + tw], 640, 114)
    x = np.ascontiguousarray(lb[:, :, ::-1].transpose(2, 0, 1),
                             dtype=np.float32)[None] / 255.0
    print(f"[info] anh {s.image_id}, o {tw}x{th} -> 640x640\n")

    ref = summarise(run(args.fp32, x), args.conf)
    rows = [{"precision": "fp32", "path": os.path.basename(args.fp32),
             **{k: v for k, v in ref.items() if not k.startswith("_")},
             "conf_corr_vs_fp32": 1.0, "usable": "yes"}]
    def line(name, d, corr, verdict):
        print(f"{name:9s}{d['n_above_thres']:>8d}{d['n_box_valid']:>10d}"
              f"{d['mean_wh']:>12s}{d['conf_max']:>10.5f}{corr:>8.4f}  {verdict}")

    print(f"{'precision':9s}{'n>conf':>8s}{'box hop le':>10s}"
          f"{'rong x cao':>12s}{'conf_max':>10s}{'corr':>8s}  ket luan")
    line("fp32", ref, 1.0, "moc so sanh")

    for kv in args.jobs:
        name, jid = kv.split("=", 1)
        p = fetch(jid, os.path.join(args.dest, name))
        if not p:
            print(f"{name:9s} tai that bai")
            continue
        cur = summarise(run(p, x), args.conf)
        corr = float(np.corrcoef(ref["_conf"], cur["_conf"])[0, 1]) \
            if cur["_conf"].shape == ref["_conf"].shape and cur["conf_max"] > 0 else 0.0
        # Dung duoc = vua co detection vua co box that. Thieu mot trong hai
        # thi model khong dung duoc, du con lai co dep den may.
        if cur["n_above_thres"] == 0:
            usable = "NO - conf sap ve 0"
        elif cur["n_box_valid"] == 0:
            usable = "NO - box sap thanh mot diem"
        else:
            usable = "yes"
        rows.append({"precision": name, "path": os.path.basename(p),
                     **{k: v for k, v in cur.items() if not k.startswith("_")},
                     "conf_corr_vs_fp32": round(corr, 5), "usable": usable})
        line(name, cur, corr,
             "dung duoc" if usable == "yes" else "*** " + usable[5:] + " ***")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)
    print(f"\n[ok] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
