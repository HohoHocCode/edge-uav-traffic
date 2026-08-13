#!/usr/bin/env python3
"""Stream-mode latency: one frame at a time, 100 frames, mean +/- std.

    python 4-bench/bench_stream.py --images <dir> --n 100 --onnx model.onnx
    python 4-bench/bench_stream.py --images <dir> --n 100 --cpu-only   # board

Batch benchmarking answers "how fast can this chip chew through a pile of
images", which is not the question a traffic camera asks. Stream mode feeds one
frame at a time and measures the wall time the deployed loop would actually see,
including per-frame allocation and the tiling merge.

On outlier rejection
--------------------
The caller asked for noisy samples to be discarded. That is fair for the
*steady-state mean*: a scheduler hiccup or a thermal event is not a property of
the model. It is not fair for the tail. p95 exists precisely to describe the bad
frames, so trimming first and quoting p95 after would report a tail with the
tail removed.

So this script reports both, side by side, and never lets one masquerade as the
other:

  * raw       -- every retained frame, and where p50/p95/p99 come from
  * trimmed   -- MAD-based rejection at --mad-k, for mean +/- std only

Median absolute deviation is used rather than mean +/- 3sigma because latency
distributions are right-skewed: a few slow frames inflate sigma enough to hide
themselves inside the acceptance window. MAD does not move when the tail does.

Warm-up frames are dropped before any statistic is computed; a cold cache is a
real cost, but it is a startup cost, not a streaming one, and it is reported
separately instead of being averaged into the steady state.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
import time

import cv2
import numpy as np

# ----------------------------------------------------------------- pipeline


def letterbox(img, size):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    out = np.full((size, size, 3), 114, np.uint8)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (size - nh) // 2, (size - nw) // 2
    out[top:top + nh, left:left + nw] = resized
    return out


def tile_rects(w, h, grid=2, overlap=0.2):
    tw, th = w / (grid - overlap), h / (grid - overlap)
    step_x, step_y = tw * (1 - overlap), th * (1 - overlap)
    rects = []
    for gy in range(grid):
        for gx in range(grid):
            x = min(int(round(gx * step_x)), max(0, w - int(round(tw))))
            y = min(int(round(gy * step_y)), max(0, h - int(round(th))))
            rects.append((x, y, min(int(round(tw)), w - x),
                          min(int(round(th)), h - y)))
    return rects


def nms(boxes, scores, thr):
    if boxes.size == 0:
        return np.empty(0, np.int64)
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= thr]
    return np.asarray(keep, np.int64)


def decode(pred, conf_thres, iou_thres, max_det):
    """Anchor-layout decode: (1, 4+nc, A) -> boxes, scores, classes."""
    p = pred[0] if pred.ndim == 3 else pred
    if p.shape[0] < p.shape[1]:
        p = p.T                                   # (A, 4+nc)
    nc = p.shape[1] - 4
    cls = p[:, 4:4 + nc].argmax(1)
    conf = p[np.arange(p.shape[0]), 4 + cls]
    m = conf >= conf_thres
    if not m.any():
        return np.empty((0, 4), np.float32), np.empty(0), np.empty(0, np.int64)
    xywh, conf, cls = p[m, :4], conf[m], cls[m]
    xy, wh = xywh[:, :2], xywh[:, 2:4] / 2
    boxes = np.concatenate([xy - wh, xy + wh], 1).astype(np.float32)
    keep = nms(boxes + cls[:, None] * 8192, conf, iou_thres)[:max_det]
    return boxes[keep], conf[keep], cls[keep]


# -------------------------------------------------------------- statistics


def mad_trim(xs, k):
    """Drop samples further than k MADs from the median. Returns (kept, dropped)."""
    a = np.asarray(xs, float)
    med = np.median(a)
    mad = np.median(np.abs(a - med))
    if mad == 0:                                  # degenerate: keep everything
        return a, np.empty(0)
    scaled = 1.4826 * mad                         # MAD -> sigma for a normal
    keep = np.abs(a - med) <= k * scaled
    return a[keep], a[~keep]


def summarise(xs, k):
    a = np.asarray(xs, float)
    kept, dropped = mad_trim(a, k)
    q = np.percentile(a, [50, 95, 99])
    return {
        "n": int(a.size),
        # tail statistics: computed on every retained frame, never trimmed
        "mean_raw": float(a.mean()),
        "std_raw": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "p50": float(q[0]), "p95": float(q[1]), "p99": float(q[2]),
        "min": float(a.min()), "max": float(a.max()),
        # steady-state: MAD-trimmed, for mean +/- std only
        "n_kept": int(kept.size),
        "n_dropped": int(dropped.size),
        "mean_trim": float(kept.mean()) if kept.size else float("nan"),
        "std_trim": float(kept.std(ddof=1)) if kept.size > 1 else 0.0,
        "dropped_values": [round(v, 3) for v in sorted(dropped.tolist())][:20],
    }


# -------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="directory of frames")
    ap.add_argument("--n", type=int, default=100, help="frames to measure")
    ap.add_argument("--warmup", type=int, default=10,
                    help="frames discarded before measuring")
    ap.add_argument("--onnx", default=None,
                    help="model; omit to measure the CPU stages only")
    ap.add_argument("--cpu-only", action="store_true",
                    help="skip inference even if --onnx is given")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--grid", type=int, default=2)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.65)
    ap.add_argument("--max-det", type=int, default=500)
    ap.add_argument("--mad-k", type=float, default=3.0,
                    help="MAD multiplier for steady-state trimming")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    files = sorted(sum((glob.glob(os.path.join(args.images, e))
                        for e in ("*.jpg", "*.jpeg", "*.png")), []))
    if not files:
        print(f"[err] no images in {args.images}", file=sys.stderr)
        return 1
    need = args.n + args.warmup
    if len(files) < need:                        # cycle rather than measure fewer
        files = (files * (need // len(files) + 1))[:need]
    files = files[:need]

    sess = None
    if args.onnx and not args.cpu_only:
        import onnxruntime as ort
        sess = ort.InferenceSession(args.onnx,
                                    providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0].name

    per_frame, stages = [], {k: [] for k in
                             ("read", "tile", "pre", "infer", "post")}
    cold = None

    for idx, f in enumerate(files):
        t_frame = time.perf_counter()

        t = time.perf_counter()
        img = cv2.imread(f)
        if img is None:
            continue
        t_read = (time.perf_counter() - t) * 1000

        h, w = img.shape[:2]
        t = time.perf_counter()
        rects = tile_rects(w, h, args.grid, args.overlap)
        crops = [img[y:y + th, x:x + tw] for x, y, tw, th in rects]
        t_tile = (time.perf_counter() - t) * 1000

        t_pre = t_inf = t_post = 0.0
        for crop in crops:
            t = time.perf_counter()
            lb = letterbox(crop, args.imgsz)
            x8 = lb[:, :, ::-1].transpose(2, 0, 1)
            xf = (np.ascontiguousarray(x8).astype(np.float32) / 255.0)[None]
            t_pre += (time.perf_counter() - t) * 1000

            if sess is not None:
                t = time.perf_counter()
                out = sess.run(None, {inp: xf})[0]
                t_inf += (time.perf_counter() - t) * 1000

                t = time.perf_counter()
                decode(out, args.conf, args.iou, args.max_det)
                t_post += (time.perf_counter() - t) * 1000

        total = (time.perf_counter() - t_frame) * 1000

        if idx < args.warmup:
            if cold is None:
                cold = total                      # the very first frame
            continue

        per_frame.append(total)
        stages["read"].append(t_read)
        stages["tile"].append(t_tile)
        stages["pre"].append(t_pre)
        stages["infer"].append(t_inf)
        stages["post"].append(t_post)

    if not per_frame:
        print("[err] nothing measured", file=sys.stderr)
        return 1

    res = {
        "frame": summarise(per_frame, args.mad_k),
        "stages": {k: summarise(v, args.mad_k) for k, v in stages.items()
                   if any(v)},
        "cold_first_frame_ms": cold,
        "config": {
            "n": args.n, "warmup": args.warmup, "imgsz": args.imgsz,
            "grid": args.grid, "overlap": args.overlap, "mad_k": args.mad_k,
            "tiles_per_frame": args.grid ** 2,
            "onnx": os.path.basename(args.onnx) if args.onnx else None,
            "cpu_only": bool(args.cpu_only or not args.onnx),
            "backend": "onnxruntime-cpu" if sess else "none",
            "cv2": cv2.__version__, "numpy": np.__version__,
        },
    }

    f = res["frame"]
    print(f"\nstream, {f['n']} frames after {args.warmup} warm-up\n")
    print(f"  steady state   {f['mean_trim']:7.2f} +/- {f['std_trim']:5.2f} ms"
          f"   ({f['n_dropped']} frame(s) dropped at {args.mad_k} MAD)")
    print(f"  untrimmed      {f['mean_raw']:7.2f} +/- {f['std_raw']:5.2f} ms")
    print(f"  p50 / p95 / p99{f['p50']:7.2f} /{f['p95']:6.2f} /{f['p99']:6.2f} ms"
          f"   <- full sample, tail intact")
    print(f"  min / max      {f['min']:7.2f} /{f['max']:6.2f} ms")
    if cold:
        print(f"  first frame    {cold:7.2f} ms   (cold, excluded)")
    print(f"\n  throughput     {1000 / f['mean_trim']:7.2f} FPS")
    if f["dropped_values"]:
        print(f"  dropped        {f['dropped_values']}")

    print(f"\n  {'stage':7} {'mean':>8} {'+/-':>7} {'p95':>8}  (ms/frame)")
    for k, v in res["stages"].items():
        print(f"  {k:7} {v['mean_trim']:8.2f} {v['std_trim']:7.2f} "
              f"{v['p95']:8.2f}")

    out = args.out or os.path.join("docs", "results", "stream_latency.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=2)
    print(f"\n[ok] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
