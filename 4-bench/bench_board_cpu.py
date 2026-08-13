#!/usr/bin/env python3
"""Measure what the board itself does, with no NPU and no torch.

    python3 bench_board_cpu.py --n 60

Runs on the QCS8550 over SSH with nothing but Python 3.10, numpy and cv2 --
which is all the stock Qualcomm Linux image has. It deliberately does not need
onnxruntime, because the board has none and no way to install one.

What this measures is the *host side* of the frame budget: the work the Kryo
cores have to do around the NPU whatever the NPU costs. On this pipeline that
is letterbox, colour conversion, the float cast, and NMS -- and NMS alone was
18 ms per frame on the laptop, larger than the entire quantized forward pass.
Those milliseconds do not disappear when the model gets faster, so they belong
in the budget as their own line.

Thermals and clocks are sampled around the measurement rather than quoted from
a datasheet, because a board that throttles reports different numbers on the
second run than the first, and the only way to know is to look.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import numpy as np

try:
    import cv2
except ImportError:
    raise SystemExit("cv2 khong co tren board")


def read_first(*paths, default=None):
    for p in paths:
        try:
            with open(p) as f:
                return f.read().strip()
        except OSError:
            continue
    return default


def thermals():
    """Hottest zone plus the ones named for the compute blocks."""
    out, hot = {}, (-1e9, "")
    base = "/sys/class/thermal"
    try:
        zones = sorted(z for z in os.listdir(base) if z.startswith("thermal_zone"))
    except OSError:
        return out, None
    for z in zones:
        t = read_first(f"{base}/{z}/temp")
        n = read_first(f"{base}/{z}/type", default=z)
        if t is None:
            continue
        c = int(t) / 1000.0
        if c > hot[0]:
            hot = (c, n)
        if any(k in (n or "") for k in ("nspss", "cpu-1-0", "gpuss-0")):
            out[n] = c
    return out, hot


def clocks():
    out = {}
    for c in (0, 4, 7):
        f = f"/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_cur_freq"
        v = read_first(f)
        if v:
            out[f"cpu{c}_mhz"] = int(v) // 1000
    g = read_first("/sys/class/devfreq/3d00000.qcom,kgsl-3d0/cur_freq")
    if g:
        out["gpu_mhz"] = int(g) // 1_000_000
    return out


def rss_mb():
    v = read_first("/proc/self/status")
    if not v:
        return None
    for line in v.splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024
    return None


# --------------------------------------------------------------------------- #
def letterbox(img, new=640, pad=114):
    h, w = img.shape[:2]
    r = min(new / h, new / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    im = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (new - nh) // 2
    left = (new - nw) // 2
    out = np.full((new, new, 3), pad, np.uint8)
    out[top:top + nh, left:left + nw] = im
    return out


def nms_numpy(boxes, scores, thr):
    if boxes.size == 0:
        return np.empty((0,), np.int64)
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


def timed(fn, n, warm=3):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1000)
    ts.sort()
    return {"mean": statistics.fmean(ts), "p50": ts[len(ts) // 2],
            "p95": ts[int(len(ts) * .95) - 1], "min": ts[0], "max": ts[-1]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--src-w", type=int, default=1067, help="be rong o da cat")
    ap.add_argument("--src-h", type=int, default=600)
    ap.add_argument("--anchors", type=int, default=8400)
    ap.add_argument("--nc", type=int, default=10)
    ap.add_argument("--cands", type=int, default=300,
                    help="so ung vien vao NMS; VisDrone dong thi cao hon nhieu")
    ap.add_argument("--out", default="board_cpu.json")
    args = ap.parse_args()

    print(f"nproc {os.cpu_count()}  cv2 {cv2.__version__}  numpy {np.__version__}")
    t0, hot0 = thermals()
    print(f"nhiet truoc: {hot0[1]} {hot0[0]:.1f}C   clocks {clocks()}")

    rng = np.random.default_rng(0)
    img = (rng.random((args.src_h, args.src_w, 3)) * 255).astype(np.uint8)

    res = {}
    res["letterbox"] = timed(lambda: letterbox(img, args.imgsz), args.n)

    lb = letterbox(img, args.imgsz)
    res["bgr2rgb_chw"] = timed(
        lambda: np.ascontiguousarray(lb[:, :, ::-1].transpose(2, 0, 1)), args.n)

    x8 = np.ascontiguousarray(lb[:, :, ::-1].transpose(2, 0, 1))
    res["uint8_to_float"] = timed(lambda: x8.astype(np.float32) / 255.0, args.n)

    # Decode the raw head the way the pipeline does, then NMS. Candidate count
    # is the knob that matters: NMS is superlinear in it, and dense aerial
    # frames sit far above the COCO-ish default.
    raw = rng.random((1, 4 + args.nc, args.anchors)).astype(np.float32)
    raw[0, :4] *= args.imgsz

    def decode_nms():
        p = raw[0].T
        cls = p[:, 4:].argmax(1)
        conf = p[np.arange(p.shape[0]), 4 + cls]
        m = conf >= np.partition(conf, -args.cands)[-args.cands]
        b, c, k = p[m, :4], conf[m], cls[m]
        xy, wh = b[:, :2], b[:, 2:4]
        xyxy = np.concatenate([xy - wh / 2, xy + wh / 2], 1)
        off = k.astype(np.float32) * (args.imgsz * 4.0)
        nms_numpy(xyxy + off[:, None], c, 0.65)

    res["decode_nms"] = timed(decode_nms, max(10, args.n // 4))

    pre = sum(res[k]["mean"] for k in ("letterbox", "bgr2rgb_chw", "uint8_to_float"))
    post = res["decode_nms"]["mean"]

    print(f"\n{'khau':18s}{'mean':>9s}{'p50':>9s}{'p95':>9s}  (ms)")
    for k, v in res.items():
        print(f"{k:18s}{v['mean']:>9.2f}{v['p50']:>9.2f}{v['p95']:>9.2f}")
    print(f"\n  tien xu ly (CPU)      {pre:6.2f} ms")
    print(f"  hau xu ly + NMS (CPU) {post:6.2f} ms")
    print(f"  TONG phan CPU         {pre + post:6.2f} ms / o")
    print(f"  x4 o cho mot khung    {(pre + post) * 4:6.2f} ms")

    t1, hot1 = thermals()
    print(f"\nnhiet sau : {hot1[1]} {hot1[0]:.1f}C  (tang {hot1[0]-hot0[0]:+.1f}C)")
    print(f"clocks    : {clocks()}")
    print(f"RSS       : {rss_mb():.0f} MB")

    json.dump({"stages": res, "pre_ms": pre, "post_ms": post,
               "cpu_total_ms_per_tile": pre + post,
               "cpu_total_ms_per_frame_4tiles": (pre + post) * 4,
               "thermal_before": {"zone": hot0[1], "c": hot0[0]},
               "thermal_after": {"zone": hot1[1], "c": hot1[0]},
               "zones_before": t0, "zones_after": t1,
               "clocks_after": clocks(), "rss_mb": rss_mb(),
               "nproc": os.cpu_count(), "cv2": cv2.__version__,
               "numpy": np.__version__, "imgsz": args.imgsz,
               "tile": [args.src_w, args.src_h], "anchors": args.anchors,
               "nms_candidates": args.cands},
              open(args.out, "w"), indent=2)
    print(f"\n[ok] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
