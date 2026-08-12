#!/usr/bin/env python3
"""Latency ladder — measures the compute cost, independent of any training.

Runs the model over synthetic frames on every requested backend/resolution and
reports the three-way split. Because the numbers do not depend on the weights
being any good, this ladder can be produced the moment the board is reachable,
long before the quality table exists.

    python 4-bench/bench_latency.py --model models/yolov8n_visdrone_640.onnx \
        --backends onnxruntime-cpu --iters 100

What is deliberately measured:

``infer_ms``     the forward pass alone — the compute-unit number
``post_ms``      decode + NMS on the CPU, at a *realistic detection load*

That second point matters. NMS cost scales with the number of surviving
candidates, so timing it on an empty synthetic frame reports a number that
never occurs in the field. The ladder therefore drives postprocessing with a
configurable synthetic candidate load (``--load-boxes``) and reports the cost
at each level, which is what makes the CPU tail legible.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "3-pipeline"))

from detector import Yolov8Detector, nms_numpy  # noqa: E402
from runtime import available_backends, create_session  # noqa: E402


def host_info() -> dict:
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "python": platform.python_version(),
    }
    # On Linux the model name is far more useful than platform.processor().
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.lower().startswith(("model name", "hardware")):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    try:
        import onnxruntime as ort
        info["onnxruntime"] = ort.__version__
    except ImportError:
        pass
    return info


def bench_infer(session, imgsz: int, iters: int, warmup: int) -> dict:
    x = np.random.default_rng(0).random((1, 3, imgsz, imgsz)).astype(np.float32)
    for _ in range(warmup):
        session.run(x)
    session.reset_timers()
    for _ in range(iters):
        session.run(x)
    return session.stats()


def bench_postprocess(det: Yolov8Detector, n_candidates: int, iters: int) -> dict:
    """Time decode+NMS against a synthetic head output with a known load."""
    rng = np.random.default_rng(0)
    nc = 10
    anchors = 8400
    raw = np.zeros((1, 4 + nc, anchors), dtype=np.float32)
    # Boxes spread over the image so NMS actually has to compare them.
    raw[0, 0] = rng.uniform(20, det.imgsz - 20, anchors)
    raw[0, 1] = rng.uniform(20, det.imgsz - 20, anchors)
    raw[0, 2] = rng.uniform(8, 60, anchors)
    raw[0, 3] = rng.uniform(8, 60, anchors)
    # Only n_candidates anchors carry a score above the threshold.
    idx = rng.choice(anchors, size=min(n_candidates, anchors), replace=False)
    cls = rng.integers(0, nc, size=idx.size)
    raw[0, 4 + cls, idx] = rng.uniform(det.conf_thres + 0.01, 0.99, idx.size)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        det.postprocess(raw, 1.0, (0.0, 0.0), (det.imgsz, det.imgsz))
        times.append((time.perf_counter() - t0) * 1000.0)
    a = np.asarray(times)
    return {
        "n_candidates": int(n_candidates),
        "post_ms_avg": float(a.mean()),
        "post_ms_p50": float(np.percentile(a, 50)),
        "post_ms_p95": float(np.percentile(a, 95)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--imgsz", type=int, nargs="*", default=[640])
    ap.add_argument("--backends", nargs="*", default=["auto"])
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--load-boxes", type=int, nargs="*", default=[50, 150, 300, 600],
                    help="synthetic surviving-candidate counts for the NMS tail")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "latency.csv"))
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    backends = args.backends
    if backends == ["auto"]:
        backends = available_backends()
    print(f"[info] backends to measure: {backends}")
    print(f"[info] host: {json.dumps(host_info())}")

    rows = []
    for backend in backends:
        for imgsz in args.imgsz:
            print(f"[run] backend={backend} imgsz={imgsz}")
            try:
                session = create_session(args.model, backend=backend)
            except Exception as exc:
                print(f"    [skip] {type(exc).__name__}: {exc}")
                continue

            det = Yolov8Detector(session, imgsz=imgsz)
            try:
                st = bench_infer(session, imgsz, args.iters, args.warmup)
            except Exception as exc:
                # A model exported at 640 cannot run at 1280 without a
                # re-export; record the failure instead of dropping the cell.
                print(f"    [fail] inference: {type(exc).__name__}: {exc}")
                rows.append({
                    "backend": backend, "imgsz": imgsz, "status": "infer_failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "model": os.path.basename(args.model), "tag": args.tag,
                })
                continue

            print(f"    infer avg {st['avg_ms']:.2f} p50 {st['p50_ms']:.2f} "
                  f"p95 {st['p95_ms']:.2f} ms")

            for nb in args.load_boxes:
                pst = bench_postprocess(det, nb, max(20, args.iters // 4))
                print(f"    post @{nb:4d} cand: avg {pst['post_ms_avg']:.2f} "
                      f"p95 {pst['post_ms_p95']:.2f} ms")
                rows.append({
                    "backend": backend,
                    "providers_active": ";".join(session.info.providers_active),
                    "imgsz": imgsz,
                    "status": "ok",
                    "n_nodes_total": session.info.n_nodes_total,
                    "n_nodes_fallback": session.info.n_nodes_fallback,
                    "infer_ms_avg": st["avg_ms"],
                    "infer_ms_p50": st["p50_ms"],
                    "infer_ms_p95": st["p95_ms"],
                    "iters": st["n"],
                    **pst,
                    "total_ms_avg": st["avg_ms"] + pst["post_ms_avg"],
                    "fps_est": 1000.0 / max(st["avg_ms"] + pst["post_ms_avg"], 1e-9),
                    "model": os.path.basename(args.model),
                    "tag": args.tag,
                })

    if not rows:
        print("[fatal] no rows produced", file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    write_header = not os.path.exists(args.out) or os.path.getsize(args.out) == 0
    with open(args.out, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerows(rows)
    print(f"[ok] {len(rows)} row(s) -> {args.out}")

    with open(os.path.splitext(args.out)[0] + "_host.json", "w", encoding="utf-8") as f:
        json.dump(host_info(), f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
