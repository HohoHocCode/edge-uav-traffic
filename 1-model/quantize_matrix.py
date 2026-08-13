#!/usr/bin/env python3
"""Quantize one ONNX model at several precisions and profile each on QCS8550.

    python 1-model/quantize_matrix.py --model models/new/v26n-base-encoder-md500.onnx \
        --calib results/calib_tile/calibration.npy --tag v26n-base

The flow, and why each step is where it is:

    ONNX fp32
      -> upload calibration once, reuse for every precision
      -> quantize job   (weights x activations dtype)   -> QDQ ONNX
      -> compile job    (--target_runtime qnn_context_binary, pinned QAIRT)
      -> profile job    (real QCS8550)                  -> latency, fallback
      -> inference job  (one frame)                     -> does it still detect?

The last step is the one that is easy to skip and expensive to skip. On the
previous model w8a8 was the fastest configuration by a wide margin and produced
zero detections -- every class score exactly 0.0 while the box branch stayed
healthy. A table of latencies with no output check recommends that
configuration.

QAIRT is pinned because a context binary is compiled for a specific runtime
version. The lab board runs QNN v2.28, so a binary built against a newer SDK
loads on Qualcomm's cloud device and fails on the hardware we actually ship on.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import qai_hub as hub

MATRIX = [
    ("w8a8",   hub.QuantizeDtype.INT8,  hub.QuantizeDtype.INT8),
    ("w8a16",  hub.QuantizeDtype.INT8,  hub.QuantizeDtype.INT16),
    ("w16a16", hub.QuantizeDtype.INT16, hub.QuantizeDtype.INT16),
    ("w4a16",  hub.QuantizeDtype.INT4,  hub.QuantizeDtype.INT16),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--device", default="QCS8550 (Proxy)")
    ap.add_argument("--qairt", default="2.28",
                    help="pin to the board's runtime; a context binary is not "
                         "portable across QAIRT versions")
    ap.add_argument("--input-name", default="images")
    ap.add_argument("--only", nargs="*", default=None,
                    help="subset of %s" % [m[0] for m in MATRIX])
    ap.add_argument("--out", default="results/quant_matrix.csv")
    args = ap.parse_args()

    device = hub.Device(args.device)
    calib = np.load(args.calib)
    print(f"[info] calibration {calib.shape}  {calib.nbytes / 1e6:.0f} MB")
    print(f"[info] model {args.model}")
    print(f"[info] device {device.name}, QAIRT pinned to {args.qairt}\n")

    print("[1/4] tai model va calibration len AI Hub ...", flush=True)
    src = hub.upload_model(args.model)
    data = hub.upload_dataset({args.input_name: [calib[i:i + 1] for i in
                                                 range(calib.shape[0])]})
    print(f"      model {src.model_id}   dataset {data.dataset_id}\n")

    todo = [m for m in MATRIX if not args.only or m[0] in args.only]

    # Submit every quantize job before waiting on any of them: they run
    # concurrently on the service, so serialising the waits would multiply the
    # wall time by the number of precisions for no reason.
    qjobs = {}
    print("[2/4] gui cac job quantize ...", flush=True)
    for name, w, a in todo:
        j = hub.submit_quantize_job(model=src, calibration_data=data,
                                    weights_dtype=w, activations_dtype=a,
                                    name=f"{args.tag}-{name}")
        qjobs[name] = j
        print(f"      {name:7s} {j.job_id}")

    print("\n[3/4] cho quantize, roi compile + profile ...", flush=True)
    rows = []
    for name, w, a in todo:
        r = {"tag": args.tag, "precision": name,
             "weights": w.name, "activations": a.name,
             "device": device.name, "qairt": args.qairt,
             "quantize_job": qjobs[name].job_id}
        try:
            qm = qjobs[name].get_target_model()
            if qm is None:
                r["status"] = f"quantize that bai: {qjobs[name].get_status()}"
                rows.append(r); print(f"  {name:7s} {r['status']}"); continue

            cj = hub.submit_compile_job(
                model=qm, device=device,
                options=f"--target_runtime qnn_context_binary "
                        f"--qairt_version {args.qairt}",
                name=f"{args.tag}-{name}-compile")
            r["compile_job"] = cj.job_id
            tm = cj.get_target_model()
            if tm is None:
                r["status"] = f"compile that bai: {cj.get_status()}"
                rows.append(r); print(f"  {name:7s} {r['status']}"); continue

            pj = hub.submit_profile_job(model=tm, device=device,
                                        name=f"{args.tag}-{name}-profile")
            r["profile_job"] = pj.job_id
            prof = pj.download_profile()
            ex = prof["execution_summary"]
            layers = prof["execution_detail"]
            r["inference_us"] = ex.get("estimated_inference_time")
            r["peak_mem_mb"] = round(
                ex.get("estimated_inference_peak_memory", 0) / 1e6, 1)
            units = [l.get("compute_unit") for l in layers]
            r["ops_total"] = len(units)
            r["ops_npu"] = sum(1 for u in units if u == "NPU")
            r["ops_fallback"] = sum(1 for u in units if u != "NPU")
            r["status"] = "ok"
            print(f"  {name:7s} {r['inference_us']:>7} us  "
                  f"NPU {r['ops_npu']}/{r['ops_total']}  "
                  f"fallback {r['ops_fallback']}", flush=True)
        except Exception as e:
            r["status"] = f"loi: {type(e).__name__}: {e}"
            print(f"  {name:7s} {r['status']}")
        rows.append(r)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader(); wr.writerows(rows)
    print(f"\n[4/4] [ok] {args.out}")
    print("      Buoc con lai: tai .bin ve, kiem output co sap khong, "
          "roi scp sang board.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
