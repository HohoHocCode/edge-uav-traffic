#!/usr/bin/env python3
"""Side-by-side demo video: the same footage, clean vs degraded.

This is the demo artefact that carries the finding. A robustness table is
convincing to a reviewer; a split screen where the right-hand pane visibly
loses half its boxes the moment rain starts is convincing to everyone else.

Both panes run the *same* model, the same thresholds and the same tracker.
The only difference is the degradation applied to the right-hand input, so
whatever the viewer sees is attributable to the condition and nothing else.

    python scripts/make_comparison_video.py --source assets/demo_clip.mp4 \
        --model models/yolov8n_visdrone_640.onnx --condition rain_heavy \
        --out results/compare_rain_heavy.mp4
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "3-pipeline"))
sys.path.insert(0, os.path.join(ROOT, "2-augment"))

import cv2  # noqa: E402

import degradations as D  # noqa: E402
import viz  # noqa: E402
import yaml  # noqa: E402
from detector import Yolov8Detector  # noqa: E402
from runtime import create_session  # noqa: E402
from tracker import ByteTrack  # noqa: E402


def banner(frame: np.ndarray, text: str, sub: str, colour) -> None:
    h, w = frame.shape[:2]
    bar = frame[0:52, :].copy()
    cv2.rectangle(bar, (0, 0), (w, 52), (18, 18, 18), -1)
    frame[0:52, :] = cv2.addWeighted(bar, 0.72, frame[0:52, :], 0.28, 0)
    cv2.putText(frame, text, (14, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                colour, 2, cv2.LINE_AA)
    cv2.putText(frame, sub, (14, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                (215, 215, 215), 1, cv2.LINE_AA)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=os.path.join(ROOT, "assets", "demo_clip.mp4"))
    ap.add_argument("--model", default=os.path.join(ROOT, "models",
                                                    "yolov8n_visdrone_640.onnx"))
    ap.add_argument("--classes", default=os.path.join(ROOT, "configs", "visdrone.yaml"))
    ap.add_argument("--condition", default="rain_heavy")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "comparison.mp4"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--pane-width", type=int, default=960)
    ap.add_argument("--csv", default=None,
                    help="per-frame record; defaults to <out>.csv. This is the "
                         "evidence behind any deployment-threshold number "
                         "quoted from this run")
    args = ap.parse_args()

    if args.condition not in D.CONDITION_IDS:
        print(f"[fatal] unknown condition {args.condition!r}; "
              f"known: {D.CONDITION_IDS}", file=sys.stderr)
        return 2

    with open(args.classes, "r", encoding="utf-8") as f:
        class_names = {int(k): v for k, v in yaml.safe_load(f)["names"].items()}

    # Two independent trackers: sharing one would let the clean pane's tracks
    # prop up the degraded pane and hide exactly the effect we are showing.
    session = create_session(args.model, backend="auto")
    det = Yolov8Detector(session, imgsz=640, conf_thres=args.conf)
    det.warmup(n=5)
    trk_a, trk_b = ByteTrack(), ByteTrack()

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"[fatal] cannot open {args.source}", file=sys.stderr)
        return 2
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    pw = args.pane_width
    ph = None
    writer = None
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    n = 0
    det_a_hist: list[int] = []
    det_b_hist: list[int] = []
    post_a_hist: list[float] = []
    post_b_hist: list[float] = []
    t0 = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        left = frame.copy()
        right = D.apply_condition(frame, args.condition, seed=n)

        da = det(left)
        pa = da.post_ms
        ta = trk_a.update(da.xyxy, da.conf, da.cls)

        db = det(right)
        pb = db.post_ms
        tb = trk_b.update(db.xyxy, db.conf, db.cls)

        det_a_hist.append(len(da))
        det_b_hist.append(len(db))
        post_a_hist.append(pa)
        post_b_hist.append(pb)

        viz.draw_tracks(left, ta, class_names, show_id=False, show_trail=False)
        viz.draw_tracks(right, tb, class_names, show_id=False, show_trail=False)

        if ph is None:
            ph = int(round(frame.shape[0] * pw / frame.shape[1]))
        left = cv2.resize(left, (pw, ph))
        right = cv2.resize(right, (pw, ph))

        banner(left, "CLEAN", f"{len(da)} detections   |   NMS {pa:.1f} ms",
               (150, 240, 150))
        banner(right, args.condition.upper().replace("_", " "),
               f"{len(db)} detections   |   NMS {pb:.1f} ms", (140, 170, 255))

        canvas = np.hstack([left, np.full((ph, 4, 3), 40, np.uint8), right])

        # Footer: running loss, which is the number the report quotes.
        foot = np.full((58, canvas.shape[1], 3), 22, np.uint8)
        ma, mb = float(np.mean(det_a_hist)), float(np.mean(det_b_hist))
        keep = 100.0 * mb / ma if ma > 0 else float("nan")
        dpost = 100.0 * (float(np.mean(post_b_hist)) / max(float(np.mean(post_a_hist)), 1e-9) - 1.0)
        cv2.putText(foot, f"detections retained: {keep:5.1f}%",
                    (16, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                    (120, 220, 120) if keep > 70 else (110, 130, 250), 2, cv2.LINE_AA)
        cv2.putText(foot, f"NMS cost on CPU: {dpost:+5.1f}%",
                    (16, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (215, 215, 215), 1,
                    cv2.LINE_AA)
        cv2.putText(foot, "same model - same thresholds - same tracker settings",
                    (canvas.shape[1] - 470, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                    (150, 150, 150), 1, cv2.LINE_AA)
        canvas = np.vstack([canvas, foot])

        if writer is None:
            writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps, (canvas.shape[1], canvas.shape[0]))
            if not writer.isOpened():
                print("[fatal] cannot open video writer", file=sys.stderr)
                return 2
        writer.write(canvas)

        n += 1
        if args.max_frames and n >= args.max_frames:
            break
        if n % 60 == 0:
            print(f"  {n} frames | clean {ma:.1f} det | "
                  f"{args.condition} {mb:.1f} det ({keep:.0f}%)", flush=True)

    cap.release()
    if writer is not None:
        writer.release()

    # Every other number in this repository ships with the CSV it came from;
    # the deployment-threshold comparison must too.
    csv_path = args.csv or (os.path.splitext(args.out)[0] + ".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["frame_id", "condition", "conf_thres", "model",
                       "n_det_clean", "n_det_degraded",
                       "post_ms_clean", "post_ms_degraded"])
        for i in range(len(det_a_hist)):
            wcsv.writerow([i, args.condition, args.conf,
                           os.path.basename(args.model),
                           det_a_hist[i], det_b_hist[i],
                           round(post_a_hist[i], 4), round(post_b_hist[i], 4)])
    print(f"[ok] {csv_path}")

    ma, mb = float(np.mean(det_a_hist)), float(np.mean(det_b_hist))
    print(f"[ok] {args.out}  {n} frames, {time.perf_counter() - t0:.0f}s")
    print(f"[ok] clean {ma:.1f} det/frame | {args.condition} {mb:.1f} det/frame "
          f"({100.0 * mb / max(ma, 1e-9):.1f}% retained)")
    print(f"[ok] NMS: clean {np.mean(post_a_hist):.1f} ms -> "
          f"{args.condition} {np.mean(post_b_hist):.1f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
