#!/usr/bin/env python3
"""SkySentry — UAV traffic-order edge node.

    video/camera -> detect (NPU) -> track (CPU) -> analytics -> telemetry -> overlay

Run on the host for development and on the QCS8550 for the real thing; the only
difference is which backend ``create_session`` resolves to.

Examples
--------
    # host, video file, write an annotated mp4
    python 3-pipeline/run_pipeline.py --source data/clip.mp4 \
        --model models/yolov8n_visdrone.onnx --save-video results/demo.mp4

    # board, live camera, stream telemetry to the command post
    python3 3-pipeline/run_pipeline.py --source 0 --backend onnxruntime-qnn \
        --post-url http://192.168.1.50:8000/ingest --headless
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "2-augment"))

import cv2  # noqa: E402

import analytics as A  # noqa: E402
import viz  # noqa: E402
from detector import Yolov8Detector  # noqa: E402
from runtime import create_session  # noqa: E402
from telemetry import TelemetryConfig, TelemetrySink  # noqa: E402
from tracker import ByteTrack  # noqa: E402


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_source(src: str):
    """A bare integer means a camera index; anything else is a path or URL."""
    try:
        return int(src)
    except (TypeError, ValueError):
        return src


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="SkySentry edge pipeline")
    p.add_argument("--config", default=os.path.join(ROOT, "configs", "pipeline.yaml"))
    p.add_argument("--classes", default=os.path.join(ROOT, "configs", "visdrone.yaml"))
    p.add_argument("--model", default=None, help="override model path")
    p.add_argument("--source", default=None, help="camera index, video path, or rtsp url")
    p.add_argument("--backend", default=None,
                   choices=["auto", "onnxruntime-cpu", "onnxruntime-qnn"])
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--conf", type=float, default=None)
    p.add_argument("--max-frames", type=int, default=0, help="0 = until the source ends")
    p.add_argument("--post-url", default=None)
    p.add_argument("--save-video", default=None)
    p.add_argument("--headless", action="store_true", help="never open a window")
    p.add_argument("--no-telemetry", action="store_true")
    p.add_argument("--degrade", default=None,
                   help="apply a degradation condition to every frame "
                        "(see 2-augment/degradations.py CONDITION_IDS)")
    p.add_argument("--session-id", default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_yaml(args.config)
    cls_cfg = load_yaml(args.classes)
    class_names = {int(k): v for k, v in cls_cfg["names"].items()}

    mcfg = cfg["model"]
    model_path = args.model or os.path.join(ROOT, mcfg["weights"])
    imgsz = args.imgsz or int(mcfg["imgsz"])
    backend = args.backend or mcfg.get("backend", "auto")
    conf_thres = args.conf if args.conf is not None else float(mcfg["conf_thres"])

    if not os.path.exists(model_path):
        print(f"[fatal] model not found: {model_path}\n"
              f"        run 1-model/export_onnx.py first.", file=sys.stderr)
        return 2

    # ---- detector -----------------------------------------------------
    session = create_session(model_path, backend=backend)
    det = Yolov8Detector(
        session, imgsz=imgsz, conf_thres=conf_thres,
        iou_thres=float(mcfg["iou_thres"]), max_det=int(mcfg["max_det"]),
        pad_value=int(mcfg["pad_value"]),
    )
    det.warmup(n=5)
    print(f"[info] backend={session.backend_name} "
          f"providers={session.info.providers_active} imgsz={imgsz}")

    # ---- tracker ------------------------------------------------------
    tcfg = cfg["tracker"]
    tracker = ByteTrack(
        track_high_thresh=float(tcfg["track_high_thresh"]),
        track_low_thresh=float(tcfg["track_low_thresh"]),
        new_track_thresh=float(tcfg["new_track_thresh"]),
        match_iou=float(tcfg["match_iou"]),
        match_iou_low=float(tcfg["match_iou_low"]),
        track_buffer=int(tcfg["track_buffer"]),
        min_box_area=float(tcfg["min_box_area"]),
        min_hits=int(tcfg["min_hits"]),
    ) if tcfg.get("enabled", True) else None

    # ---- analytics ----------------------------------------------------
    ana = A.build_from_config(cfg["analytics"], class_names, cls_cfg.get("groups", {}))

    # ---- degradation (robustness demo) --------------------------------
    degrade_fn = None
    if args.degrade:
        import degradations as D
        if args.degrade not in D.CONDITION_IDS:
            print(f"[fatal] unknown condition {args.degrade!r}; "
                  f"known: {D.CONDITION_IDS}", file=sys.stderr)
            return 2
        degrade_fn = lambda img, i: D.apply_condition(img, args.degrade, seed=i)  # noqa: E731

    # ---- capture ------------------------------------------------------
    source = resolve_source(args.source if args.source is not None
                            else cfg["capture"]["source"])
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[fatal] cannot open source {source!r}", file=sys.stderr)
        return 2
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    print(f"[info] source={source!r} {src_w}x{src_h} @ {src_fps:.1f} fps")

    # ---- telemetry ----------------------------------------------------
    sink = None
    if not args.no_telemetry and cfg["telemetry"].get("enabled", True):
        tel = cfg["telemetry"]
        sink = TelemetrySink(
            TelemetryConfig(
                local_db=os.path.join(ROOT, tel["local_db"]),
                csv_path=os.path.join(ROOT, tel["csv_path"]),
                post_url=args.post_url if args.post_url is not None else tel.get("post_url", ""),
                post_interval_s=float(tel.get("post_interval_s", 2.0)),
                session_id=args.session_id or "",
            ),
            meta={
                "backend": session.backend_name, "model": os.path.basename(model_path),
                "imgsz": imgsz, "device": os.uname().nodename
                if hasattr(os, "uname") else os.environ.get("COMPUTERNAME", "host"),
                "condition": args.degrade or "clean",
            },
        )

    # ---- writer -------------------------------------------------------
    writer = None
    save_path = args.save_video or cfg["overlay"].get("save_video")
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 src_fps, (src_w, src_h))

    # opencv-python-headless has no HighGUI at all, and a board over SSH has no
    # display. Probe once rather than crashing on the first frame.
    show_window = not args.headless
    if show_window:
        try:
            cv2.namedWindow("SkySentry", cv2.WINDOW_NORMAL)
        except cv2.error:
            print("[warn] no GUI available (headless OpenCV or no display); "
                  "continuing without a window")
            show_window = False
    ocfg = cfg["overlay"]
    frame_id = 0
    t_start = time.perf_counter()
    totals: list[float] = []
    last_level = "normal"

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if degrade_fn is not None:
                frame = degrade_fn(frame, frame_id)

            dets = det(frame)

            t0 = time.perf_counter()
            if tracker is not None:
                tracks = tracker.update(dets.xyxy, dets.conf, dets.cls)
            else:  # detection-only mode still needs a track-like object
                tracks = []
            track_ms = (time.perf_counter() - t0) * 1000.0

            ts_ms = (time.perf_counter() - t_start) * 1000.0
            report = ana.update(tracks, frame.shape[:2], frame_id, ts_ms)

            timings = {
                "pre_ms": dets.pre_ms, "infer_ms": dets.infer_ms,
                "post_ms": dets.post_ms, "track_ms": track_ms,
                "total_ms": dets.total_ms + track_ms,
            }
            totals.append(timings["total_ms"])

            if sink is not None:
                sink.write(report, timings)
                if report.congestion_level != last_level:
                    sink.event(frame_id, "congestion_change", report.congestion_level,
                               {"from": last_level, "to": report.congestion_level,
                                "vehicles": report.vehicle_count})
                    last_level = report.congestion_level

            if ocfg.get("enabled", True) and (writer is not None or show_window):
                viz.draw_regions(frame, ana.rois, ana.lines)
                viz.draw_tracks(frame, tracks, class_names,
                                show_id=ocfg.get("show_track_id", True))
                viz.draw_hud(frame, report, timings, session.backend_name,
                             condition=args.degrade)
                if writer is not None:
                    writer.write(frame)
                if show_window:
                    cv2.imshow("SkySentry", frame)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        break

            frame_id += 1
            if args.max_frames and frame_id >= args.max_frames:
                break
    except KeyboardInterrupt:
        print("\n[info] interrupted")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if show_window:
            cv2.destroyAllWindows()
        if sink is not None:
            # close() drains the uploader queue, so the summary is only
            # accurate afterwards; printing it first under-reports `posted`.
            sink.close()
            print(f"[info] telemetry: {sink.summary()}")

    if totals:
        a = np.asarray(totals)
        print(f"[done] {frame_id} frames | total_ms avg {a.mean():.2f} "
              f"p50 {np.percentile(a, 50):.2f} p95 {np.percentile(a, 95):.2f} "
              f"| {1000.0 / a.mean():.1f} FPS avg")
        print(f"[done] crossings: {ana.total_crossings}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
