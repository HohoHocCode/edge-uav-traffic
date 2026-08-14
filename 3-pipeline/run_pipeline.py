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
from mjpeg import MjpegServer  # noqa: E402
from tracker import ByteTrack, detections_as_tracks  # noqa: E402


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def is_gst_pipeline(src) -> bool:
    """A GStreamer launch string, as opposed to a path, URL or camera index.

    Detected by the pipeline separator plus a known source element. A file
    path will not contain ' ! ', so the test is safe.
    """
    return isinstance(src, str) and " ! " in src and any(
        e in src for e in ("qtiqmmfsrc", "v4l2src", "filesrc", "rtspsrc",
                           "videotestsrc", "nvarguscamerasrc")
    )


def open_capture(src):
    """Open a capture from a camera index, a path/URL, or a GStreamer pipeline."""
    if is_gst_pipeline(src):
        cap = cv2.VideoCapture(src, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            info = cv2.getBuildInformation()
            if "GStreamer:                   NO" in info:
                print("[fatal] this OpenCV build has NO GStreamer support, so a "
                      "GStreamer\n        pipeline can never open. The pip wheels "
                      "are built without it.\n        Fix: apt install python3-opencv, "
                      "or use a USB camera index.", file=sys.stderr)
        return cap
    return cv2.VideoCapture(src)


def resolve_source(src: str):
    """A bare integer means a camera index; anything else is a path, URL or pipeline."""
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
    p.add_argument("--loop", action="store_true",
                   help="when a file/URL source ends, reopen it and keep going "
                        "(seamless continuous demo; servers and counters stay "
                        "up). Ignored for a live camera index.")
    p.add_argument("--post-url", default=None)
    p.add_argument("--save-video", default=None)
    p.add_argument("--headless", action="store_true", help="never open a window")
    p.add_argument("--no-telemetry", action="store_true")
    p.add_argument("--degrade", default=None,
                   help="apply a degradation condition to every frame "
                        "(see 2-augment/degradations.py CONDITION_IDS)")
    p.add_argument("--session-id", default=None)
    p.add_argument("--mjpeg-port", type=int, default=0,
                   help="serve annotated frames at http://<board-ip>:PORT/ . "
                        "The only practical way to watch a headless board; "
                        "0 disables it")
    p.add_argument("--mjpeg-quality", type=int, default=75)
    p.add_argument("--mjpeg-width", type=int, default=0,
                   help="downscale the streamed frame only (0 = as rendered). "
                        "Useful over a weak uplink; does not affect detection")
    p.add_argument("--speed", type=float, default=1.0,
                   help="advance the source faster than one frame per step, by "
                        "dropping the frames in between (1.5 = play 1.5x). It "
                        "cannot make the pipeline faster -- the pipeline is "
                        "already going flat out -- it makes the clip cover more "
                        "ground per second of wall clock. Ignored for a live "
                        "camera, where there is nothing to skip ahead to.")
    p.add_argument("--threads", type=int, default=0,
                   help="ORT intra-op threads; 0 leaves the ORT default. On a "
                        "big.LITTLE SoC the default (one per core) is a trap: "
                        "the slow cores hold the fast ones back. Measured on "
                        "QCS8550, v26n at 640: 1 thread 113 ms, 3 threads "
                        "65 ms, 6 threads 81 ms.")
    return p.parse_args(argv)


#: What to draw on the frame for each dashboard task. The board renders one
#: video, so the view has to follow whichever task the operator is looking at;
#: drawing all of it at once is what made the three tabs look identical.
#:
#:   2 detection  boxes + class + confidence. No ids: identity is not this
#:                task's subject and a "#41" beside every box is noise.
#:   4 tracking   boxes + class + id + motion trail. No confidence, no ROI.
#:   5 counting   boxes + the counting line captioned with its own tally.
#:                Ids and trails off so the line is the loudest thing present.
OVERLAY_VIEWS = {
    "2": {"id": False, "trail": False, "conf": True,  "rois": True,  "counts": False},
    "4": {"id": True,  "trail": True,  "conf": False, "rois": False, "counts": False},
    "5": {"id": False, "trail": False, "conf": False, "rois": False, "counts": True},
}


def frame_stats(dets, tracks, tracker, prev_issued: int) -> tuple[dict, int]:
    """Per-frame numbers the analytics report cannot produce, by task.

    The dashboard shows three views -- detection, tracking, counting -- and
    they can only be as different from each other as the telemetry behind
    them. ``vehicle_count`` and ``n_tracks`` describe all three equally badly,
    so each task gets the quantity it is actually judged on.

    task 2  conf_mean / conf_min say how sure the detector was, which
            ``n_dets`` alone cannot; a frame with 9 detections at 0.26 is not
            the same result as 9 at 0.81.

    task 4  identity churn. ``n_tracks`` is blind to the failure that matters
            most in tracking: a tracker that discards an object and reissues a
            fresh id for it every few frames reports a perfectly steady track
            count while producing useless trajectories. ``n_new`` (ids minted
            this frame) against the number of objects present is the cheap
            proxy for that, and needs no ground truth.
    """
    conf = getattr(dets, "conf", None)
    n_dets = int(len(dets.xyxy))
    stats = {
        "n_dets": n_dets,
        "conf_mean": round(float(conf.mean()), 4) if n_dets else 0.0,
        "conf_min": round(float(conf.min()), 4) if n_dets else 0.0,
    }

    if tracker is None:
        # Detections passed straight through as pseudo-tracks: there are no
        # identities to churn, so reporting zeros here would look like a
        # perfectly stable tracker rather than the absence of one.
        stats.update({"n_new": 0, "n_lost": 0, "n_tentative": 0,
                      "ids_issued": 0, "track_age_mean": 0.0})
        return stats, prev_issued

    from tracker import Track
    issued = int(Track._next_id) - 1
    ages = [t.age for t in tracks]
    stats.update({
        "n_new": max(0, issued - prev_issued),
        "n_lost": sum(1 for t in tracker.tracks if t.state == "lost"),
        "n_tentative": sum(1 for t in tracker.tracks if t.state == "tentative"),
        "ids_issued": issued,
        "track_age_mean": round(sum(ages) / len(ages), 2) if ages else 0.0,
    })
    return stats, issued


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
              f"        weights are not in git (10 MB each). See which ones\n"
              f"        are on this machine with: run_demo.py --list",
              file=sys.stderr)
        return 2

    # ---- detector -----------------------------------------------------
    session = create_session(model_path, backend=backend,
                             intra_threads=args.threads or None)
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
        try:
            import degradations as D
        except ImportError:
            # 2-augment/ lives on the research branch: it is what the
            # robustness experiments need, not what the demo runs. Say so
            # rather than surfacing a bare ImportError for a missing folder.
            print("[fatal] --degrade can 2-augment/degradations.py, khong co "
                  "trong cay demo.\n"
                  "        lay bang: git checkout research -- 2-augment",
                  file=sys.stderr)
            return 2
        if args.degrade not in D.CONDITION_IDS:
            print(f"[fatal] unknown condition {args.degrade!r}; "
                  f"known: {D.CONDITION_IDS}", file=sys.stderr)
            return 2
        degrade_fn = lambda img, i: D.apply_condition(img, args.degrade, seed=i)  # noqa: E731

    # ---- capture ------------------------------------------------------
    source = resolve_source(args.source if args.source is not None
                            else cfg["capture"]["source"])
    cap = open_capture(source)
    if not cap.isOpened():
        print(f"[fatal] cannot open source {source!r}", file=sys.stderr)
        print("        run 0-setup/probe_camera.py to find one that works",
              file=sys.stderr)
        return 2
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

    resize_to = cfg["capture"].get("resize_input")
    if resize_to:
        src_w, src_h = int(resize_to[0]), int(resize_to[1])
        print(f"[info] resizing capture to {src_w}x{src_h}")
    target_fps = float(cfg["capture"].get("target_fps") or 0)
    min_period = 1.0 / target_fps if target_fps > 0 else 0.0
    if min_period:
        print(f"[info] capping processing at {target_fps:.1f} fps")
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

    # ---- writer & frame streamer -------------------------------------
    writer = None
    save_path = args.save_video or cfg["overlay"].get("save_video")
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        writer = cv2.VideoWriter(
            save_path, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (src_w, src_h)
        )

    # TẠO THƯ MỤC FRAMES ĐỂ LƯU ẢNH REALTIME CHO DASHBOARD
    frames_dir = os.path.join(ROOT, "results", "frames")
    os.makedirs(frames_dir, exist_ok=True)
    latest_frame_path = os.path.join(frames_dir, "latest_frame.jpg")

    # ---- live view ----------------------------------------------------
    streamer = None
    if args.mjpeg_port:
        try:
            streamer = MjpegServer(
                port=args.mjpeg_port,
                quality=args.mjpeg_quality,
                max_width=args.mjpeg_width,
            ).start()
            print(f"[info] live view on http://<board-ip>:{args.mjpeg_port}/")
        except OSError as exc:
            print(
                f"[warn] could not start the MJPEG server on port "
                f"{args.mjpeg_port}: {exc}"
            )

    # opencv-python-headless has no HighGUI at all, and a board over SSH has no
    # display. Probe once rather than crashing on the first frame.
    show_window = not args.headless
    if show_window:
        try:
            cv2.namedWindow("SkySentry", cv2.WINDOW_NORMAL)
        except cv2.error:
            print(
                "[warn] no GUI available (headless OpenCV or no display); "
                "continuing without a window"
            )
            show_window = False
    ocfg = cfg["overlay"]
    frame_id = 0
    t_start = time.perf_counter()
    totals: list[float] = []
    last_level = "normal"
    prev_issued = 0                 # ids minted so far, for the churn metric

    # --speed, as a fractional debt of frames to throw away after each one we
    # keep. A live camera index is excluded: there is no "ahead" to skip to,
    # and dropping its frames would only discard the present.
    speed_skip = max(0.0, args.speed - 1.0) if not isinstance(source, int) else 0.0
    skip_debt = 0.0
    if speed_skip:
        print(f"[info] speed x{args.speed:g}: bo {speed_skip:.2f} khung sau moi "
              f"khung xu ly (tracking va dem se doi theo)")

    try:
        while True:
            loop_start = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                # A file/URL that ended: reopen and keep the demo running,
                # without tearing down the MJPEG server, telemetry sink or
                # tracker. A live camera index cannot be "reopened" to rewind,
                # so looping is only meaningful for a finite source.
                if args.loop and not isinstance(source, int):
                    cap.release()
                    cap = open_capture(source)
                    if cap.isOpened():
                        continue
                    print("[warn] --loop could not reopen the source; stopping",
                          file=sys.stderr)
                break

            # Move the read head past the frames --speed says to skip. grab()
            # advances without converting to BGR, so a skipped frame costs a
            # decode but not a colour conversion or an allocation.
            if speed_skip:
                skip_debt += speed_skip
                while skip_debt >= 1.0:
                    if not cap.grab():
                        break
                    skip_debt -= 1.0

            if resize_to:
                frame = cv2.resize(frame, (src_w, src_h), interpolation=cv2.INTER_AREA)
            if degrade_fn is not None:
                frame = degrade_fn(frame, frame_id)

            dets = det(frame)

            t0 = time.perf_counter()
            if tracker is not None:
                tracks = tracker.update(dets.xyxy, dets.conf, dets.cls)
            else:
                tracks = detections_as_tracks(dets.xyxy, dets.conf, dets.cls)
            track_ms = (time.perf_counter() - t0) * 1000.0

            ts_ms = (time.perf_counter() - t_start) * 1000.0
            report = ana.update(tracks, frame.shape[:2], frame_id, ts_ms)

            timings = {
                "pre_ms": dets.pre_ms,
                "infer_ms": dets.infer_ms,
                "post_ms": dets.post_ms,
                "track_ms": track_ms,
                "total_ms": dets.total_ms + track_ms,
            }
            totals.append(timings["total_ms"])

            if sink is not None:
                stats, prev_issued = frame_stats(dets, tracks, tracker, prev_issued)
                sink.write(report, timings, stats)
                if report.congestion_level != last_level:
                    sink.event(
                        frame_id,
                        "congestion_change",
                        report.congestion_level,
                        {
                            "from": last_level,
                            "to": report.congestion_level,
                            "vehicles": report.vehicle_count,
                        },
                    )
                    last_level = report.congestion_level

            if ocfg.get("enabled", True) and (
                writer is not None or show_window or streamer is not None
            ):
                # The dashboard's selected task rides back on the /ingest
                # response, so this costs no extra connection and no polling.
                # Offline (no sink, or no reply yet) it falls back to tracking.
                v = OVERLAY_VIEWS.get(
                    getattr(sink, "view", "4"), OVERLAY_VIEWS["4"])
                viz.draw_regions(
                    frame, ana.rois, ana.lines,
                    show_rois=v["rois"],
                    counts=report.line_crossings if v["counts"] else None,
                )
                viz.draw_tracks(
                    frame, tracks, class_names,
                    show_id=v["id"] and ocfg.get("show_track_id", True),
                    show_trail=v["trail"],
                    show_conf=v["conf"],
                )
                viz.draw_hud(
                    frame, report, timings, session.backend_name, condition=args.degrade
                )

                if writer is not None:
                    writer.write(frame)

                # GHI ẢNH RA FILE ĐỂ DASHBOARD LẤY REALTIME
                # 1. Ghi vào file tạm (tmp) để tránh bị đọc phải file đang ghi dở
                temp_frame_path = latest_frame_path.replace(".jpg", "_tmp.jpg")
                write_ok = cv2.imwrite(temp_frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

                # 2. Nếu ghi xong hoàn toàn (file tạm đã đóng), mới đổi tên thành file chính
                if write_ok:
                    try:
                        os.replace(temp_frame_path, latest_frame_path)  # os.replace là thao tác nguyên tử
                    except Exception:
                        pass

                if streamer is not None:
                    streamer.publish(frame)
                if show_window:
                    cv2.imshow("SkySentry", frame)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        break

            frame_id += 1
            if args.max_frames and frame_id >= args.max_frames:
                break
            if min_period:
                slack = min_period - (time.perf_counter() - loop_start)
                if slack > 0:
                    time.sleep(slack)
    except KeyboardInterrupt:
        print("\n[info] interrupted")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if show_window:
            cv2.destroyAllWindows()
        if streamer is not None:
            print(f"[info] mjpeg: {streamer.summary()}")
            streamer.stop()
        if sink is not None:
            sink.close()
            print(f"[info] telemetry: {sink.summary()}")

    if totals:
        a = np.asarray(totals)
        print(
            f"[done] {frame_id} frames | total_ms avg {a.mean():.2f} "
            f"p50 {np.percentile(a, 50):.2f} p95 {np.percentile(a, 95):.2f} "
            f"| {1000.0 / a.mean():.1f} FPS avg"
        )
        print(f"[done] crossings: {ana.total_crossings}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
