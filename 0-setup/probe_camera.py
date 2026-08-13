#!/usr/bin/env python3
"""Find a camera capture path that actually works on this board.

Run on the device. Camera bring-up on Qualcomm Linux fails for several
unrelated reasons that all look identical from Python -- ``VideoCapture`` just
returns False -- so this walks the possibilities in order and reports which
one produced a frame, with the exact source string to reuse.

    python3 0-setup/probe_camera.py
    python3 0-setup/probe_camera.py --save-frame /tmp/cam.jpg

The three paths, and why they differ
------------------------------------
**USB UVC** — a webcam in a USB-A port appears as ``/dev/videoN`` and works
with plain ``cv2.VideoCapture(N)``. Simplest, and the one to try first.

**CSI / MIPI** — a ribbon-cable camera does *not* appear as a usable V4L2
device on Qualcomm Linux. It goes through the camera server and is reached
with GStreamer (``qtiqmmfsrc``). No amount of ``VideoCapture(0)`` will find it.

**OpenCV without GStreamer** — the pip wheels (`opencv-python`,
`opencv-python-headless`) are built *without* GStreamer support. If the board
has a CSI camera and pip-installed OpenCV, the GStreamer path cannot work at
all until the distro's OpenCV is used instead. This script checks the build
flags rather than letting that fail as a mysterious False.
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
import time


def sh(cmd: list[str], timeout: int = 8) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except (OSError, subprocess.SubprocessError) as exc:
        return 1, f"{type(exc).__name__}: {exc}"


def section(t: str) -> None:
    print(f"\n=== {t} ===")


# --------------------------------------------------------------------------- #
def check_opencv() -> tuple[object, bool]:
    section("OpenCV")
    try:
        import cv2
    except ImportError:
        print("  cv2 not installed — pip3 install opencv-python-headless")
        return None, False

    print(f"  version: {cv2.__version__}")
    info = cv2.getBuildInformation()
    gst = "GStreamer:" in info and "GStreamer:                   NO" not in info
    v4l = "V4L/V4L2" in info

    line = next((l.strip() for l in info.splitlines() if "GStreamer" in l), "")
    print(f"  GStreamer support : {'YES' if gst else 'NO'}   {line}")
    print(f"  V4L2 support      : {'YES' if v4l else 'NO'}")
    if not gst:
        print("  note: pip OpenCV wheels ship without GStreamer. USB cameras are")
        print("        unaffected; a CSI camera cannot be opened at all this way.")
        print("        Fix: apt install python3-opencv  (distro build has it)")
    return cv2, gst


def list_v4l2() -> list[str]:
    section("V4L2 devices")
    nodes = sorted(glob.glob("/dev/video*"))
    if not nodes:
        print("  none — no /dev/video* nodes")
        print("  a USB webcam should create one within a second of being plugged in")
        return []
    for n in nodes:
        rc, out = sh(["v4l2-ctl", "--device", n, "--info"])
        name = ""
        for line in out.splitlines():
            if "Card type" in line:
                name = line.split(":", 1)[1].strip()
                break
        print(f"  {n}  {name or '(v4l2-ctl unavailable)'}")
    return nodes


def list_formats(node: str) -> None:
    if not shutil.which("v4l2-ctl"):
        return
    rc, out = sh(["v4l2-ctl", "--device", node, "--list-formats-ext"])
    if rc != 0:
        return
    keep = [l for l in out.splitlines()
            if "Size:" in l or "]:" in l or "Pixel Format" in l]
    for l in keep[:14]:
        print(f"      {l.strip()}")


def check_gst() -> bool:
    section("GStreamer")
    if not shutil.which("gst-inspect-1.0"):
        print("  gst-inspect-1.0 not found — GStreamer not installed")
        return False
    rc, out = sh(["gst-inspect-1.0", "--version"])
    print("  " + out.strip().splitlines()[0])

    for plugin in ("qtiqmmfsrc", "v4l2src", "videoconvert", "appsink", "jpegenc"):
        rc, _ = sh(["gst-inspect-1.0", plugin])
        mark = "ok  " if rc == 0 else "MISS"
        note = ""
        if plugin == "qtiqmmfsrc":
            note = "  <- Qualcomm CSI camera source" if rc == 0 else \
                   "  <- absent: no CSI path on this image"
        print(f"  {mark} {plugin}{note}")
    return True


# --------------------------------------------------------------------------- #
def try_open(cv2, source, label: str, api=None, warmup: int = 3) -> dict | None:
    """Open a source and actually pull frames — isOpened() alone lies."""
    try:
        cap = cv2.VideoCapture(source, api) if api is not None \
            else cv2.VideoCapture(source)
    except Exception as exc:
        print(f"  {label:52s} EXC {type(exc).__name__}")
        return None

    if not cap.isOpened():
        cap.release()
        print(f"  {label:52s} not opened")
        return None

    # Some drivers report opened then fail on read; some need a few frames
    # before the sensor settles. Both are why this reads more than once.
    ok = False
    frame = None
    for _ in range(warmup):
        ok, frame = cap.read()
        if ok and frame is not None:
            break
        time.sleep(0.15)

    if not ok or frame is None:
        cap.release()
        print(f"  {label:52s} opened but no frame")
        return None

    t0 = time.perf_counter()
    n = 0
    for _ in range(10):
        r, f = cap.read()
        if r:
            n += 1
    fps = n / max(time.perf_counter() - t0, 1e-9)
    h, w = frame.shape[:2]
    cap.release()
    print(f"  {label:52s} OK  {w}x{h}  ~{fps:.1f} fps")
    return {"source": source, "label": label, "w": w, "h": h,
            "fps": fps, "frame": frame}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-frame", default=None,
                    help="write the first good frame here, to check focus/exposure")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    print(f"probe_camera — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"uname: {' '.join(os.uname()) if hasattr(os, 'uname') else 'n/a'}")

    cv2, has_gst = check_opencv()
    if cv2 is None:
        return 2

    nodes = list_v4l2()
    for n in nodes[:2]:
        print(f"    formats for {n}:")
        list_formats(n)

    has_gstbin = check_gst()

    section("Trying capture paths")
    found: list[dict] = []

    # 1. plain V4L2 by index — the USB webcam case
    for n in nodes:
        idx = int("".join(ch for ch in os.path.basename(n) if ch.isdigit()) or -1)
        if idx < 0:
            continue
        r = try_open(cv2, idx, f"cv2.VideoCapture({idx})")
        if r:
            found.append(r)
            break

    # 2. V4L2 with an explicit MJPG request. Many UVC cameras only reach 30 fps
    #    at 720p in MJPG; in YUYV they silently drop to 5-10 fps.
    if nodes and not found:
        idx = 0
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
            ok, frame = cap.read()
            cap.release()
            if ok:
                print(f"  {'V4L2 + MJPG ' + str(idx):52s} OK")
                found.append({"source": idx, "label": f"v4l2 MJPG {idx}",
                              "w": frame.shape[1], "h": frame.shape[0],
                              "fps": 0.0, "frame": frame})

    # 3. GStreamer pipelines
    if has_gst and has_gstbin:
        pipes = [
            (f"qtiqmmfsrc camera=0 ! video/x-raw,format=NV12,"
             f"width={args.width},height={args.height},framerate=30/1 "
             f"! videoconvert ! video/x-raw,format=BGR ! appsink drop=1 sync=false",
             "GStreamer qtiqmmfsrc (CSI)"),
            (f"v4l2src device=/dev/video0 ! video/x-raw,"
             f"width={args.width},height={args.height} "
             f"! videoconvert ! video/x-raw,format=BGR ! appsink drop=1 sync=false",
             "GStreamer v4l2src (USB)"),
        ]
        for pipe, label in pipes:
            r = try_open(cv2, pipe, label, api=cv2.CAP_GSTREAMER)
            if r:
                found.append(r)
                break
    elif not has_gst:
        print("  (GStreamer paths skipped — this OpenCV build has no GStreamer)")

    # ---- verdict ------------------------------------------------------
    section("Verdict")
    if not found:
        print("  No working capture path.")
        print("  Checklist:")
        print("    - USB webcam plugged into a USB-A port? It should create /dev/video0")
        print("    - `ls /dev/video*` after plugging in, and `dmesg | tail`")
        print("    - CSI camera: needs GStreamer + qtiqmmfsrc, and an OpenCV built")
        print("      with GStreamer (apt install python3-opencv, not the pip wheel)")
        print("    - permissions: are you root, or in the `video` group?")
        return 1

    best = found[0]
    print(f"  Working source: {best['label']}")
    print(f"  Resolution    : {best['w']}x{best['h']}")
    if best["fps"]:
        print(f"  Capture rate  : ~{best['fps']:.1f} fps")

    if args.save_frame:
        cv2.imwrite(args.save_frame, best["frame"])
        print(f"  Saved a frame : {args.save_frame}")

    src = best["source"]
    src_arg = str(src) if isinstance(src, int) else f'"{src}"'
    print("\n  Run the pipeline with:")
    print(f"    python3 3-pipeline/run_pipeline.py \\")
    print(f"        --source {src_arg} \\")
    print(f"        --model models/yolov8n_visdrone_640.onnx \\")
    print(f"        --mjpeg-port 8090 --headless")
    print("\n  Then open http://<board-ip>:8090/ in a browser on your laptop.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
