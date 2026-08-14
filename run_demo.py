#!/usr/bin/env python3
"""One entry point for the two things the board is ever asked to do.

    ssh -p 2222 qrobot@192.168.0.103
    cd ~/edge-uav-traffic
    .venv-device/bin/python run_demo.py --version sample
    .venv-device/bin/python run_demo.py --version realtime

``sample``    replays ``video/uav0000076_00241_s_30fps.mp4`` and pushes both the
              annotated stream and the per-frame telemetry to the command post.
              This is the demo: a known clip, reproducible, no camera to aim.

``realtime``  the USB webcam, live. Same detector, same tracker, same overlay.

Everything else here is one thing: making the detector land on the cores that
can actually run it.

Why this file exists at all
---------------------------
``3-pipeline/run_pipeline.py`` already does the work, and this does not wrap it
to hide it -- it wraps it to pin down the two settings that are invisible from
the command line and cost more than any other knob on this board.

Measured on QCS8550, YOLO26n at 640x640, ONNX Runtime CPU:

    threads   all 6 cores    pinned to cpu3,4,5    pinned to cpu0,1,2
    1            113.0 ms            114.2 ms              879.8 ms
    2             82.8 ms             82.1 ms              443.9 ms
    3             64.9 ms             65.3 ms              331.9 ms
    4             74.1 ms            122.6 ms
    6             80.6 ms            223.5 ms

Two facts hide in that table. Three threads beat six -- the default is one
thread per core, and on big.LITTLE the fast cores then wait on the slow ones.
And cpu0,1,2 are not merely slower, they are 5x slower, so a thread that
migrates there stalls the whole forward pass. The full pipeline measured
129 ms per inference against the 65 ms this benchmark shows, and the gap is
exactly that: under load the scheduler moves threads onto the small cores.

Capping the thread count is therefore the whole win. Pinning is *not*: it was
measured too, and it makes things worse, because ``sched_setaffinity`` binds
the entire process -- decode, tracking and JPEG encoding then compete with the
inference threads for the same three cores instead of using the idle ones.

    full pipeline, 60 frames of the sample clip

        ORT default (6 threads)          5.3 fps    p95 231.9 ms
        3 threads, pinned to cpu3,4,5    7.2 fps    p95 259.9 ms
        3 threads, unpinned              7.9 fps    p95 143.0 ms

Unpinned is both faster and far steadier, and the steadiness is what the
project's own real-time criterion is written against. So the default here is
three threads and no affinity mask; ``--cores`` remains for experiments.

Reaching the command post
-------------------------
The board has no network of its own -- ``eth0`` has a link but never gets a
DHCP lease, and there is no Wi-Fi. Its only route out is the USB cable. So the
laptop opens a reverse tunnel before this runs::

    adb reverse tcp:8000 tcp:8000      # on the laptop, once

after which ``127.0.0.1:8000`` on the board *is* port 8000 on the laptop, and
``--cloud`` needs no address discovery. Without it the POSTs fail, the run
continues, and the telemetry still lands in ``results/`` -- offline is an
expected state here, not an error.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "3-pipeline"))

#: Detectors this demo can load, by short name. Weights are not in git (10 MB
#: each), so a fresh clone has none of these until they are fetched -- hence
#: ``--list``, which reports what is actually on this machine rather than what
#: the table claims.
MODELS = {
    "v26n":    "models/v26n-base-encoder/v26n-base-encoder.onnx",
    "v26n-p2": "models/v26n-p2-encoder/v26n-p2-encoder.onnx",
    "v11n":    "models/v11n-base-encoder/v11n-base-encoder.onnx",
    "v8n":     "models/yolov8n_visdrone_640.onnx",
}
DEFAULT_MODEL = "v26n"

#: Stored clips ``--version sample`` can replay. Keyed by a short name because
#: the second one's filename is a 70-character download artefact that nobody
#: should have to type at an ssh prompt.
SAMPLES = {
    "uav": "video/uav0000076_00241_s_30fps.mp4",
    "vietnam": ("video/YTDown.com_YouTube_Crazy-yet-Organized-Traffic-in-"
                "Vietnam-H_Media_fm0cX7P6PSw_001_1080p.mp4"),
}
DEFAULT_SAMPLE = "uav"


def resolve_model(name: str) -> str:
    """A registry key, or a path to any other .onnx. Keys win over paths."""
    return MODELS.get(name, name)


def describe_models() -> str:
    """One line per model: is it here, how big, and how good was it.

    mAP comes from the ``summary.json`` the training run wrote next to the
    weights, so the number shown is that model's own measured score and not
    something restated by hand here, which would drift.
    """
    rows = ["  name       size     mAP50-95  mAP50   path"]
    for key, rel in MODELS.items():
        path = os.path.join(HERE, rel)
        if not os.path.exists(path):
            rows.append(f"  {key:<10} {'(chua co)':<8} {'':<9} {'':<7} {rel}")
            continue
        mb = f"{os.path.getsize(path) / 1e6:.1f}MB"
        m95 = m50 = ""
        summary = os.path.join(os.path.dirname(path), "summary.json")
        try:
            with open(summary, encoding="utf-8") as fh:
                s = json.load(fh)
            m95 = f"{float(s.get('mAP50-95', 0)):.4f}"
            m50 = f"{float(s.get('mAP50', 0)):.4f}"
        except (OSError, ValueError, TypeError):
            pass                        # no summary next to it: leave blank
        rows.append(f"  {key:<10} {mb:<8} {m95:<9} {m50:<7} {rel}")
    return "\n".join(rows)


def fast_cores(want: int = 3) -> list[int]:
    """The ``want`` highest-clocked CPUs, by measured maximum frequency.

    Read from sysfs rather than hardcoded: the core numbering is not a
    guarantee, and a board with a different cluster layout should still get
    its big cores rather than whichever ids happened to be fast here.
    """
    freqs: list[tuple[int, int]] = []
    base = "/sys/devices/system/cpu"
    try:
        entries = sorted(e for e in os.listdir(base) if e.startswith("cpu")
                         and e[3:].isdigit())
    except OSError:
        return []
    for e in entries:
        try:
            with open(f"{base}/{e}/cpufreq/cpuinfo_max_freq") as fh:
                freqs.append((int(fh.read().strip()), int(e[3:])))
        except OSError:
            continue          # cpufreq absent (a VM, or a host run) -- skip
    if not freqs:
        return []
    freqs.sort(reverse=True)
    return sorted(cpu for _f, cpu in freqs[:want])


def check_writable(path: str) -> str:
    """Empty string if telemetry can be written to ``path``, else the reason.

    Worth a preflight because of how this board is actually used: ``adb shell``
    is root, ``ssh qrobot@`` is not, and anything the pipeline wrote during a
    root-side test leaves a root-owned SQLite file behind. The next run over SSH
    then dies six frames deep inside sqlite3 with 'attempt to write a readonly
    database', which names neither the file nor the fix.
    """
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as e:
        return f"khong tao duoc {path}: {e}"
    if not os.access(path, os.W_OK | os.X_OK):
        return f"thu muc {path} khong ghi duoc"
    # The directory being writable is not enough: SQLite needs to write the
    # existing file itself, plus the -wal and -shm beside it.
    for name in ("telemetry.sqlite", "runtime_telemetry.csv"):
        f = os.path.join(path, name)
        if os.path.exists(f) and not os.access(f, os.W_OK):
            return f"file {f} khong ghi duoc"
    return ""


def pin(cores: list[int]) -> bool:
    if not cores or not hasattr(os, "sched_setaffinity"):
        return False
    try:
        os.sched_setaffinity(0, set(cores))
        return True
    except OSError:
        return False


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--version", choices=("sample", "realtime"),
                   help="sample = replay the stored clip; realtime = live camera")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"detector: one of {', '.join(MODELS)}, or a path to "
                        f"any other .onnx. Default {DEFAULT_MODEL}.")
    p.add_argument("--list", action="store_true",
                   help="show which models are present on this machine, with "
                        "their measured mAP, then exit")
    p.add_argument("--cloud", default="http://127.0.0.1:8000/ingest",
                   help="command post ingest URL; '' to run fully offline. "
                        "Reached through 'adb reverse tcp:8000 tcp:8000'.")
    p.add_argument("--port", type=int, default=8090,
                   help="MJPEG port for the annotated stream")
    p.add_argument("--cores", type=int, default=0,
                   help="pin to the N fastest cores; 0 (the default) does not "
                        "pin at all, which measured fastest and steadiest")
    p.add_argument("--threads", type=int, default=3,
                   help="ORT intra-op threads. 3 is not a guess -- see the "
                        "table in this file's docstring")
    p.add_argument("--max-frames", type=int, default=0,
                   help="stop after N frames; 0 runs to the end of the source")
    p.add_argument("--loop", action="store_true",
                   help="replay the clip forever. The sample runs out after "
                        "1560 frames (~3 min at the measured rate) and the "
                        "live view dies with it, which is short for a demo")
    p.add_argument("--sample", default=DEFAULT_SAMPLE, choices=sorted(SAMPLES),
                   help="which stored clip --version sample replays")
    p.add_argument("--speed", type=float, default=1.0,
                   help="play the clip faster by dropping frames in between "
                        "(1.5 = 1.5x). It does not raise FPS -- the pipeline is "
                        "already flat out -- it covers more of the clip per "
                        "second. Bigger jumps between frames make tracking and "
                        "line counting harder, so numbers taken at >1 are not "
                        "comparable with numbers taken at 1.")
    p.add_argument("--source", default=None,
                   help="override the source for the chosen mode")
    args = p.parse_args(argv)

    os.chdir(HERE)

    if args.list:
        print("Models:"); print(describe_models())
        print("\nClips:")
        for k, v in SAMPLES.items():
            mark = "" if os.path.exists(v) else "  (chua co)"
            print(f"  {k:<10}{mark}  {v}")
        return 0
    if not args.version:
        p.error("--version la bat buoc (sample hoac realtime); "
                "hoac dung --list de xem model co san")

    model = resolve_model(args.model)
    if not os.path.exists(model):
        print(f"[fatal] khong thay model: {model}\n"
              f"        xem cai nao co san bang: run_demo.py --list",
              file=sys.stderr)
        return 2

    why = check_writable("results")
    if why:
        try:
            me = os.getlogin()
        except OSError:                       # no controlling tty (cron, adb)
            me = str(getattr(os, "getuid", lambda: "?")())
        print(f"[fatal] khong ghi duoc telemetry: {why}\n"
              f"        Nguyen nhan thuong gap: 'adb shell' chay bang root nen\n"
              f"        file test truoc do thuoc root, con ban dang la '{me}'.\n"
              f"        Sua mot lan:\n"
              f"            sudo chown -R $(id -un):$(id -gn) {HERE}",
              file=sys.stderr)
        return 2

    cores = fast_cores(args.cores)
    pinned = pin(cores)
    threads = args.threads or (len(cores) if cores else 0)

    print(f"[demo] mode      {args.version}"
          f"{'  clip=' + args.sample if args.version == 'sample' else ''}"
          f"{'  speed=x' + format(args.speed, 'g') if args.speed != 1.0 else ''}")
    print(f"[demo] model     {args.model}  ({model})")
    print(f"[demo] cores     {cores if pinned else 'khong ghim duoc'}"
          f"{'' if pinned else ' (chay tren mac dinh)'}")
    print(f"[demo] threads   {threads or 'mac dinh ORT'}")

    if args.version == "sample":
        source = args.source or SAMPLES[args.sample]
        if not os.path.exists(source):
            have = [k for k, v in SAMPLES.items() if os.path.exists(v)]
            print(f"[fatal] khong thay video mau: {source}\n"
                  f"        co san tren board: {', '.join(have) or '(khong co)'}\n"
                  f"        day len bang: adb push video/... "
                  f"/home/qrobot/edge-uav-traffic/video/", file=sys.stderr)
            return 2
    else:
        # An index, not a path: run_pipeline resolves a bare integer to
        # cv2.VideoCapture(N). Probing for the right one belongs in
        # 0-setup/probe_camera.py, not here.
        source = args.source or "2"

    argv_pipe = [
        "--model", model,
        "--source", source,
        "--backend", "onnxruntime-cpu",
        "--headless",
        "--mjpeg-port", str(args.port),
        "--session-id", f"{args.version}",
    ]
    if threads:
        argv_pipe += ["--threads", str(threads)]
    if args.max_frames:
        argv_pipe += ["--max-frames", str(args.max_frames)]
    if args.speed != 1.0:
        argv_pipe += ["--speed", str(args.speed)]
    if args.loop and args.version == "sample":
        # A camera index cannot be rewound, so looping is only meaningful for
        # the stored clip; run_pipeline says the same but silently.
        argv_pipe += ["--loop"]
    if args.cloud:
        argv_pipe += ["--post-url", args.cloud]

    print(f"[demo] cloud     {args.cloud or '(tat)'}")
    print(f"[demo] xem tai   http://<ip-laptop>:{args.port}/  "
          f"(qua 'adb forward tcp:{args.port} tcp:{args.port}')")
    print()

    import run_pipeline
    return run_pipeline.main(argv_pipe)


if __name__ == "__main__":
    raise SystemExit(main())
