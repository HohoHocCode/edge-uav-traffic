"""Threaded video I/O.

Decoding a 1904x1070 H.264 frame costs 5-8 ms and encoding costs about the same.
Run in the main loop that is 10-16 ms per frame spent doing nothing but waiting
on the CPU, which on this pipeline is more than the GPU forward pass itself.
Both are moved onto their own threads so they overlap with inference; the main
loop then only pays whichever of the three is slowest, not their sum.

This is why ``sv.VideoSink`` is not used: it writes synchronously inside the
loop, which is exactly the cost this module exists to remove.

Frame order is the thing that must not break. Everything downstream --
ByteTrack, the crossing counter, the congestion hysteresis -- is stateful, and
feeding it frames out of order corrupts track identity silently rather than
raising. The reader therefore emits ``(index, frame)`` pairs from a single
thread in decode order, and the writer consumes from a single queue, so order is
structural rather than something the caller has to be careful about.
"""

from __future__ import annotations

import os
import queue
import threading
from dataclasses import dataclass

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise SystemExit("opencv is required: pip install opencv-python") from exc


@dataclass
class VideoMeta:
    path: str
    width: int
    height: int
    fps: float
    n_frames: int          # frames this run will actually process
    total_frames: int      # frames in the file
    start_frame: int

    @property
    def duration_s(self) -> float:
        return self.n_frames / max(self.fps, 1e-6)

    def as_dict(self) -> dict:
        return {
            "source": os.path.basename(self.path),
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 3),
            "n_frames": self.n_frames,
            "total_frames": self.total_frames,
            "start_frame": self.start_frame,
            "duration_s": round(self.duration_s, 3),
        }


def probe(path: str, start_s: float = 0.0, duration_s: float = 0.0) -> VideoMeta:
    """Read the header and resolve --start/--duration into a frame range."""
    if not os.path.isfile(path):
        raise SystemExit(f"[fatal] source not found: {path}")
    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise SystemExit(f"[fatal] cannot open source: {path}")
    try:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()

    start_frame = max(0, int(round(start_s * fps)))
    if start_frame >= total > 0:
        raise SystemExit(
            f"[fatal] --start {start_s}s is past the end of a {total / fps:.1f}s video"
        )
    remaining = (total - start_frame) if total > 0 else 0
    n = int(round(duration_s * fps)) if duration_s > 0 else remaining
    if remaining > 0:
        n = min(n, remaining)
    return VideoMeta(path, w, h, fps, n, total, start_frame)


# --------------------------------------------------------------------------- #
class ThreadedReader:
    """Decode on a worker thread; yield ``(index, frame)`` in decode order."""

    def __init__(self, meta: VideoMeta, queue_size: int = 48) -> None:
        self.meta = meta
        self._q: queue.Queue = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="reader", daemon=True)
        self._error: BaseException | None = None
        self.frames_read = 0

    def _run(self) -> None:
        cap = cv2.VideoCapture(self.meta.path, cv2.CAP_FFMPEG)
        try:
            if not cap.isOpened():
                raise RuntimeError(f"cannot open {self.meta.path}")
            if self.meta.start_frame:
                cap.set(cv2.CAP_PROP_POS_FRAMES, self.meta.start_frame)
            i = 0
            while i < self.meta.n_frames and not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    break                      # short file, or a decode error
                self._q.put((i, frame))
                i += 1
            self.frames_read = i
        except BaseException as exc:            # surfaced in the consumer
            self._error = exc
        finally:
            cap.release()
            self._q.put(None)                   # sentinel, always

    def __enter__(self) -> "ThreadedReader":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        # Drain so a blocked producer can reach its sentinel and exit.
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass
        self._thread.join(timeout=5.0)

    def __iter__(self):
        while True:
            item = self._q.get()
            if item is None:
                break
            yield item
        if self._error is not None:
            raise SystemExit(f"[fatal] reader failed: {self._error}")

    def batches(self, size: int):
        """Yield ``(indices, frames)`` chunks of at most `size`, in order."""
        idxs: list[int] = []
        frames: list[np.ndarray] = []
        for i, frame in self:
            idxs.append(i)
            frames.append(frame)
            if len(frames) >= size:
                yield idxs, frames
                idxs, frames = [], []
        if frames:
            yield idxs, frames


# --------------------------------------------------------------------------- #
class ThreadedWriter:
    """Encode on a worker thread. Writes in the order frames are submitted."""

    def __init__(
        self,
        path: str,
        width: int,
        height: int,
        fps: float,
        fourcc: str = "mp4v",
        queue_size: int = 48,
    ) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.path = path
        self.writer = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*fourcc), fps, (width, height)
        )
        if not self.writer.isOpened():
            raise SystemExit(f"[fatal] cannot open VideoWriter for {path}")
        self._q: queue.Queue = queue.Queue(maxsize=queue_size)
        self._thread = threading.Thread(target=self._run, name="writer", daemon=True)
        self.frames_written = 0

    def _run(self) -> None:
        while True:
            frame = self._q.get()
            if frame is None:
                break
            self.writer.write(frame)
            self.frames_written += 1

    def __enter__(self) -> "ThreadedWriter":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._q.put(None)
        self._thread.join(timeout=30.0)
        self.writer.release()

    def write(self, frame: np.ndarray) -> None:
        self._q.put(frame)


# --------------------------------------------------------------------------- #
def transcode_h264(src: str, crf: int = 18, ffmpeg: str | None = None) -> str | None:
    """Re-encode the mp4v output to H.264 in place. Returns the path, or None.

    ``mp4v`` is what every VideoWriter in this repository uses and it is what
    OpenCV can be relied on to have, but it produces files several times larger
    than H.264 and some players refuse them outright.
    """
    import shutil
    import subprocess

    exe = ffmpeg or shutil.which("ffmpeg")
    if not exe:
        print("[warn] ffmpeg not found on PATH; keeping the mp4v file")
        return None

    tmp = src.replace(".mp4", ".h264.mp4")
    cmd = [exe, "-y", "-loglevel", "error", "-i", src,
           "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", tmp]
    try:
        subprocess.run(cmd, check=True)
    except (subprocess.CalledProcessError, OSError) as exc:
        print(f"[warn] ffmpeg transcode failed ({exc}); keeping the mp4v file")
        if os.path.exists(tmp):
            os.remove(tmp)
        return None

    before, after = os.path.getsize(src), os.path.getsize(tmp)
    os.replace(tmp, src)
    print(f"[info] h264: {before / 1e6:.1f} MB -> {after / 1e6:.1f} MB")
    return src
