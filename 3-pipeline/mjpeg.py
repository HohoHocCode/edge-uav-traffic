"""MJPEG stream server — how you watch a headless board.

The board is reached over SSH. There is no display, so ``cv2.imshow`` is not
an option, and writing a file means the demo can only be reviewed afterwards.
A live demo needs the annotated frames to leave the device over the network,
and MJPEG over HTTP is the way to do that with no client software at all: any
browser renders ``multipart/x-mixed-replace`` natively.

The single rule this module obeys: **the pipeline never waits for a viewer.**
Encoding happens on the server thread, the latest frame simply overwrites the
previous one, and a slow or absent client costs the detector nothing. A demo
that stutters because someone opened a second browser tab is worse than no
demo.
"""

from __future__ import annotations

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

BOUNDARY = "skysentryframe"

PAGE = """<!doctype html>
<html lang="vi"><head><meta charset="utf-8">
<title>SkySentry — luồng trực tiếp</title>
<style>
 :root{color-scheme:dark}
 body{margin:0;background:#0f1115;color:#e8eaed;
      font:14px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
 header{padding:12px 18px;border-bottom:1px solid #252a34;display:flex;
        align-items:baseline;gap:14px;flex-wrap:wrap}
 h1{font-size:15px;margin:0;letter-spacing:.2px}
 .sub{color:#9aa3b2;font-size:12px}
 .dot{width:8px;height:8px;border-radius:50%;background:#4ade80;
      display:inline-block;margin-right:6px}
 main{padding:16px 18px}
 img{max-width:100%;height:auto;display:block;border:1px solid #252a34;
     border-radius:8px;background:#000}
 footer{padding:10px 18px;color:#9aa3b2;font-size:12px;border-top:1px solid #252a34}
 code{background:#171a21;padding:1px 6px;border-radius:4px}
</style></head><body>
<header>
  <h1>SkySentry — node biên</h1>
  <span class="sub"><span class="dot"></span>đang phát</span>
  <span class="sub" style="margin-left:auto">MJPEG · làm mới liên tục</span>
</header>
<main><img src="/stream" alt="luồng trực tiếp"></main>
<footer>Nếu ảnh đứng yên, tải lại trang. Ảnh tĩnh một khung:
<code>/snapshot</code></footer>
</body></html>"""


class FrameBuffer:
    """Single-slot buffer. A new frame replaces the old one; nothing queues."""

    def __init__(self) -> None:
        self._jpeg: bytes | None = None
        self._seq = 0
        self._cv = threading.Condition()
        self.frames_served = 0
        self.clients = 0

    def put(self, jpeg: bytes) -> None:
        with self._cv:
            self._jpeg = jpeg
            self._seq += 1
            self._cv.notify_all()

    def get(self, last_seq: int, timeout: float = 5.0):
        """Block until a frame newer than ``last_seq`` exists."""
        with self._cv:
            if self._seq <= last_seq:
                self._cv.wait(timeout)
            if self._jpeg is None or self._seq <= last_seq:
                return None, last_seq
            return self._jpeg, self._seq

    def latest(self) -> bytes | None:
        with self._cv:
            return self._jpeg


class _Handler(BaseHTTPRequestHandler):
    buffer: FrameBuffer = None      # type: ignore[assignment]
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *a):
        pass                         # a request log per frame is unusable

    def handle_one_request(self):
        # A viewer closing the tab aborts the socket mid-read, which
        # socketserver reports as an unhandled exception and prints as a
        # traceback. During a live demo that fills the console with noise
        # every time someone switches away. Closed connections are the
        # normal end of an MJPEG stream, not an error.
        try:
            super().handle_one_request()
        except (ConnectionError, TimeoutError, OSError):
            self.close_connection = True

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif self.path == "/snapshot":
            jpg = self.buffer.latest()
            if jpg is None:
                self.send_error(503, "no frame yet")
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpg)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(jpg)

        elif self.path == "/stream":
            self._stream()

        elif self.path == "/health":
            body = b'{"ok":true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def _stream(self) -> None:
        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type",
                         f"multipart/x-mixed-replace; boundary={BOUNDARY}")
        self.end_headers()

        self.buffer.clients += 1
        seq = 0
        try:
            while True:
                jpg, seq = self.buffer.get(seq)
                if jpg is None:
                    continue          # timed out waiting; keep the socket open
                self.wfile.write(b"--" + BOUNDARY.encode() + b"\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(
                    b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n")
                self.wfile.write(jpg)
                self.wfile.write(b"\r\n")
                self.buffer.frames_served += 1
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass                      # viewer closed the tab; entirely normal
        finally:
            self.buffer.clients = max(0, self.buffer.clients - 1)


class MjpegServer:
    """Serve annotated frames over HTTP. Start once, ``publish`` per frame."""

    def __init__(self, port: int = 8090, host: str = "0.0.0.0",
                 quality: int = 75, max_width: int = 0,
                 idle_interval_s: float = 1.0) -> None:
        self.buffer = FrameBuffer()
        self.quality = int(quality)
        self.max_width = int(max_width)
        self.idle_interval_s = float(idle_interval_s)
        self._last_idle_encode = 0.0
        _Handler.buffer = self.buffer
        self._srv = ThreadingHTTPServer((host, port), _Handler)
        self._srv.daemon_threads = True
        self._thread = threading.Thread(target=self._srv.serve_forever,
                                        daemon=True)
        self.port = port
        self.published = 0
        self.encode_ms = 0.0

    def start(self) -> "MjpegServer":
        self._thread.start()
        return self

    def publish(self, frame: np.ndarray) -> None:
        """Encode and hand off. Throttles hard when nobody is watching."""
        now = time.perf_counter()
        if self.buffer.clients == 0:
            # No viewer: encoding every frame is pure waste, and on the Kryo
            # cores at 1280x720 it is not cheap. But dropping *every* frame
            # leaves /snapshot with nothing to serve and the page blank until
            # a stream client connects, so keep a slow trickle alive.
            if now - self._last_idle_encode < self.idle_interval_s:
                return
            self._last_idle_encode = now
        f = frame
        if self.max_width and f.shape[1] > self.max_width:
            s = self.max_width / f.shape[1]
            f = cv2.resize(f, (self.max_width, int(round(f.shape[0] * s))),
                           interpolation=cv2.INTER_AREA)
        t0 = time.perf_counter()
        ok, buf = cv2.imencode(".jpg", f,
                               [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
        if not ok:
            return
        self.encode_ms = (time.perf_counter() - t0) * 1000.0
        self.buffer.put(buf.tobytes())
        self.published += 1

    def stop(self) -> None:
        try:
            self._srv.shutdown()
            self._srv.server_close()
        except Exception:
            pass

    def summary(self) -> dict:
        return {"port": self.port, "published": self.published,
                "frames_served": self.buffer.frames_served,
                "viewers": self.buffer.clients}
