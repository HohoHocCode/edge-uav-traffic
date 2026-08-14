#!/usr/bin/env python3
"""Command post — ingest server + dashboard.

Zero third-party dependencies: stdlib ``http.server`` and ``sqlite3`` only.
That is a deliberate constraint. The command post has to come up on whatever
machine is available on the day, including the board itself, and "pip install
failed on the venue wifi" is not an acceptable failure mode for a demo.

    python 5-server/app.py --host 0.0.0.0 --port 8000

Endpoints
    POST /ingest          batch of frame rows from an edge node
    POST /frame           one annotated JPEG (multipart-free: raw body)
    GET  /live            MJPEG re-broadcast of the latest frame
    GET  /api/telemetry   recent rows, flattened for the dashboard
    GET  /api/state       latest state of every node
    GET  /api/timeseries  recent counts for the chart
    GET  /api/events      recent congestion events
    GET  /                dashboard

Why the store is SQLite and not the telemetry CSV
-------------------------------------------------
An earlier revision of this file read ``results/runtime_telemetry.csv`` on
every poll and returned its last hundred rows. That is wrong twice over. It
re-parses the whole file once a second, so a ten-minute run at 30 fps means
re-reading eighteen thousand rows every second and the dashboard slows down
the longer it is left open. And it means the command post can only ever watch
a node that shares its filesystem -- there is no ingest, so the board cannot
push anything, and the whole edge-to-cloud path the pipeline was built around
does not exist. Both endpoints are kept here, but they read the database that
``/ingest`` fills.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sqlite3
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))

#: Where the edge node's own MjpegServer is reachable *from this process*. Over
#: USB that is an ``adb forward`` of the board's 8090 onto localhost; over a LAN
#: it is the board's IP. ``/api/stream`` pipes it through so the dashboard reads
#: live annotated video from the same origin as its telemetry, no matter which
#: transport carried it here. Set by --mjpeg.
MJPEG_UPSTREAM: tuple[str, int] = ("127.0.0.1", 8090)

#: Largest accepted /ingest body. A node posts batches of <= 64 frame rows,
#: so 8 MB is generous by two orders of magnitude.
MAX_BODY_BYTES = 8 * 1024 * 1024
#: A 1920x1080 JPEG at quality 95 is about 1 MB; 16 MB is room to spare.
MAX_FRAME_BYTES = 16 * 1024 * 1024

BOUNDARY = "skysentryframe"

SCHEMA = """
CREATE TABLE IF NOT EXISTS ingest (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id       TEXT NOT NULL,
    node             TEXT,
    frame_id         INTEGER,
    received_at      REAL NOT NULL,
    timestamp_ms     REAL,
    n_tracks         INTEGER,
    vehicle_count    INTEGER,
    person_count     INTEGER,
    congestion_level TEXT,
    total_ms         REAL,
    infer_ms         REAL,
    raw_json         TEXT
);
CREATE INDEX IF NOT EXISTS idx_ingest_session ON ingest(session_id, id);
CREATE INDEX IF NOT EXISTS idx_ingest_time ON ingest(received_at);

CREATE TABLE IF NOT EXISTS node_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    node        TEXT,
    at          REAL NOT NULL,
    level       TEXT,
    detail      TEXT
);
"""


class FrameBuffer:
    """Single-slot buffer for the live view.

    Deliberately the same shape as the one in ``3-pipeline/mjpeg.py``: a new
    frame replaces the old one and nothing queues, so a slow browser costs the
    edge node nothing and the picture is never behind by more than one frame.
    """

    def __init__(self) -> None:
        self._jpeg: bytes | None = None
        self._seq = 0
        self._cv = threading.Condition()
        self.received = 0

    def put(self, jpeg: bytes) -> None:
        with self._cv:
            self._jpeg = jpeg
            self._seq += 1
            self.received += 1
            self._cv.notify_all()

    def get(self, last_seq: int, timeout: float = 5.0):
        with self._cv:
            if self._seq <= last_seq:
                self._cv.wait(timeout)
            if self._jpeg is None or self._seq <= last_seq:
                return None, last_seq
            return self._jpeg, self._seq

    def latest(self) -> bytes | None:
        with self._cv:
            return self._jpeg


class Store:
    def __init__(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.executescript(SCHEMA)
        self.conn.commit()
        self.lock = threading.Lock()

    def ingest(self, payload: dict) -> int:
        session = str(payload.get("session_id", "unknown"))
        meta = payload.get("meta", {}) or {}
        node = str(meta.get("device", "node"))
        frames = payload.get("frames", []) or []
        now = time.time()

        rows = []
        for fr in frames:
            rows.append((
                session, node, fr.get("frame_id"), now, fr.get("timestamp_ms"),
                fr.get("n_tracks"), fr.get("vehicle_count"), fr.get("person_count"),
                fr.get("congestion_level"), fr.get("total_ms"), fr.get("infer_ms"),
                json.dumps(fr, default=str),
            ))
        if not rows:
            return 0

        with self.lock:
            self.conn.executemany(
                "INSERT INTO ingest (session_id, node, frame_id, received_at,"
                " timestamp_ms, n_tracks, vehicle_count, person_count,"
                " congestion_level, total_ms, infer_ms, raw_json)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows,
            )
            # Record a level change as an event, so the dashboard has an
            # activity feed without the client having to diff the stream.
            prev = self.conn.execute(
                "SELECT level FROM node_events WHERE session_id=? "
                "ORDER BY id DESC LIMIT 1", (session,)
            ).fetchone()
            prev_level = prev[0] if prev else "normal"
            for fr in frames:
                lvl = fr.get("congestion_level")
                if lvl and lvl != prev_level:
                    self.conn.execute(
                        "INSERT INTO node_events (session_id, node, at, level, detail)"
                        " VALUES (?,?,?,?,?)",
                        (session, node, now, lvl,
                         json.dumps({"from": prev_level, "to": lvl,
                                     "vehicles": fr.get("vehicle_count"),
                                     "frame_id": fr.get("frame_id")})),
                    )
                    prev_level = lvl
            self.conn.commit()
        return len(rows)

    def state(self) -> list[dict]:
        with self.lock:
            cur = self.conn.execute(
                "SELECT session_id, node, MAX(id) AS mid FROM ingest GROUP BY session_id"
            )
            sessions = cur.fetchall()
            out = []
            for session, node, mid in sessions:
                r = self.conn.execute(
                    "SELECT frame_id, received_at, n_tracks, vehicle_count,"
                    " person_count, congestion_level, total_ms, infer_ms"
                    " FROM ingest WHERE id=?", (mid,)
                ).fetchone()
                n = self.conn.execute(
                    "SELECT COUNT(*) FROM ingest WHERE session_id=?", (session,)
                ).fetchone()[0]
                if not r:
                    continue
                out.append({
                    "session_id": session, "node": node, "frame_id": r[0],
                    "age_s": round(time.time() - r[1], 1),
                    "n_tracks": r[2], "vehicle_count": r[3], "person_count": r[4],
                    "congestion_level": r[5], "total_ms": r[6], "infer_ms": r[7],
                    "frames_received": n,
                })
        return out

    def timeseries(self, session: str | None, limit: int = 240) -> list[dict]:
        with self.lock:
            if session:
                cur = self.conn.execute(
                    "SELECT frame_id, vehicle_count, person_count, n_tracks, total_ms"
                    " FROM ingest WHERE session_id=? ORDER BY id DESC LIMIT ?",
                    (session, limit),
                )
            else:
                cur = self.conn.execute(
                    "SELECT frame_id, vehicle_count, person_count, n_tracks, total_ms"
                    " FROM ingest ORDER BY id DESC LIMIT ?", (limit,),
                )
            rows = cur.fetchall()
        rows.reverse()
        return [
            {"frame_id": r[0], "vehicle_count": r[1], "person_count": r[2],
             "n_tracks": r[3], "total_ms": r[4]}
            for r in rows
        ]

    def telemetry(self, limit: int = 100) -> list[dict]:
        """Recent rows in the shape the dashboard expects.

        The row the node posted is stored verbatim in ``raw_json``, so the
        per-class and per-ROI columns survive without this table having to
        grow a column per class. Indexing by ``id DESC`` then reversing costs
        one index seek, against the whole-file re-parse this replaces.
        """
        with self.lock:
            cur = self.conn.execute(
                "SELECT raw_json FROM ingest ORDER BY id DESC LIMIT ?", (limit,)
            )
            rows = [r[0] for r in cur.fetchall()]
        rows.reverse()

        out = []
        for raw in rows:
            try:
                r = json.loads(raw)
            except (TypeError, ValueError):
                continue

            def num(key, cast=int, default=0):
                try:
                    return cast(r.get(key, default) or default)
                except (TypeError, ValueError):
                    return default

            out.append({
                "frame_id": num("frame_id"),
                "timestamp_ms": num("timestamp_ms", float, 0.0),
                # Per-task fields. Absent from runs recorded before they
                # existed, hence the 0 defaults rather than a KeyError.
                "n_dets": num("n_dets"),                    # task 2
                "conf_mean": num("conf_mean", float, 0.0),
                "conf_min": num("conf_min", float, 0.0),
                "n_new": num("n_new"),                      # task 4
                "n_lost": num("n_lost"),
                "n_tentative": num("n_tentative"),
                "ids_issued": num("ids_issued"),
                "track_age_mean": num("track_age_mean", float, 0.0),
                "n_tracks": num("n_tracks"),
                "vehicle_count": num("vehicle_count"),
                "person_count": num("person_count"),
                "congestion_level": r.get("congestion_level", "normal"),
                # Whatever cls_* the node sent, without a hardcoded class list:
                # the taxonomy lives in configs/visdrone.yaml, not here.
                "cls": {k[4:]: num(k) for k in r if k.startswith("cls_")},
                "roi_intersection": num("roi_intersection"),
                "roi_full_frame": num("roi_full_frame"),
                "line_fwd": num("line_main_road_fwd"),
                "line_bwd": num("line_main_road_bwd"),
                "pre": num("pre_ms", float, 0.0),
                "infer": num("infer_ms", float, 0.0),
                "post": num("post_ms", float, 0.0),
                "track": num("track_ms", float, 0.0),
                "total": num("total_ms", float, 0.0),
                "session_id": r.get("session_id", ""),
            })
        return out

    def events(self, limit: int = 40) -> list[dict]:
        with self.lock:
            cur = self.conn.execute(
                "SELECT session_id, node, at, level, detail FROM node_events"
                " ORDER BY id DESC LIMIT ?", (limit,),
            )
            rows = cur.fetchall()
        return [
            {"session_id": r[0], "node": r[1],
             "at": time.strftime("%H:%M:%S", time.localtime(r[2])),
             "level": r[3], "detail": json.loads(r[4]) if r[4] else {}}
            for r in rows
        ]


class Handler(BaseHTTPRequestHandler):
    store: Store = None            # type: ignore[assignment]
    frames: FrameBuffer = None     # type: ignore[assignment]
    #: Task the dashboard is showing; handed to the board on each /ingest
    #: reply so its overlay matches. Class-level: every request thread reads
    #: and writes the same one, and a str rebind is atomic under the GIL.
    view: str = "4"
    server_version = "SkySentryCommandPost/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *a):   # quieter default logging
        if os.environ.get("SKYSENTRY_VERBOSE"):
            super().log_message(fmt, *a)

    def handle_one_request(self):
        # A viewer closing the tab aborts an MJPEG socket mid-write, which
        # socketserver reports as an unhandled exception and prints as a
        # traceback. That is the normal end of a stream, not an error.
        try:
            super().handle_one_request()
        except (ConnectionError, TimeoutError, OSError):
            self.close_connection = True

    # ------------------------------------------------------------------ #
    def _json(self, obj, code: int = 200) -> None:
        body = json.dumps(obj, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: str, ctype: str) -> None:
        try:
            with open(path, "rb") as f:
                body = f.read()
        except OSError:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _limit(q: dict, default: int, cap: int = 5000) -> int:
        """Parse a ?limit= value defensively.

        A non-numeric value used to raise ValueError inside the handler and
        return a 500; an unbounded one let a single request pull the whole
        table into memory.
        """
        raw = (q.get("limit") or [str(default)])[0]
        try:
            n = int(raw)
        except (TypeError, ValueError):
            return default
        return max(1, min(n, cap))

    def _stream(self) -> None:
        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Content-Type",
                         f"multipart/x-mixed-replace; boundary={BOUNDARY}")
        self.end_headers()
        seq = 0
        try:
            while True:
                jpg, seq = self.frames.get(seq)
                if jpg is None:
                    continue          # timed out waiting; keep the socket open
                self.wfile.write(b"--" + BOUNDARY.encode() + b"\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(
                    b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n")
                self.wfile.write(jpg)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass                      # viewer closed the tab; entirely normal

    def _proxy_stream(self) -> None:
        """Pipe the edge node's multipart MJPEG straight through to the browser.

        The board already solved "serve annotated frames as MJPEG" in
        ``3-pipeline/mjpeg.py``; re-implementing it here would be a second copy
        to keep in sync. Instead this opens one upstream connection to that
        server (reached over ``adb forward`` or the LAN) and forwards its body
        verbatim. A dropped viewer closes only this hop; the board never waits.
        """
        host, port = MJPEG_UPSTREAM
        try:
            up = socket.create_connection((host, port), timeout=5)
        except OSError:
            self.send_error(503, "edge stream unavailable")
            return
        try:
            up.sendall(
                b"GET /stream HTTP/1.1\r\nHost: " + host.encode()
                + b"\r\nConnection: keep-alive\r\n\r\n"
            )
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Type",
                             f"multipart/x-mixed-replace; boundary={BOUNDARY}")
            self.end_headers()
            # Drop the upstream status line and headers, forward the body.
            buf = b""
            while b"\r\n\r\n" not in buf:
                chunk = up.recv(4096)
                if not chunk:
                    return
                buf += chunk
            body = buf.split(b"\r\n\r\n", 1)[1]
            if body:
                self.wfile.write(body)
            while True:
                chunk = up.recv(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)
        except (ConnectionError, TimeoutError, OSError):
            pass                      # viewer or upstream went away; normal
        finally:
            try:
                up.close()
            except OSError:
                pass

    def do_GET(self) -> None:
        u = urlparse(self.path)
        q = parse_qs(u.query)

        if u.path in ("/", "/index.html"):
            self._file(os.path.join(HERE, "dashboard.html"),
                       "text/html; charset=utf-8")
        elif u.path == "/live":
            self._stream()
        elif u.path == "/api/stream":
            self._proxy_stream()
        elif u.path in ("/api/video_feed", "/snapshot"):
            jpg = self.frames.latest()
            if jpg is None:
                self.send_error(503, "no frame yet")
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpg)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(jpg)
        elif u.path == "/api/telemetry":
            self._json(self.store.telemetry(self._limit(q, 100, cap=2000)))
        elif u.path == "/api/state":
            self._json(self.store.state())
        elif u.path == "/api/timeseries":
            self._json(self.store.timeseries(
                (q.get("session") or [None])[0], self._limit(q, 240),
            ))
        elif u.path == "/api/events":
            self._json(self.store.events(self._limit(q, 40, cap=1000)))
        elif u.path == "/health":
            self._json({"ok": True, "t": time.time(),
                        "frames_received": self.frames.received})
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        u = urlparse(self.path)
        try:
            n = int(self.headers.get("Content-Length", 0) or 0)
        except (TypeError, ValueError):
            self._json({"ok": False, "error": "bad Content-Length"}, 400)
            return

        if u.path == "/api/view":
            # The dashboard tells the command post which task is on screen.
            # It is not pushed to the board from here: the board is behind a
            # USB tunnel with no inbound route, so the value waits and rides
            # back on the node's next /ingest reply instead.
            try:
                want = json.loads(self.rfile.read(max(0, n)) or b"{}").get("view")
            except (ValueError, AttributeError):
                want = None
            if str(want) not in ("2", "4", "5"):
                self._json({"ok": False, "error": "view must be 2, 4 or 5"}, 400)
                return
            Handler.view = str(want)
            self._json({"ok": True, "view": Handler.view})
            return

        if u.path == "/frame":
            if n < 0 or n > MAX_FRAME_BYTES:
                self._json({"ok": False, "error": "frame too large"}, 413)
                return
            body = self.rfile.read(n)
            if not body[:2] == b"\xff\xd8":
                self._json({"ok": False, "error": "expected a JPEG body"}, 400)
                return
            self.frames.put(body)
            self._json({"ok": True, "bytes": len(body)})
            return

        if u.path != "/ingest":
            self.send_error(404)
            return

        if n < 0 or n > MAX_BODY_BYTES:
            # Without a cap, one client claiming a 4 GB body makes the server
            # allocate it. The node posts batches of at most 64 frame rows.
            self._json({"ok": False, "error": "payload too large"}, 413)
            return
        try:
            payload = json.loads(self.rfile.read(n) or b"{}")
        except (ValueError, json.JSONDecodeError) as exc:
            self._json({"ok": False, "error": str(exc)}, 400)
            return
        if not isinstance(payload, dict):
            self._json({"ok": False, "error": "expected a JSON object"}, 400)
            return
        try:
            written = self.store.ingest(payload)
        except sqlite3.Error as exc:
            self._json({"ok": False, "error": str(exc)}, 500)
            return
        # The reply is the only channel back to the board, so the current task
        # travels on it. A node from an older build simply ignores the field.
        self._json({"ok": True, "written": written, "view": Handler.view})

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Content-Length", "0")
        self.end_headers()


def main() -> int:
    global MJPEG_UPSTREAM
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--db", default=os.path.join(HERE, "..", "results", "command_post.sqlite"))
    ap.add_argument("--mjpeg", default="127.0.0.1:8090",
                    help="host:port where the edge node's MJPEG server is "
                         "reachable from here (adb forward or LAN); served back "
                         "at /api/stream")
    args = ap.parse_args()

    if ":" in args.mjpeg:
        h, p = args.mjpeg.rsplit(":", 1)
        MJPEG_UPSTREAM = (h, int(p))

    Handler.store = Store(os.path.abspath(args.db))
    Handler.frames = FrameBuffer()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.daemon_threads = True
    shown = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
    if args.host in ("0.0.0.0", "::"):
        print("[command post] WARNING: listening on all interfaces with no "
              "authentication.\n"
              "               Anyone on this network can post telemetry and "
              "read the dashboard.\n"
              "               Intended for a trusted lab LAN or a demo. Use "
              "--host 127.0.0.1,\n"
              "               or put a reverse proxy with auth in front, for "
              "anything else.")
    print(f"[command post] dashboard  http://{shown}:{args.port}/")
    print(f"[command post] ingest     http://{shown}:{args.port}/ingest")
    print(f"[command post] frames     http://{shown}:{args.port}/frame")
    print(f"[command post] mjpeg      /api/stream <- {MJPEG_UPSTREAM[0]}:{MJPEG_UPSTREAM[1]}")
    print(f"[command post] db         {os.path.abspath(args.db)}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[command post] stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
