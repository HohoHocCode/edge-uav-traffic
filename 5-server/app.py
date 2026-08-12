#!/usr/bin/env python3
"""Command post — ingest server + dashboard.

Zero third-party dependencies: stdlib ``http.server`` and ``sqlite3`` only.
That is a deliberate constraint. The command post has to come up on whatever
machine is available on the day, including the board itself, and "pip install
failed on the venue wifi" is not an acceptable failure mode for a demo.

    python 5-server/app.py --host 0.0.0.0 --port 8000

Endpoints
    POST /ingest          batch of frame rows from an edge node
    GET  /api/state       latest state of every node
    GET  /api/timeseries  recent counts for the chart
    GET  /api/events      recent congestion events
    GET  /                dashboard
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))

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
    server_version = "SkySentryCommandPost/1.0"

    def log_message(self, fmt, *a):   # quieter default logging
        if os.environ.get("SKYSENTRY_VERBOSE"):
            super().log_message(fmt, *a)

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
    def do_GET(self) -> None:
        u = urlparse(self.path)
        q = parse_qs(u.query)

        if u.path in ("/", "/index.html"):
            self._file(os.path.join(HERE, "static", "dashboard.html"), "text/html; charset=utf-8")
        elif u.path == "/api/state":
            self._json(self.store.state())
        elif u.path == "/api/timeseries":
            self._json(self.store.timeseries(
                (q.get("session") or [None])[0],
                int((q.get("limit") or [240])[0]),
            ))
        elif u.path == "/api/events":
            self._json(self.store.events(int((q.get("limit") or [40])[0])))
        elif u.path == "/health":
            self._json({"ok": True, "t": time.time()})
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        u = urlparse(self.path)
        if u.path != "/ingest":
            self.send_error(404)
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n) or b"{}")
        except (ValueError, json.JSONDecodeError) as exc:
            self._json({"ok": False, "error": str(exc)}, 400)
            return
        try:
            written = self.store.ingest(payload)
        except sqlite3.Error as exc:
            self._json({"ok": False, "error": str(exc)}, 500)
            return
        self._json({"ok": True, "written": written})

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--db", default=os.path.join(HERE, "..", "results", "command_post.sqlite"))
    args = ap.parse_args()

    Handler.store = Store(os.path.abspath(args.db))
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    shown = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
    print(f"[command post] dashboard  http://{shown}:{args.port}/")
    print(f"[command post] ingest     http://{shown}:{args.port}/ingest")
    print(f"[command post] db         {os.path.abspath(args.db)}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[command post] stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
