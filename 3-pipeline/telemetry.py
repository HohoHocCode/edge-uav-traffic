"""Telemetry sink: local CSV + SQLite, optional POST to the command post.

Design constraint that shapes everything here: the edge node must keep working
when the uplink is down. A UAV over a district loses line-of-sight constantly,
so the network is treated as a best-effort mirror of a local store, never as
the store itself.

    frame -> FrameReport -> CSV row      (always, append-only, flushed)
                         -> SQLite row   (always)
                         -> POST batch   (if configured, non-blocking, drops
                                          on failure and logs the drop)

Nothing in the hot path blocks on the network.
"""

from __future__ import annotations

import csv
import json
import os
import queue
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


SCHEMA = """
CREATE TABLE IF NOT EXISTS frames (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id        TEXT    NOT NULL,
    frame_id          INTEGER NOT NULL,
    timestamp_ms      REAL    NOT NULL,
    wall_clock        TEXT    NOT NULL,
    n_tracks          INTEGER NOT NULL,
    vehicle_count     INTEGER NOT NULL,
    person_count      INTEGER NOT NULL,
    congestion_level  TEXT    NOT NULL,
    pre_ms            REAL,
    infer_ms          REAL,
    post_ms           REAL,
    track_ms          REAL,
    total_ms          REAL,
    payload_json      TEXT
);
CREATE INDEX IF NOT EXISTS idx_frames_session ON frames(session_id);

CREATE TABLE IF NOT EXISTS events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT NOT NULL,
    frame_id      INTEGER NOT NULL,
    wall_clock    TEXT NOT NULL,
    kind          TEXT NOT NULL,
    severity      TEXT NOT NULL,
    detail_json   TEXT
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id    TEXT PRIMARY KEY,
    started_at    TEXT NOT NULL,
    ended_at      TEXT,
    device        TEXT,
    backend       TEXT,
    model         TEXT,
    imgsz         INTEGER,
    config_json   TEXT
);
"""


@dataclass
class TelemetryConfig:
    local_db: str = "results/telemetry.sqlite"
    csv_path: str = "results/runtime_telemetry.csv"
    post_url: str = ""
    post_interval_s: float = 2.0
    session_id: str = ""


class TelemetrySink:
    def __init__(self, cfg: TelemetryConfig, meta: dict | None = None) -> None:
        self.cfg = cfg
        self.session_id = cfg.session_id or time.strftime("%Y%m%d-%H%M%S")
        self.meta = meta or {}

        for p in (cfg.local_db, cfg.csv_path):
            if p:
                os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)

        # SQLite. check_same_thread=False because the uploader thread reads it.
        self.conn = sqlite3.connect(cfg.local_db, check_same_thread=False)
        self.conn.executescript(SCHEMA)
        self.conn.execute(
            "INSERT OR REPLACE INTO sessions "
            "(session_id, started_at, device, backend, model, imgsz, config_json) "
            "VALUES (?,?,?,?,?,?,?)",
            (
                self.session_id,
                time.strftime("%Y-%m-%dT%H:%M:%S"),
                self.meta.get("device", ""),
                self.meta.get("backend", ""),
                self.meta.get("model", ""),
                int(self.meta.get("imgsz", 0) or 0),
                json.dumps(self.meta, default=str),
            ),
        )
        self.conn.commit()
        self._db_lock = threading.Lock()

        # CSV — header written lazily on the first row so the column set can
        # include the ROI/line columns, which depend on the config.
        self._csv_file = open(cfg.csv_path, "w", newline="", encoding="utf-8") \
            if cfg.csv_path else None
        self._csv_writer: csv.DictWriter | None = None

        # Uploader
        self._q: queue.Queue = queue.Queue(maxsize=512)
        self._stop = threading.Event()
        self.dropped = 0
        self.posted = 0
        self.post_failures = 0
        self._thread: threading.Thread | None = None
        if cfg.post_url:
            self._thread = threading.Thread(target=self._uploader, daemon=True)
            self._thread.start()

    # ------------------------------------------------------------------ #
    def write(self, report, timings: dict) -> None:
        row = report.to_row()
        row.update({k: round(float(v), 3) for k, v in timings.items()})
        row["session_id"] = self.session_id

        if self._csv_file is not None:
            if self._csv_writer is None:
                self._csv_writer = csv.DictWriter(
                    self._csv_file, fieldnames=list(row.keys()), extrasaction="ignore"
                )
                self._csv_writer.writeheader()
            self._csv_writer.writerow(row)
            self._csv_file.flush()

        with self._db_lock:
            self.conn.execute(
                "INSERT INTO frames (session_id, frame_id, timestamp_ms, wall_clock,"
                " n_tracks, vehicle_count, person_count, congestion_level,"
                " pre_ms, infer_ms, post_ms, track_ms, total_ms, payload_json)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    self.session_id,
                    report.frame_id,
                    report.timestamp_ms,
                    time.strftime("%Y-%m-%dT%H:%M:%S"),
                    report.n_tracks,
                    report.vehicle_count,
                    report.person_count,
                    report.congestion_level,
                    timings.get("pre_ms"),
                    timings.get("infer_ms"),
                    timings.get("post_ms"),
                    timings.get("track_ms"),
                    timings.get("total_ms"),
                    json.dumps(row, default=str),
                ),
            )

        if self.cfg.post_url:
            try:
                self._q.put_nowait(row)
            except queue.Full:
                # Uplink is slower than the pipeline. Dropping the oldest
                # telemetry is correct: fresh state matters more than a
                # complete history, and the history is already on disk.
                self.dropped += 1

    # ------------------------------------------------------------------ #
    def event(self, frame_id: int, kind: str, severity: str, detail: dict) -> None:
        with self._db_lock:
            self.conn.execute(
                "INSERT INTO events (session_id, frame_id, wall_clock, kind,"
                " severity, detail_json) VALUES (?,?,?,?,?,?)",
                (
                    self.session_id,
                    frame_id,
                    time.strftime("%Y-%m-%dT%H:%M:%S"),
                    kind,
                    severity,
                    json.dumps(detail, default=str),
                ),
            )

    # ------------------------------------------------------------------ #
    def _uploader(self) -> None:
        batch: list[dict] = []
        last = time.time()
        while not self._stop.is_set():
            try:
                batch.append(self._q.get(timeout=0.25))
            except queue.Empty:
                pass
            if batch and (time.time() - last >= self.cfg.post_interval_s or len(batch) >= 64):
                self._flush(batch)
                batch = []
                last = time.time()
        if batch:
            self._flush(batch)

    def _flush(self, batch: list[dict]) -> None:
        payload = json.dumps(
            {"session_id": self.session_id, "meta": self.meta, "frames": batch},
            default=str,
        ).encode("utf-8")
        req = urllib.request.Request(
            self.cfg.post_url, data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                resp.read()
            self.posted += len(batch)
        except (urllib.error.URLError, OSError, TimeoutError):
            self.post_failures += 1   # offline is an expected state, not an error

    # ------------------------------------------------------------------ #
    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        with self._db_lock:
            self.conn.execute(
                "UPDATE sessions SET ended_at=? WHERE session_id=?",
                (time.strftime("%Y-%m-%dT%H:%M:%S"), self.session_id),
            )
            self.conn.commit()
            self.conn.close()
        if self._csv_file is not None:
            self._csv_file.close()

    def summary(self) -> dict:
        return {
            "session_id": self.session_id,
            "posted": self.posted,
            "dropped": self.dropped,
            "post_failures": self.post_failures,
        }
