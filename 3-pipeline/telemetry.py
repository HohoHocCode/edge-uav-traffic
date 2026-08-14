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
    #: Rows between SQLite commits. An uncommitted transaction is lost on
    #: power failure, which on a drone is the expected way for a run to end.
    commit_every: int = 30
    #: Seconds between commits when frames arrive slower than commit_every.
    commit_interval_s: float = 1.0


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
        # WAL survives an abrupt process death far better than the rollback
        # journal, and lets the commit cadence below stay cheap.
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
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
        #: Dashboard task the overlay should follow ("2"/"4"/"5"), refreshed
        #: from each /ingest reply. Plain attribute assignment on a str is
        #: atomic under the GIL, so the render loop reads it without a lock.
        self.view = "4"
        self._db_lock = threading.Lock()
        self._uncommitted = 0
        self._last_commit = time.time()

        # CSV — the header is locked on the first row, so the first row must
        # already carry every column the session will ever produce.
        # TrafficAnalytics guarantees that by seeding all classes, ROIs and
        # lines at zero; the guard in write() catches it if that ever stops
        # being true, rather than silently dropping the column.
        self._csv_file = open(cfg.csv_path, "w", newline="", encoding="utf-8") \
            if cfg.csv_path else None
        self._csv_writer: csv.DictWriter | None = None
        self._csv_fields: list[str] = []
        self.dropped_columns: set[str] = set()

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
    def write(self, report, timings: dict, extra: dict | None = None) -> None:
        """Append one frame. ``extra`` carries non-timing scalars.

        It exists so a caller can record something the analytics report cannot
        know -- the raw detection count, for instance, which is measured before
        the tracker runs and is therefore not a property of the tracked report.
        Unlike ``timings`` these are not coerced to rounded floats.
        """
        row = report.to_row()
        row.update({k: round(float(v), 3) for k, v in timings.items()})
        if extra:
            row.update(extra)
        row["session_id"] = self.session_id

        if self._csv_file is not None:
            if self._csv_writer is None:
                self._csv_fields = list(row.keys())
                self._csv_writer = csv.DictWriter(
                    self._csv_file, fieldnames=self._csv_fields,
                    extrasaction="ignore",
                )
                self._csv_writer.writeheader()
            else:
                # extrasaction="ignore" is what makes a late-appearing column
                # vanish without a trace. Record it instead: the CSV is the
                # benchmark's raw record and losing a column silently would
                # corrupt the evidence rather than merely the file.
                extra = set(row) - set(self._csv_fields)
                if extra:
                    self.dropped_columns |= extra
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
            self._uncommitted += 1
            now = time.time()
            if (self._uncommitted >= self.cfg.commit_every
                    or now - self._last_commit >= self.cfg.commit_interval_s):
                self.conn.commit()
                self._uncommitted = 0
                self._last_commit = now

        if self.cfg.post_url:
            try:
                self._q.put_nowait(row)
            except queue.Full:
                # Uplink is slower than the pipeline. Discard the *oldest*
                # queued row and enqueue this one: fresh state matters more
                # than a complete stream, and the full history is on disk
                # either way. Simply failing the put would drop the newest
                # row instead, which is the opposite of what we want.
                try:
                    self._q.get_nowait()
                    self.dropped += 1
                    self._q.put_nowait(row)
                except (queue.Empty, queue.Full):
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
            # Events are rare and are the thing an operator reviews after an
            # incident, so they are committed immediately rather than waiting
            # for the frame cadence.
            self.conn.commit()
            self._uncommitted = 0
            self._last_commit = time.time()

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
                body = resp.read()
            self.posted += len(batch)
            # The command post answers with the task the operator is looking
            # at, so the board can draw the matching overlay. Riding on a reply
            # the node already waits for costs no extra connection and no
            # polling; the price is that a tab change only takes effect on the
            # next flush, i.e. within post_interval_s.
            try:
                view = json.loads(body).get("view")
            except (TypeError, ValueError, AttributeError):
                view = None
            if view in ("2", "4", "5"):
                self.view = view
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
        d = {
            "session_id": self.session_id,
            "posted": self.posted,
            "dropped": self.dropped,
            "post_failures": self.post_failures,
        }
        if self.dropped_columns:
            # Never silent: a missing CSV column invalidates the record.
            d["DROPPED_CSV_COLUMNS"] = sorted(self.dropped_columns)
        return d
