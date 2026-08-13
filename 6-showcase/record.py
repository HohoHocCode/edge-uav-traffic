"""Per-frame record: CSV for tooling, JSON for the web app.

The video is the disposable half of this pipeline. The readout burned into the
frames is temporary -- the real dashboard is a web app, and it will read these
two files. So the schema, not the overlay, is the thing to get right here.

Two rules carried over from the rest of the repository:

**Every row is seeded with the full key set on frame 0.** ``FrameReport`` does
this deliberately (``analytics.py:176``): a consumer that locks its schema on
the first row -- ``csv.DictWriter`` does -- permanently loses the columns for
any class absent from that frame.

**Nothing is emitted without provenance.** ``meta`` carries the model hash,
engine, device, resolution and thresholds. A count of 240 vehicles measured at
imgsz 1280 is not the same measurement as 240 at 640, and a file that does not
say which one it is cannot be compared to anything.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import time

from draw import palette_hex


class Recorder:
    """Streams CSV as frames complete; holds the compact record for the JSON."""

    def __init__(self, out_mp4: str, meta: dict) -> None:
        base = os.path.splitext(out_mp4)[0]
        self.csv_path = base + ".csv"
        self.json_path = base + ".json"
        os.makedirs(os.path.dirname(os.path.abspath(base)), exist_ok=True)

        self.meta = dict(meta)
        self.meta["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.meta["python"] = platform.python_version()
        self.meta["versions"] = _versions()

        self._fh = open(self.csv_path, "w", newline="", encoding="utf-8")
        self._writer: csv.DictWriter | None = None
        self._fields: list[str] = []
        self._dropped: set[str] = set()
        self.frames: list[dict] = []
        self.n = 0

    # ------------------------------------------------------------------ #
    def add(self, row: dict) -> None:
        if self._writer is None:
            self._fields = list(row.keys())
            self._writer = csv.DictWriter(self._fh, fieldnames=self._fields)
            self._writer.writeheader()
        else:
            # A key that appears only on a later frame would be silently
            # dropped by DictWriter. Say so once rather than 1500 times.
            extra = set(row) - set(self._fields)
            for k in extra - self._dropped:
                print(f"[warn] column '{k}' appeared after frame 0 and is not in the CSV")
                self._dropped.add(k)
            row = {k: row.get(k, 0) for k in self._fields}

        self._writer.writerow(row)
        self.frames.append(row)
        self.n += 1

    # ------------------------------------------------------------------ #
    def close(self, class_names: dict[int, str]) -> None:
        self._fh.close()
        colours = palette_hex()
        payload = {
            "meta": self.meta,
            "classes": [
                {
                    "id": int(i),
                    "name": str(n),
                    "color": colours[int(i) % len(colours)],
                }
                for i, n in sorted(class_names.items())
            ],
            "frames": self.frames,
        }
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
        print(f"[ok] {self.csv_path}  ({self.n} rows)")
        print(f"[ok] {self.json_path}")

    # ------------------------------------------------------------------ #
    def check_alignment(self, n_video_frames: int) -> None:
        """The row count must match the frame count.

        This is the test for frame ordering. Batched inference plus a threaded
        reader and writer makes it easy to emit frames out of order, and every
        stateful stage downstream -- track identity, line crossings, congestion
        hysteresis -- degrades silently when that happens rather than raising.
        A count mismatch is the cheap symptom that catches it.
        """
        if abs(n_video_frames - self.n) > 2:
            print(f"[warn] frame/row mismatch ({n_video_frames} video vs {self.n} rows); "
                  f"the record may not line up with the video")


def _versions() -> dict:
    out = {}
    for mod in ("torch", "ultralytics", "supervision", "cv2", "numpy", "onnxruntime"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "?")
        except Exception:
            pass
    try:
        import torch
        if torch.cuda.is_available():
            out["cuda"] = torch.version.cuda
    except Exception:
        pass
    return out
