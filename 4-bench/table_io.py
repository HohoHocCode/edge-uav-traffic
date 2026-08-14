"""Appending rows to a results CSV without silently corrupting it.

``csv.DictWriter`` writes values in *its* field order and skips the header when
the file already exists. Point two scripts with different columns at one path and
the second one's rows land under the first one's header, shifted -- a file that
parses cleanly and means something entirely different from what it says. This
happened here: ``bench_tracking.py`` appended 40 rows under a header written by
``score_mot.py``, and the corruption was invisible until a column was read back
by name.

So the header is compared before appending, and a mismatch is an error with the
diff spelled out. A benchmark that quietly reports the wrong number is worse than
one that refuses to run.
"""

from __future__ import annotations

import csv
import os


def append_rows(path: str, rows: list[dict]) -> int:
    """Append ``rows`` to ``path``, writing the header if the file is new.

    Raises ``SystemExit`` if the existing header does not match the keys of the
    rows being written, naming the offending columns and suggesting a new file.
    """
    if not rows:
        return 0

    fieldnames = list(rows[0].keys())
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    exists = os.path.exists(path) and os.path.getsize(path) > 0

    if exists:
        with open(path, "r", newline="", encoding="utf-8") as f:
            existing = next(csv.reader(f), [])
        if existing != fieldnames:
            missing = [c for c in existing if c not in fieldnames]
            extra = [c for c in fieldnames if c not in existing]
            raise SystemExit(
                f"[fatal] {path} has a different schema; appending would shift "
                f"every column.\n"
                f"        file has {len(existing)} columns, these rows have "
                f"{len(fieldnames)}\n"
                + (f"        only in file: {missing}\n" if missing else "")
                + (f"        only in rows: {extra}\n" if extra else "")
                + (f"        same columns, different order\n"
                   if not missing and not extra else "")
                + f"        write to a different --out, or delete {path}"
            )

    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerows(rows)
    return len(rows)
