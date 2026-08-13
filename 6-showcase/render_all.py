#!/usr/bin/env python3
"""Run all three task renderers in one go, sharing the same flags.

The engine is rebuilt per task rather than shared. That costs a few seconds of
model load each time and buys process-level isolation: an out-of-memory or a
CUDA fault in one task cannot leave the next one running against a poisoned
context, and each task's reported latency is measured from its own warmup.
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import render_counting      # noqa: E402
import render_detection     # noqa: E402
import render_tracking      # noqa: E402

TASKS = (
    ("task2 detection", render_detection),
    ("task4 tracking", render_tracking),
    ("task5 counting", render_counting),
)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if any(a == "--out" or a.startswith("--out=") for a in argv):
        print("[fatal] --out is per-task here; run the individual renderers "
              "if you need to name the files")
        return 2

    failed = []
    for name, mod in TASKS:
        print(f"\n{'=' * 70}\n[run] {name}\n{'=' * 70}")
        try:
            rc = mod.main(argv)
        except SystemExit as exc:
            rc = exc.code if isinstance(exc.code, int) else 1
            print(f"[fail] {name}: {exc}")
        except Exception as exc:                       # keep going
            rc = 1
            print(f"[fail] {name}: {type(exc).__name__}: {exc}")
        if rc:
            failed.append(name)

    print(f"\n{'=' * 70}")
    if failed:
        print(f"[fail] {len(failed)}/{len(TASKS)} failed: {', '.join(failed)}")
        return 1
    print(f"[done] all {len(TASKS)} renders complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
