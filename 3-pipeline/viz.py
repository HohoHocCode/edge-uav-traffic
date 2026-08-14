"""Overlay rendering for the on-device demo and for the recorded video."""

from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise SystemExit("opencv is required") from exc


# Distinct, colour-blind-safe-ish palette for 10 VisDrone classes (BGR).
PALETTE = [
    (86, 180, 233), (230, 159, 0), (0, 158, 115), (240, 228, 66),
    (0, 114, 178), (213, 94, 0), (204, 121, 167), (140, 140, 140),
    (120, 200, 255), (100, 220, 120),
]

LEVEL_COLOUR = {
    "normal": (110, 200, 110),
    "warn": (0, 200, 255),
    "alert": (60, 60, 235),
}


def colour_for(cls_id: int) -> tuple[int, int, int]:
    return PALETTE[int(cls_id) % len(PALETTE)]


def draw_tracks(
    frame: np.ndarray,
    tracks,
    class_names: dict[int, str],
    show_id: bool = True,
    show_trail: bool = True,
    show_conf: bool = False,
) -> np.ndarray:
    """Draw one box per track.

    The flags exist so the same frame can answer three different questions.
    Showing everything at once is the wrong default for a demo: an id beside
    every box is noise when the question is "what did the detector find", and
    a confidence beside every box is noise when the question is "did the
    identities hold".
    """
    for t in tracks:
        x1, y1, x2, y2 = (int(v) for v in t.box)
        c = colour_for(t.cls)
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)

        label = class_names.get(int(t.cls), str(t.cls))
        if show_id:
            label = f"{label} #{t.track_id}"
        if show_conf:
            label = f"{label} {float(t.conf):.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        # Keep the label inside the frame when the box touches the top edge.
        ly = y1 - 4 if y1 - th - 6 >= 0 else y2 + th + 6
        cv2.rectangle(frame, (x1, ly - th - 4), (x1 + tw + 4, ly + 2), c, -1)
        cv2.putText(frame, label, (x1 + 2, ly - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 20, 20), 1, cv2.LINE_AA)

        if show_trail and len(t.history) > 1:
            pts = np.asarray(t.history, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame, [pts], False, c, 1, cv2.LINE_AA)
    return frame


def draw_regions(frame: np.ndarray, rois, lines, show_rois: bool = True,
                 counts: dict | None = None) -> np.ndarray:
    """ROI boxes and counting lines.

    ``counts`` is the analytics ``line_crossings`` mapping. When given, the
    line is drawn thicker and captioned with its own tally, which is the whole
    point of the counting view: the number belongs next to the line it came
    from, not only in a panel on the far side of the screen.
    """
    h, w = frame.shape[:2]
    if show_rois:
        for r in rois:
            if r.name == "full_frame":
                continue
            x1, y1, x2, y2 = r.box
            cv2.rectangle(frame, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)),
                          (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(frame, r.name, (int(x1 * w) + 4, int(y1 * h) + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
    for ln in lines:
        x1, y1, x2, y2 = ln.seg
        p1 = (int(x1 * w), int(y1 * h))
        p2 = (int(x2 * w), int(y2 * h))
        thick = 3 if counts else 2
        cv2.line(frame, p1, p2, (255, 230, 120), thick, cv2.LINE_AA)
        if counts:
            d = counts.get(ln.name, {})
            txt = f"{ln.name}  {d.get('forward', 0)} -> / <- {d.get('backward', 0)}"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            ox, oy = p1[0], max(th + 8, p1[1] - 10)
            cv2.rectangle(frame, (ox - 4, oy - th - 6), (ox + tw + 6, oy + 6),
                          (30, 30, 30), -1)
            cv2.putText(frame, txt, (ox, oy), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 230, 120), 2, cv2.LINE_AA)
    return frame


def draw_hud(
    frame: np.ndarray,
    report,
    timings: dict,
    backend: str,
    condition: str | None = None,
) -> np.ndarray:
    """Top-left HUD: backend, the latency split, counts, congestion state."""
    h, w = frame.shape[:2]
    lines = [
        f"backend  {backend}",
        f"pre {timings.get('pre_ms', 0):5.1f} | infer {timings.get('infer_ms', 0):5.1f} "
        f"| post {timings.get('post_ms', 0):5.1f} | trk {timings.get('track_ms', 0):5.1f} ms",
        f"total    {timings.get('total_ms', 0):5.1f} ms  "
        f"({1000.0 / max(timings.get('total_ms', 1e-6), 1e-6):5.1f} FPS)",
        f"tracks   {report.n_tracks}   vehicles {report.vehicle_count}   "
        f"people {report.person_count}",
    ]
    if condition:
        lines.append(f"condition {condition}")

    pad, lh = 8, 18
    box_h = lh * len(lines) + pad
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (max(430, w // 3), box_h), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    for i, txt in enumerate(lines):
        cv2.putText(frame, txt, (pad, pad + lh * i + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (235, 235, 235), 1, cv2.LINE_AA)

    # Congestion badge, top-right.
    lvl = report.congestion_level
    col = LEVEL_COLOUR.get(lvl, (200, 200, 200))
    txt = lvl.upper()
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (w - tw - 24, 8), (w - 8, 16 + th + 8), col, -1)
    cv2.putText(frame, txt, (w - tw - 16, 16 + th),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (25, 25, 25), 2, cv2.LINE_AA)
    return frame
