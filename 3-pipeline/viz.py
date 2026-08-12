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
) -> np.ndarray:
    for t in tracks:
        x1, y1, x2, y2 = (int(v) for v in t.box)
        c = colour_for(t.cls)
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)

        label = class_names.get(int(t.cls), str(t.cls))
        if show_id:
            label = f"{label} #{t.track_id}"
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


def draw_regions(frame: np.ndarray, rois, lines) -> np.ndarray:
    h, w = frame.shape[:2]
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
        cv2.line(frame, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)),
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
