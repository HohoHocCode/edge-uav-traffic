"""OpenCV drawing. Grouped rather than per-object, because the object count is the problem.

VisDrone aerial frames carry 200-400 objects. supervision's annotators loop in
Python once per detection, so the cost of *drawing* a frame lands in the same
order as the GPU forward pass -- the annotation, not the network, becomes the
bottleneck.

The fix is not micro-optimisation, it is batching the calls. ``cv2.polylines``
accepts a *list* of polylines and draws all of them inside one C++ call, so a
box is expressed as a closed 4-point polyline and boxes are grouped by colour.
400 boxes across 10 classes then cost 10 calls instead of 400.

Labels stay in a Python loop, but only boxes above ``min_label_px`` get one --
on aerial footage a label on a 12-pixel car is larger than the car and hides
the thing it names, so suppressing them is legibility as much as speed.

``--no-fast-draw`` switches to supervision's annotators instead; the two paths
are kept side by side so the fast one can be checked against the reference.
"""

from __future__ import annotations

import os
import sys

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise SystemExit("opencv is required: pip install opencv-python") from exc

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "3-pipeline"))

from viz import LEVEL_COLOUR, PALETTE, colour_for   # noqa: E402  (3-pipeline/viz.py)

FONT = cv2.FONT_HERSHEY_SIMPLEX


# --------------------------------------------------------------------------- #
def sv_palette():
    """``viz.PALETTE`` as a supervision ColorPalette.

    The repository's palette is BGR and colour-blind-safe, and it is already
    frozen against the 10 VisDrone class indices. Converting it rather than
    picking a new one keeps the fast path, the supervision path and the future
    web dashboard on one set of colours.
    """
    import supervision as sv

    return sv.ColorPalette(
        colors=[sv.Color(r=int(r), g=int(g), b=int(b)) for (b, g, r) in PALETTE]
    )


def palette_hex() -> list[str]:
    """The same palette as ``#rrggbb``, for the JSON the web app will read."""
    return [f"#{r:02x}{g:02x}{b:02x}" for (b, g, r) in PALETTE]


def _contours(xyxy: np.ndarray) -> list[np.ndarray]:
    """Boxes -> closed 4-point polylines, the form ``cv2.polylines`` batches."""
    b = xyxy.astype(np.int32)
    x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    pts = np.empty((len(b), 4, 2), np.int32)
    pts[:, 0, 0], pts[:, 0, 1] = x1, y1
    pts[:, 1, 0], pts[:, 1, 1] = x2, y1
    pts[:, 2, 0], pts[:, 2, 1] = x2, y2
    pts[:, 3, 0], pts[:, 3, 1] = x1, y2
    return list(pts)


# --------------------------------------------------------------------------- #
def draw_boxes(
    frame: np.ndarray,
    xyxy: np.ndarray,
    colour_key: np.ndarray,
    thickness: int = 1,
) -> np.ndarray:
    """Draw every box, grouped into one ``polylines`` call per colour.

    ``colour_key`` is whatever the caller wants colour to mean: class id for
    detection, track id for tracking.
    """
    if len(xyxy) == 0:
        return frame
    keys = np.asarray(colour_key).astype(np.int64) % len(PALETTE)
    conts = _contours(xyxy)
    for k in np.unique(keys):
        idx = np.flatnonzero(keys == k)
        cv2.polylines(
            frame, [conts[i] for i in idx], True, PALETTE[int(k)], thickness
        )
    return frame


def draw_labels(
    frame: np.ndarray,
    xyxy: np.ndarray,
    labels: list[str],
    colour_key: np.ndarray,
    min_label_px: int = 40,
    scale: float = 0.4,
) -> np.ndarray:
    """Label only the boxes big enough for the label to fit beside them."""
    if len(xyxy) == 0:
        return frame
    b = xyxy.astype(np.int32)
    big = np.flatnonzero(np.maximum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]) >= min_label_px)
    keys = np.asarray(colour_key).astype(np.int64) % len(PALETTE)
    h = frame.shape[0]
    for i in big:
        x1, y1, _, y2 = b[i]
        text = labels[i]
        c = PALETTE[int(keys[i])]
        (tw, th), _ = cv2.getTextSize(text, FONT, scale, 1)
        # Flip the chip below the box when the box is against the top edge,
        # the same rule viz.draw_tracks uses.
        ly = y1 - 3 if y1 - th - 5 >= 0 else min(y2 + th + 5, h - 2)
        cv2.rectangle(frame, (x1, ly - th - 3), (x1 + tw + 4, ly + 2), c, -1)
        cv2.putText(frame, text, (x1 + 2, ly - 1), FONT, scale, (20, 20, 20), 1, cv2.LINE_AA)
    return frame


def draw_traces(
    frame: np.ndarray,
    traces: dict[int, "np.ndarray | list"],
    colour_key: dict[int, int] | None = None,
    thickness: int = 1,
) -> np.ndarray:
    """Draw all track trails, again one ``polylines`` call per colour."""
    if not traces:
        return frame
    groups: dict[int, list[np.ndarray]] = {}
    for tid, pts in traces.items():
        if len(pts) < 2:
            continue
        k = int((colour_key.get(tid, tid) if colour_key else tid)) % len(PALETTE)
        groups.setdefault(k, []).append(
            np.asarray(pts, dtype=np.int32).reshape(-1, 1, 2)
        )
    for k, polys in groups.items():
        cv2.polylines(frame, polys, False, PALETTE[k], thickness, cv2.LINE_AA)
    return frame


def draw_dots(
    frame: np.ndarray, xy: np.ndarray, colour: tuple[int, int, int], radius: int = 2
) -> np.ndarray:
    for x, y in xy.astype(np.int32):
        cv2.circle(frame, (int(x), int(y)), radius, colour, -1, cv2.LINE_AA)
    return frame


# --------------------------------------------------------------------------- #
class Heatmap:
    """Accumulated density map.

    Built once and updated in place. Constructing it inside the frame loop --
    the mistake ``sv.HeatMapAnnotator`` invites -- resets the accumulator every
    frame, and a density map with no memory is just a scatter plot of the
    current frame.

    Decay is deliberate: an undecayed accumulator over 1560 frames saturates
    every road in the scene and stops distinguishing anything. With decay the
    map answers "where has traffic been *recently*", which is the question a
    congestion monitor actually asks.

    Everything below the blend runs at ``1/scale`` resolution. A density map is
    low-frequency by construction -- it is a wide Gaussian over point counts --
    so nothing survives full resolution that a quarter-scale map does not also
    carry, while a 51x51 blur over 1904x1070 float32 costs about 70 ms a frame
    and the same blur at quarter scale costs under two. That one change is the
    difference between this task running at 12 fps and at 45.
    """

    def __init__(
        self,
        shape: tuple[int, int],
        # In source pixels. 25 was far too tight: on a 1904x1070 frame it puts a
        # 20-pixel dot on each object, which reads as a marker rather than a
        # density field. 60 gives blobs that merge when objects cluster, which
        # is the entire point of a density map.
        radius: int = 60,
        decay: float = 0.97,
        opacity: float = 0.7,
        floor: float = 0.06,
        gain: float = 1.6,
        scale: int = 4,
        colormap: int = cv2.COLORMAP_TURBO,
    ) -> None:
        h, w = shape
        self.scale = max(1, int(scale))
        self.sh, self.sw = h // self.scale, w // self.scale
        self.acc = np.zeros((self.sh, self.sw), np.float32)
        self.decay = float(decay)
        self.opacity = float(opacity)
        self.floor = float(floor)
        self.colormap = colormap
        # Odd kernel, scaled with the accumulator so the blur covers the same
        # area of the original frame regardless of `scale`.
        k = max(3, int(radius * 2 / self.scale) | 1)
        self.ksize = k

        # Absolute colour scale, self-calibrated: blur one impulse held at its
        # steady state under `decay` and measure the peak it reaches. `cap` is
        # then "`gain` objects sitting on top of each other" in real units.
        #
        # Normalising against the *running* peak instead -- the obvious choice
        # -- makes the colour mean nothing: the busiest pixel is always full
        # red, so a deserted frame and a packed one render identically, and the
        # scale silently rescales itself between frames.
        imp = np.zeros((self.sh, self.sw), np.float32)
        imp[self.sh // 2, self.sw // 2] = 1.0 / max(1.0 - self.decay, 1e-6)
        self.cap = float(gain) * float(cv2.GaussianBlur(imp, (k, k), 0).max())
        self._dirty = True
        self._colour: np.ndarray | None = None
        self._a: np.ndarray | None = None
        self._inv: np.ndarray | None = None

    def warp(self, A: np.ndarray) -> None:
        """Slide the accumulator with the ground under a moving camera.

        Without this the map accumulates in image coordinates while the scene
        translates beneath it, so a stationary cluster paints a streak in the
        direction the drone flew -- the map would describe the flight path, not
        the crowd. `A` is the full-resolution frame-to-frame affine; only its
        translation needs rescaling to the accumulator's resolution.
        """
        M = A.astype(np.float32).copy()
        M[:, 2] /= self.scale
        self.acc = cv2.warpAffine(
            self.acc, M, (self.sw, self.sh),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
        )
        self._dirty = True

    def update(self, xy: np.ndarray) -> None:
        if self.decay < 1.0:
            self.acc *= self.decay
        if len(xy):
            pts = (np.asarray(xy, np.float32) / self.scale).astype(np.int32)
            xs = np.clip(pts[:, 0], 0, self.sw - 1)
            ys = np.clip(pts[:, 1], 0, self.sh - 1)
            # np.add.at accumulates duplicates correctly; plain fancy-index
            # assignment would keep only the last write at a repeated pixel.
            np.add.at(self.acc, (ys, xs), 1.0)
        self._dirty = True

    def render(self, frame: np.ndarray) -> np.ndarray:
        """Blend with a *per-pixel* alpha proportional to density.

        A hard mask plus a constant opacity was the obvious implementation and
        it looked wrong: every pixel that cleared the threshold got the full
        tint, so the sparse tail of the Gaussian painted wide flat smears in the
        colour map's bottom colour, and the result read as motion streaks rather
        than density. Scaling alpha with the value instead lets the map fade out
        at its own edges, and ``floor`` discards the tail outright.

        ``cv2.blendLinear`` does the per-pixel blend in one C++ pass (~4 ms at
        this resolution); the same expression in numpy costs several times that.
        """
        h, w = frame.shape[:2]
        if self._dirty:
            m = cv2.GaussianBlur(self.acc, (self.ksize, self.ksize), 0)
            if float(m.max()) <= 1e-6:
                self._colour = None
            else:
                # Absolute scale (see `cap`), then drop the Gaussian's sparse
                # tail so the map fades out instead of smearing.
                t = np.clip((m / self.cap - self.floor) / (1.0 - self.floor), 0.0, 1.0)
                norm = (t * 255.0).astype(np.uint8)
                # Upscale the single-channel map, then colour it. Colouring
                # first and upscaling the BGR result moves three times the bytes
                # through the resize for an identical image, since applyColorMap
                # is a per-pixel LUT either way.
                big = cv2.resize(norm, (w, h), interpolation=cv2.INTER_LINEAR)
                self._colour = cv2.applyColorMap(big, self.colormap)
                self._a = big.astype(np.float32) * (self.opacity / 255.0)
                self._inv = 1.0 - self._a
            self._dirty = False

        if self._colour is None:
            return frame
        # In place: the caller keeps drawing on `frame` afterwards.
        frame[:] = cv2.blendLinear(frame, self._colour, self._inv, self._a)
        return frame


# --------------------------------------------------------------------------- #
def draw_stats(
    frame: np.ndarray,
    lines: list[str],
    corner: str = "tl",
    alpha: float = 0.55,
    scale: float = 0.5,
    width: int | None = None,
) -> np.ndarray:
    """Temporary readout block.

    Deliberately plain, and deliberately easy to delete: the real dashboard
    chrome is a web app that will read the CSV/JSON this run writes. Everything
    shown here also exists in those files, so turning the block off with
    ``--no-stats`` loses no information.
    """
    if not lines:
        return frame
    h, w = frame.shape[:2]
    pad, lh = 10, int(round(24 * scale / 0.5))
    tw = max(cv2.getTextSize(t, FONT, scale, 1)[0][0] for t in lines)
    bw = width or min(tw + pad * 2, w - 20)
    bh = lh * len(lines) + pad * 2

    x0 = 10 if corner.endswith("l") else w - bw - 10
    y0 = 10 if corner.startswith("t") else h - bh - 10

    # Blend into a fresh array and assign back. Passing the sliced view as
    # cv2's `dst` would hand OpenCV a non-contiguous buffer, which it is free to
    # copy instead of writing through -- the darkening would then be silently
    # discarded.
    panel = frame[y0:y0 + bh, x0:x0 + bw]
    frame[y0:y0 + bh, x0:x0 + bw] = cv2.addWeighted(
        panel, 1.0 - alpha, np.full(panel.shape, 22, panel.dtype), alpha, 0
    )
    cv2.rectangle(frame, (x0, y0), (x0 + bw, y0 + bh), (60, 60, 60), 1)

    for i, text in enumerate(lines):
        cv2.putText(frame, text, (x0 + pad, y0 + pad + lh * i + int(lh * 0.75)),
                    FONT, scale, (235, 235, 235), 1, cv2.LINE_AA)
    return frame


def draw_badge(
    frame: np.ndarray, text: str, level: str = "normal", scale: float = 0.9
) -> np.ndarray:
    """Big congestion chip, top-right. Colours from ``viz.LEVEL_COLOUR``."""
    colour = LEVEL_COLOUR.get(level, (200, 200, 200))
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, 2)
    w = frame.shape[1]
    x1, y1 = w - tw - 34, 10
    cv2.rectangle(frame, (x1, y1), (w - 10, y1 + th + 18), colour, -1)
    cv2.putText(frame, text, (x1 + 12, y1 + th + 6), FONT, scale, (25, 25, 25), 2, cv2.LINE_AA)
    return frame


def draw_progress(frame: np.ndarray, frac: float, colour=(96, 165, 250)) -> np.ndarray:
    """3 px timeline along the bottom edge. Cheap, and makes scrubbing readable."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 3), (w, h), (40, 40, 40), -1)
    cv2.rectangle(frame, (0, h - 3), (int(w * max(0.0, min(1.0, frac))), h), colour, -1)
    return frame


__all__ = [
    "Heatmap", "colour_for", "draw_badge", "draw_boxes", "draw_dots", "draw_labels",
    "draw_progress", "draw_stats", "draw_traces", "palette_hex", "sv_palette",
    "LEVEL_COLOUR", "PALETTE",
]
