"""Deterministic image degradations for the robustness axis.

Every degradation is a pure function ``(BGR uint8 image, severity) -> BGR uint8``
and is seeded from the image index, so the same source frame always produces the
same degraded frame from one run to the next. That determinism is what makes the
robustness table reproducible.

Scope of that guarantee: bit-exact for a given OpenCV build, deterministic but
not necessarily bit-exact across builds, because ``GaussianBlur``, the resize
kernels and float32 reductions may take different SIMD paths. Pin the OpenCV
version alongside the seed if a result must be reproduced exactly.

The severities are frozen in ``CONDITIONS`` below. Do not tune them after
looking at results.
"""

from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "opencv-python is required: pip install opencv-python-headless"
    ) from exc


# --------------------------------------------------------------------------- #
# Rain
# --------------------------------------------------------------------------- #
def rain(img: np.ndarray, severity: str = "medium", seed: int = 0) -> np.ndarray:
    """Synthetic rain: streak layer + wet-lens veil + mild contrast loss.

    Physically motivated rather than pretty: rain reduces contrast and adds
    high-frequency bright streaks, which is exactly what breaks a detector
    tuned on clean aerial footage.
    """
    params = {
        # (n_drops per megapixel, length px, angle deg, thickness, veil alpha, contrast)
        "light": (900, 12, -12, 1, 0.04, 0.95),
        "medium": (2200, 18, -16, 1, 0.09, 0.89),
        "heavy": (4200, 26, -20, 1, 0.15, 0.81),
    }
    if severity not in params:
        raise ValueError(f"rain severity must be one of {list(params)}, got {severity!r}")
    density, length, angle, thick, veil, contrast = params[severity]

    h, w = img.shape[:2]
    rng = np.random.default_rng(seed)
    n = int(density * (h * w) / 1_000_000)

    layer = np.zeros((h, w), dtype=np.uint8)
    xs = rng.integers(0, w, size=n)
    ys = rng.integers(0, h, size=n)
    rad = np.deg2rad(angle)
    dx = int(round(length * np.sin(rad)))
    dy = int(round(length * np.cos(rad)))
    for x, y in zip(xs.tolist(), ys.tolist()):
        cv2.line(layer, (x, y), (x + dx, y + dy), 255, thick, lineType=cv2.LINE_AA)

    # Streaks are motion-blurred along their own direction so they read as rain,
    # not as scratches.
    layer = cv2.blur(layer, (3, 3))
    rain_bgr = cv2.cvtColor(layer, cv2.COLOR_GRAY2BGR)

    out = img.astype(np.float32)
    # Contrast loss towards the scene mean (atmospheric scattering).
    mean = out.mean()
    out = (out - mean) * contrast + mean
    # Uniform bright veil.
    out = out * (1.0 - veil) + 255.0 * veil * 0.6
    # Additive streaks.
    out = out + rain_bgr.astype(np.float32) * 0.55
    return np.clip(out, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Brightness
# --------------------------------------------------------------------------- #
def brightness(img: np.ndarray, severity: str = "down", seed: int = 0) -> np.ndarray:
    """Gamma-based exposure change.

    Gamma rather than a linear scale, because a linear scale clips and a real
    sensor does not. ``down`` models dusk / heavy overcast, ``up`` models
    direct sun on a bright surface.
    """
    params = {
        "down_light": (1.40, 0.93),
        "down": (1.90, 0.85),
        "down_heavy": (2.40, 0.80),
        "up_light": (0.80, 1.05),
        "up": (0.65, 1.12),
        "up_heavy": (0.50, 1.22),
    }
    if severity not in params:
        raise ValueError(
            f"brightness severity must be one of {list(params)}, got {severity!r}"
        )
    gamma, gain = params[severity]

    lut = np.arange(256, dtype=np.float32) / 255.0
    lut = np.power(lut, gamma) * gain
    lut = np.clip(lut * 255.0, 0, 255).astype(np.uint8)
    return cv2.LUT(img, lut)


# --------------------------------------------------------------------------- #
# Motion blur
# --------------------------------------------------------------------------- #
def motion_blur(img: np.ndarray, severity: str = "light", seed: int = 0) -> np.ndarray:
    """Directional blur modelling UAV translation / gimbal shake during exposure.

    Kept deliberately mild: a UAV frame that is heavily blurred is one an
    operator would discard, so blurring past ~15 px is not a realistic
    operating condition.
    """
    params = {
        "light": (5, 15.0),
        "medium": (9, 25.0),
        "heavy": (15, 35.0),
    }
    if severity not in params:
        raise ValueError(
            f"motion_blur severity must be one of {list(params)}, got {severity!r}"
        )
    ksize, angle = params[severity]

    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    kernel[ksize // 2, :] = 1.0
    rot = cv2.getRotationMatrix2D((ksize / 2 - 0.5, ksize / 2 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, rot, (ksize, ksize))
    s = kernel.sum()
    if s <= 0:  # degenerate rotation
        kernel = np.zeros((ksize, ksize), dtype=np.float32)
        kernel[ksize // 2, :] = 1.0
        s = kernel.sum()
    kernel /= s
    return cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE)


# --------------------------------------------------------------------------- #
# Fog / haze (bonus condition — cheap and very common over a city at dawn)
# --------------------------------------------------------------------------- #
def fog(img: np.ndarray, severity: str = "medium", seed: int = 0) -> np.ndarray:
    """Depth-agnostic atmospheric veil with low-frequency spatial structure."""
    params = {"light": 0.18, "medium": 0.34, "heavy": 0.52}
    if severity not in params:
        raise ValueError(f"fog severity must be one of {list(params)}, got {severity!r}")
    beta = params[severity]

    h, w = img.shape[:2]
    rng = np.random.default_rng(seed)
    # Low-res noise upsampled = smooth, cloud-like transmission map.
    small = rng.random((max(2, h // 64), max(2, w // 64))).astype(np.float32)
    tmap = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    tmap = cv2.GaussianBlur(tmap, (0, 0), sigmaX=max(w, h) / 40.0)
    tmap = (tmap - tmap.min()) / (np.ptp(tmap) + 1e-6)
    t = 1.0 - beta * (0.6 + 0.4 * tmap)  # transmission in [1-beta, 1]
    t = t[..., None]

    airlight = 235.0
    out = img.astype(np.float32) * t + airlight * (1.0 - t)
    return np.clip(out, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Frozen condition set
# --------------------------------------------------------------------------- #
FUNCS = {
    "rain": rain,
    "brightness": brightness,
    "motion_blur": motion_blur,
    "fog": fog,
}

#: The exact conditions reported in the robustness table. ``clean`` is the
#: reference row and applies no transform.
CONDITIONS: list[tuple[str, str | None, str | None]] = [
    # (condition_id, function name, severity)
    ("clean", None, None),
    ("rain_light", "rain", "light"),
    ("rain_medium", "rain", "medium"),
    ("rain_heavy", "rain", "heavy"),
    ("bright_down", "brightness", "down"),
    ("bright_down_heavy", "brightness", "down_heavy"),
    ("bright_up", "brightness", "up"),
    ("blur_light", "motion_blur", "light"),
    ("blur_medium", "motion_blur", "medium"),
    ("fog_medium", "fog", "medium"),
]

CONDITION_IDS = [c[0] for c in CONDITIONS]


def apply_condition(img: np.ndarray, condition_id: str, seed: int = 0) -> np.ndarray:
    """Apply a named condition from ``CONDITIONS``. ``clean`` is a no-op copy."""
    for cid, fname, severity in CONDITIONS:
        if cid != condition_id:
            continue
        if fname is None:
            return img.copy()
        return FUNCS[fname](img, severity, seed=seed)
    raise KeyError(
        f"unknown condition {condition_id!r}; known: {CONDITION_IDS}"
    )
