"""Constant-velocity Kalman filter for axis-aligned boxes (SORT parameterisation).

State is ``[cx, cy, s, r, vcx, vcy, vs]`` where ``s`` is box area and ``r`` is
aspect ratio. Aspect ratio is modelled as constant (no velocity term) because a
car seen from a UAV changes area as the drone climbs but barely changes shape,
and giving ``r`` a velocity mostly lets boxes wander.

Pure numpy: no filterpy, no scipy. On the Kryo cores a 7x7 solve is nothing,
and the dependency would have to be cross-installed on the board.
"""

from __future__ import annotations

import numpy as np


def xyxy_to_z(box: np.ndarray) -> np.ndarray:
    """[x1,y1,x2,y2] -> [cx, cy, s, r] measurement vector."""
    w = max(float(box[2]) - float(box[0]), 1e-3)
    h = max(float(box[3]) - float(box[1]), 1e-3)
    cx = float(box[0]) + w / 2.0
    cy = float(box[1]) + h / 2.0
    return np.array([cx, cy, w * h, w / h], dtype=np.float64).reshape(4, 1)


def z_to_xyxy(z: np.ndarray) -> np.ndarray:
    """[cx, cy, s, r] -> [x1,y1,x2,y2]. Guards against a negative area."""
    cx, cy, s, r = (float(v) for v in z[:4].reshape(-1))
    s = max(s, 1e-6)
    r = max(r, 1e-6)
    w = np.sqrt(s * r)
    h = s / w if w > 0 else 0.0
    return np.array(
        [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], dtype=np.float32
    )


class KalmanBoxTracker:
    """7-state constant-velocity filter over a single box."""

    def __init__(self, box: np.ndarray) -> None:
        ndim = 7

        # State transition: position += velocity, velocity constant.
        self.F = np.eye(ndim, dtype=np.float64)
        self.F[0, 4] = 1.0
        self.F[1, 5] = 1.0
        self.F[2, 6] = 1.0

        # We observe [cx, cy, s, r] directly.
        self.H = np.zeros((4, ndim), dtype=np.float64)
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = self.H[3, 3] = 1.0

        # Measurement noise: aspect ratio is the noisiest thing a detector
        # gives us on a 20-pixel object, so it gets the loosest trust.
        self.R = np.diag([1.0, 1.0, 10.0, 10.0]).astype(np.float64)

        # Process noise: velocities drift more than positions.
        self.Q = np.eye(ndim, dtype=np.float64)
        self.Q[4:, 4:] *= 0.01
        self.Q[2, 2] *= 0.01

        # Initial covariance: velocities are entirely unknown at birth.
        self.P = np.eye(ndim, dtype=np.float64) * 10.0
        self.P[4:, 4:] *= 1000.0

        self.x = np.zeros((ndim, 1), dtype=np.float64)
        self.x[:4] = xyxy_to_z(box)

    # ------------------------------------------------------------------ #
    def predict(self) -> np.ndarray:
        # A negative predicted area is physically meaningless and makes the
        # sqrt in z_to_xyxy produce NaN, which then poisons every IoU it
        # touches. Clamp the shrink velocity instead.
        if self.x[6, 0] + self.x[2, 0] <= 0:
            self.x[6, 0] = 0.0

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return z_to_xyxy(self.x)

    # ------------------------------------------------------------------ #
    def update(self, box: np.ndarray) -> None:
        z = xyxy_to_z(box)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        try:
            K = self.P @ self.H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return  # singular innovation: skip the correction, keep the prior
        self.x = self.x + K @ y
        I_KH = np.eye(self.P.shape[0]) - K @ self.H
        self.P = I_KH @ self.P

    # ------------------------------------------------------------------ #
    def get_state(self) -> np.ndarray:
        return z_to_xyxy(self.x)

    @property
    def velocity(self) -> tuple[float, float]:
        return float(self.x[4, 0]), float(self.x[5, 0])
