"""Global motion compensation: separate what the drone did from what the traffic did.

This clip is shot from a moving UAV. Measured over 400 frames the camera
translates a mean of 4.9 px and a 95th percentile of 18.6 px per frame, ~1957 px
cumulative -- most of the frame width. Every consequence of that is a silent
wrong answer rather than an error:

* **Line crossings are fabricated.** A parked car sweeps across a screen-fixed
  counting line because the drone panned over it. The gate counts traffic that
  never moved.
* **Trails are drawn on stationary objects**, which reads as tracker failure
  when the tracker is doing exactly the right thing.
* **Association degrades.** ByteTrack matches a Kalman prediction to a detection
  by IoU, and the prediction assumes a static camera. At 18-22 px of camera
  motion a small object's predicted and observed boxes barely overlap, so the
  identity is dropped and re-created.
* **The density map smears**, because it accumulates in image coordinates while
  the ground slides underneath.

The fix is one similarity transform per frame, estimated with sparse optical
flow on background corners, then applied to everything that carries state.

Why a *similarity* (4 DoF: translation, rotation, uniform scale) rather than a
full homography: the scene is close to planar and viewed near-nadir, the
per-frame motion is small, and a homography estimated from a few hundred noisy
corners on a mostly-flat ground plane is badly conditioned -- it fits the
perspective terms to noise and produces occasional wild warps. The measured
per-frame rotation (0.12 deg) and scale (1.001) are small enough that the extra
degrees of freedom would buy nothing here.

Cost is ~2.7 ms/frame at ``scale 0.25`` with 200 corners, which measured the
same mean displacement as 400 corners at ``scale 0.5`` for a third of the time.
"""

from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise SystemExit("opencv is required: pip install opencv-python") from exc


class GlobalMotion:
    """Frame-to-frame camera motion as a 2x3 affine, in *source* pixels.

    ``update`` returns ``A`` such that a static scene point satisfies
    ``p_cur ~= A @ [p_prev, 1]``. It returns ``None`` on the first frame and
    whenever the estimate cannot be trusted -- too few tracked corners, or too
    few RANSAC inliers. ``None`` means "assume no camera motion", which is the
    honest fallback: a bad transform corrupts every track it touches, whereas
    skipping one frame's compensation costs a few pixels.
    """

    def __init__(
        self,
        scale: float = 0.25,
        max_corners: int = 200,
        quality: float = 0.01,
        min_distance: int = 10,
        min_inliers: int = 12,
    ) -> None:
        self.scale = float(scale)
        self.max_corners = int(max_corners)
        self.quality = float(quality)
        self.min_distance = int(min_distance)
        self.min_inliers = int(min_inliers)

        self._prev: np.ndarray | None = None
        self.n_inliers = 0
        self.n_failed = 0
        self.dx = self.dy = 0.0
        self.zoom = 1.0
        self.rot_deg = 0.0

    # ------------------------------------------------------------------ #
    def update(self, frame: np.ndarray) -> np.ndarray | None:
        gray = cv2.cvtColor(
            cv2.resize(frame, None, fx=self.scale, fy=self.scale,
                       interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2GRAY,
        )
        prev, self._prev = self._prev, gray
        if prev is None:
            return self._no_motion()

        p0 = cv2.goodFeaturesToTrack(
            prev, maxCorners=self.max_corners, qualityLevel=self.quality,
            minDistance=self.min_distance, blockSize=7,
        )
        if p0 is None or len(p0) < self.min_inliers:
            return self._no_motion(failed=True)

        p1, status, _ = cv2.calcOpticalFlowPyrLK(
            prev, gray, p0, None, winSize=(15, 15), maxLevel=3
        )
        ok = status.ravel() == 1
        if ok.sum() < self.min_inliers:
            return self._no_motion(failed=True)

        # RANSAC is what makes this work on a scene that contains moving
        # vehicles: those corners are outliers to the camera's own motion, and a
        # least-squares fit would let a bus drag the estimate.
        A, inliers = cv2.estimateAffinePartial2D(
            p0[ok], p1[ok], method=cv2.RANSAC, ransacReprojThreshold=3.0,
            maxIters=2000, confidence=0.99,
        )
        if A is None or inliers is None or int(inliers.sum()) < self.min_inliers:
            return self._no_motion(failed=True)

        # Estimated on the downscaled image: the linear part is scale-invariant,
        # the translation is not.
        A = A.astype(np.float64).copy()
        A[:, 2] /= self.scale

        self.n_inliers = int(inliers.sum())
        self.dx, self.dy = float(A[0, 2]), float(A[1, 2])
        self.zoom = float(np.hypot(A[0, 0], A[1, 0]))
        self.rot_deg = float(np.degrees(np.arctan2(A[1, 0], A[0, 0])))
        return A

    def _no_motion(self, failed: bool = False) -> None:
        if failed:
            self.n_failed += 1
        self.n_inliers = 0
        self.dx = self.dy = 0.0
        self.zoom = 1.0
        self.rot_deg = 0.0
        return None

    # ------------------------------------------------------------------ #
    @property
    def translation_px(self) -> float:
        return float(np.hypot(self.dx, self.dy))

    def stats(self) -> dict:
        return {
            "cam_dx": round(self.dx, 3),
            "cam_dy": round(self.dy, 3),
            "cam_px": round(self.translation_px, 3),
            "cam_zoom": round(self.zoom, 5),
            "cam_rot_deg": round(self.rot_deg, 4),
            "cam_inliers": self.n_inliers,
        }


# --------------------------------------------------------------------------- #
def warp_tracks(tracks, A: np.ndarray) -> None:
    """Move every track's Kalman state into the current frame's coordinates.

    Called *before* the tracker's own update, so the constant-velocity
    prediction happens in a frame where the camera has already been accounted
    for. Without this the predicted box sits where the object *was on screen*,
    the IoU against the new detection collapses under fast camera motion, and
    the tracker answers with a new identity.

    The state is ``[cx, cy, s, r, vcx, vcy, vs]``:

    * position is a point, so it takes the full affine including translation
    * velocity is a *difference* of positions, so it takes the linear part only
      -- adding the translation would make a stationary object appear to
      accelerate with the drone
    * area ``s`` scales with the square of the linear scale; aspect ``r`` is
      left alone, consistent with ``kalman.py`` modelling it as constant (the
      measured 0.12 deg/frame of roll does not meaningfully change it)
    """
    M = A[:, :2]                      # rotation * scale
    t = A[:, 2]
    s2 = float(np.linalg.det(M))      # area scale factor
    for tr in tracks:
        x = tr.kf.x
        cx, cy = float(x[0, 0]), float(x[1, 0])
        x[0, 0] = M[0, 0] * cx + M[0, 1] * cy + t[0]
        x[1, 0] = M[1, 0] * cx + M[1, 1] * cy + t[1]

        vx, vy = float(x[4, 0]), float(x[5, 0])
        x[4, 0] = M[0, 0] * vx + M[0, 1] * vy
        x[5, 0] = M[1, 0] * vx + M[1, 1] * vy

        x[2, 0] *= s2
        x[6, 0] *= s2

        # Position uncertainty is in the same rotated/scaled frame as the
        # position. Leaving it untransformed would slowly bias the gain.
        P = tr.kf.P
        P[:2, :2] = M @ P[:2, :2] @ M.T

        # The trail is a list of past *screen* positions. Drawn as stored, it
        # shows where the object was in frames that no longer line up with this
        # one, so a straight drive renders as a curve bent by the drone's path.
        # Carrying it forward through the same transform keeps every point in
        # current-frame coordinates, and the trail then traces the object's
        # actual path across the ground.
        h = tr.history
        if h:
            pts = np.asarray(h, dtype=np.float64)
            pts = pts @ M.T + t
            tr.history[:] = [(float(x), float(y)) for x, y in pts]


def moving_mask(tracks, min_speed: float = 1.2) -> np.ndarray:
    """Which tracks are actually moving, once the camera is compensated out.

    After :func:`warp_tracks` the Kalman velocity is expressed in a frame that
    follows the ground, so it is the object's own speed rather than its apparent
    speed on screen. That makes the test a plain threshold instead of anything
    involving the camera.

    ``min_speed`` is in source pixels per frame. Below roughly one pixel the
    signal is Kalman jitter on a parked vehicle, which is precisely what needs
    to be excluded from the crossing count.
    """
    if not tracks:
        return np.zeros((0,), bool)
    v = np.array([tr.kf.velocity for tr in tracks], dtype=np.float32)
    return np.hypot(v[:, 0], v[:, 1]) >= float(min_speed)
