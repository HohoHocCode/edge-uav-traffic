"""YOLOv8 detector: letterbox -> session -> decode -> NMS.

Latency is reported as three separate numbers and never as one:

    pre_ms    letterbox + normalise + HWC->NCHW      (CPU)
    infer_ms  the forward pass                       (NPU or CPU)
    post_ms   decode + NMS + rescale to source       (CPU)

The split is the point. A detector whose ``infer_ms`` is 3 ms and whose
``post_ms`` is 14 ms is not a 3 ms detector, and only the split makes that
visible.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise SystemExit("opencv is required: pip install opencv-python-headless") from exc


@dataclass
class Detections:
    """Detections in *source image* coordinates."""

    xyxy: np.ndarray          # (N, 4) float32
    conf: np.ndarray          # (N,)   float32
    cls: np.ndarray           # (N,)   int32
    pre_ms: float = 0.0
    infer_ms: float = 0.0
    post_ms: float = 0.0

    def __len__(self) -> int:
        return int(self.xyxy.shape[0])

    @property
    def total_ms(self) -> float:
        return self.pre_ms + self.infer_ms + self.post_ms

    @staticmethod
    def empty() -> "Detections":
        return Detections(
            np.zeros((0, 4), np.float32),
            np.zeros((0,), np.float32),
            np.zeros((0,), np.int32),
        )


# --------------------------------------------------------------------------- #
def letterbox(
    img: np.ndarray, new_shape: int = 640, pad_value: int = 114
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Resize preserving aspect ratio, pad to square. Returns (img, gain, (padw, padh))."""
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    if (nh, nw) != (h, w):
        interp = cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA
        img = cv2.resize(img, (nw, nh), interpolation=interp)
    padh, padw = (new_shape - nh) / 2, (new_shape - nw) / 2
    top, bottom = int(round(padh - 0.1)), int(round(padh + 0.1))
    left, right = int(round(padw - 0.1)), int(round(padw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=(pad_value, pad_value, pad_value),
    )
    return img, r, (left, top)


def nms_numpy(
    boxes: np.ndarray, scores: np.ndarray, iou_thres: float
) -> np.ndarray:
    """Plain greedy NMS. Returns kept indices, highest score first."""
    if boxes.size == 0:
        return np.zeros((0,), np.int64)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_thres]
    return np.asarray(keep, dtype=np.int64)


# --------------------------------------------------------------------------- #
class Yolov8Detector:
    """YOLOv8-family detector over an :class:`OrtSession`."""

    def __init__(
        self,
        session,
        imgsz: int = 640,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        max_det: int = 300,
        pad_value: int = 114,
        class_agnostic_nms: bool = False,
    ) -> None:
        self.session = session
        self.imgsz = int(imgsz)
        self.conf_thres = float(conf_thres)
        self.iou_thres = float(iou_thres)
        self.max_det = int(max_det)
        self.pad_value = int(pad_value)
        self.class_agnostic_nms = bool(class_agnostic_nms)

        # Detected on the first real forward pass.
        self.nc: int | None = None

    # ------------------------------------------------------------------ #
    def preprocess(self, img: np.ndarray) -> tuple[np.ndarray, float, tuple[float, float]]:
        lb, gain, pad = letterbox(img, self.imgsz, self.pad_value)
        x = lb[:, :, ::-1]                        # BGR -> RGB
        x = np.ascontiguousarray(x.transpose(2, 0, 1), dtype=np.float32)
        x /= 255.0
        return x[None], gain, pad

    # ------------------------------------------------------------------ #
    def postprocess(
        self, raw: np.ndarray, gain: float, pad: tuple[float, float], src_shape: tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decode a YOLOv8 head output of shape (1, 4+nc, A) or (1, A, 4+nc)."""
        p = raw[0] if raw.ndim == 3 else raw
        # Normalise orientation to (A, 4+nc): the anchor axis is the long one.
        if p.shape[0] < p.shape[1]:
            p = p.transpose(1, 0)

        nc = p.shape[1] - 4
        if nc <= 0:
            raise ValueError(
                f"unexpected head output shape {raw.shape}; "
                "expected (1, 4+nc, anchors)"
            )
        self.nc = nc

        boxes_cxcywh = p[:, :4]
        scores_all = p[:, 4:]

        cls = scores_all.argmax(axis=1)
        conf = scores_all[np.arange(scores_all.shape[0]), cls]

        m = conf >= self.conf_thres
        if not m.any():
            return (
                np.zeros((0, 4), np.float32),
                np.zeros((0,), np.float32),
                np.zeros((0,), np.int32),
            )
        boxes_cxcywh, conf, cls = boxes_cxcywh[m], conf[m], cls[m]

        # cxcywh -> xyxy, still in letterboxed pixels
        xy = boxes_cxcywh[:, :2]
        wh = boxes_cxcywh[:, 2:4]
        xyxy = np.concatenate([xy - wh / 2.0, xy + wh / 2.0], axis=1)

        # NMS, per class unless explicitly class-agnostic. An offset trick keeps
        # it a single pass while preventing cross-class suppression.
        if self.class_agnostic_nms:
            keep = nms_numpy(xyxy, conf, self.iou_thres)
        else:
            offset = cls.astype(np.float32) * (self.imgsz * 4.0)
            keep = nms_numpy(xyxy + offset[:, None], conf, self.iou_thres)
        keep = keep[: self.max_det]
        xyxy, conf, cls = xyxy[keep], conf[keep], cls[keep]

        # letterboxed pixels -> source pixels
        padw, padh = pad
        xyxy[:, [0, 2]] -= padw
        xyxy[:, [1, 3]] -= padh
        xyxy /= gain
        h, w = src_shape
        np.clip(xyxy[:, [0, 2]], 0, w, out=xyxy[:, [0, 2]])
        np.clip(xyxy[:, [1, 3]], 0, h, out=xyxy[:, [1, 3]])

        return xyxy.astype(np.float32), conf.astype(np.float32), cls.astype(np.int32)

    # ------------------------------------------------------------------ #
    def __call__(self, img: np.ndarray) -> Detections:
        t0 = time.perf_counter()
        x, gain, pad = self.preprocess(img)
        t1 = time.perf_counter()

        outputs = self.session.run(x)
        infer_ms = self.session.last_infer_ms

        t2 = time.perf_counter()
        xyxy, conf, cls = self.postprocess(outputs[0], gain, pad, img.shape[:2])
        t3 = time.perf_counter()

        return Detections(
            xyxy=xyxy,
            conf=conf,
            cls=cls,
            pre_ms=(t1 - t0) * 1000.0,
            infer_ms=infer_ms,
            post_ms=(t3 - t2) * 1000.0,
        )

    def warmup(self, n: int = 5) -> None:
        self.session.warmup((1, 3, self.imgsz, self.imgsz), n=n)
