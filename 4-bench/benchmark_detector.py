"""Common detector construction for quality benchmarks.

The deployment benchmarks use a frozen ONNX graph, while model-comparison
runs often need to score the exact training checkpoint.  This module presents
both as the small ``Yolov8Detector`` interface already consumed by the metric
scripts: image in, source-coordinate boxes plus split latency out.
"""

from __future__ import annotations

import os
import time

import numpy as np
import cv2

from detector import Detections, Yolov8Detector
from runtime import create_session


class UltralyticsDetector:
    """Ultralytics ``.pt`` detector with the repository's result interface."""

    def __init__(
        self,
        weights: str,
        imgsz: int,
        conf_thres: float,
        iou_thres: float,
        max_det: int,
        device: str = "0",
        half: bool = True,
    ) -> None:
        try:
            import torch
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover
            raise SystemExit("a .pt benchmark requires torch and ultralytics") from exc

        if device != "cpu" and not torch.cuda.is_available():
            print("[warn] CUDA unavailable; using CPU and fp32")
            device, half = "cpu", False
        if device == "cpu":
            half = False

        self.model = YOLO(weights)
        self.imgsz = int(imgsz)
        self.conf_thres = float(conf_thres)
        self.iou_thres = float(iou_thres)
        self.max_det = int(max_det)
        self.device = device
        self.half = bool(half)
        self.backend_name = "ultralytics-cuda" if device != "cpu" else "ultralytics-cpu"
        self.providers_active = [self.backend_name]
        self.class_names = {int(key): str(value) for key, value in self.model.names.items()}

    def _predict(self, image: np.ndarray):
        return self.model.predict(
            image,
            imgsz=self.imgsz,
            conf=self.conf_thres,
            iou=self.iou_thres,
            max_det=self.max_det,
            device=self.device,
            quantize=16 if self.half else None,
            verbose=False,
        )[0]

    def warmup(self, n: int = 5, image: np.ndarray | None = None) -> None:
        # Ultralytics uses rectangular inference for a single non-square image.
        # Warm up that actual shape; a square dummy would leave the first real
        # frame paying CUDA algorithm selection and poison the latency average.
        dummy = (
            image
            if image is not None
            else np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        )
        for _ in range(n):
            self._predict(dummy)

    def __call__(self, image: np.ndarray) -> Detections:
        wall_start = time.perf_counter()
        result = self._predict(image)
        wall_ms = (time.perf_counter() - wall_start) * 1000.0

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            xyxy = np.zeros((0, 4), np.float32)
            confidence = np.zeros((0,), np.float32)
            classes = np.zeros((0,), np.int32)
        else:
            xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32)
            confidence = boxes.conf.detach().cpu().numpy().astype(np.float32)
            classes = boxes.cls.detach().cpu().numpy().astype(np.int32)

        speed = getattr(result, "speed", None) or {}
        pre_ms = float(speed.get("preprocess", 0.0))
        infer_ms = float(speed.get("inference", wall_ms))
        post_ms = float(speed.get("postprocess", 0.0))
        measured = pre_ms + infer_ms + post_ms
        # Account for host transfer/framework overhead that Ultralytics does not
        # place in its three internal buckets.  This keeps total_ms honest while
        # retaining the useful inference split.
        if wall_ms > measured:
            post_ms += wall_ms - measured

        return Detections(
            xyxy=xyxy,
            conf=confidence,
            cls=classes,
            pre_ms=pre_ms,
            infer_ms=infer_ms,
            post_ms=post_ms,
        )


def create_benchmark_detector(
    model: str,
    backend: str,
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
    device: str = "0",
    half: bool = True,
):
    """Return ``(detector, backend_name, providers_active, class_names)``."""
    use_pt = os.path.splitext(model)[1].lower() == ".pt"
    if backend == "ultralytics" or (backend == "auto" and use_pt):
        detector = UltralyticsDetector(
            model,
            imgsz=imgsz,
            conf_thres=conf,
            iou_thres=iou,
            max_det=max_det,
            device=device,
            half=half,
        )
        return (
            detector,
            detector.backend_name,
            detector.providers_active,
            detector.class_names,
        )
    if use_pt:
        raise ValueError(".pt weights require --backend auto or ultralytics")

    session = create_session(model, backend=backend)
    detector = Yolov8Detector(
        session,
        imgsz=imgsz,
        conf_thres=conf,
        iou_thres=iou,
        max_det=max_det,
    )
    return detector, session.backend_name, session.info.providers_active, {}


def warmup_benchmark_detector(detector, image_path: str, n: int = 5) -> None:
    """Warm up with a real source shape for PT, static tensor for ONNX."""
    if isinstance(detector, UltralyticsDetector):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"warmup image is unreadable: {image_path}")
        detector.warmup(n=n, image=image)
    else:
        detector.warmup(n=n)
