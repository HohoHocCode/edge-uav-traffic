"""Detection engines, both reduced to one interface: frames in, sv.Detections out.

Two backends:

``ultralytics``  the fine-tuned ``.pt`` on CUDA, fp16, batched. This is the fast
                 path and the default. torch takes a dynamic input shape, so
                 ``imgsz`` is a genuine runtime knob -- 1280 on aerial footage
                 recovers small objects that 640 loses entirely.

``onnx``         the frozen ``models/yolov8n_visdrone_640.onnx`` through the
                 repository's own :class:`Yolov8Detector`. Locked to 640 by the
                 export (``dynamic: False``) and batch 1. Kept because it is the
                 path the board runs and the path every number in
                 ``docs/RESULTS.md`` was measured on -- without it there is no
                 way to check that the fast path agrees with the reported one.

Latency is reported the way the rest of this repository reports it: ``pre`` /
``infer`` / ``post`` separately, never as a single number. A detector with a
3 ms forward pass and a 15 ms NMS tail is not a 3 ms detector.
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import supervision as sv

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "3-pipeline"))


# --------------------------------------------------------------------------- #
@dataclass
class Timing:
    """Per-frame latency split, in milliseconds."""

    pre_ms: float = 0.0
    infer_ms: float = 0.0
    post_ms: float = 0.0
    track_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.pre_ms + self.infer_ms + self.post_ms + self.track_ms

    @property
    def fps(self) -> float:
        return 1000.0 / max(self.total_ms, 1e-6)


@dataclass
class EngineInfo:
    name: str
    weights: str
    weights_sha16: str
    device: str
    imgsz: int
    half: bool
    batch: int
    conf: float
    iou: float
    max_det: int
    class_names: dict[int, str] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "engine": self.name,
            "weights": os.path.basename(self.weights),
            "weights_sha256_16": self.weights_sha16,
            "device": self.device,
            "imgsz": self.imgsz,
            "half": self.half,
            "batch": self.batch,
            "conf": self.conf,
            "iou": self.iou,
            "max_det": self.max_det,
        }

    def summary(self) -> str:
        return (
            f"{self.name} | {os.path.basename(self.weights)} ({self.weights_sha16}) | "
            f"{self.device} | imgsz {self.imgsz} | "
            f"{'fp16' if self.half else 'fp32'} | batch {self.batch} | "
            f"conf {self.conf} iou {self.iou}"
        )


def sha256_short(path: str, n: int = 16) -> str:
    """First `n` hex chars of the file's sha256.

    Same convention as ``4-bench/bench_quality.py``: every emitted row carries
    the hash of the artefact that produced it, because a number that cannot name
    its model cannot be compared to anything.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


# --------------------------------------------------------------------------- #
class UltralyticsEngine:
    """YOLOv8 ``.pt`` on CUDA. Batched, optionally fp16."""

    name = "ultralytics"

    def __init__(
        self,
        weights: str,
        imgsz: int = 1280,
        conf: float = 0.25,
        iou: float = 0.45,
        max_det: int = 300,
        device: str = "0",
        half: bool = True,
        batch: int = 8,
    ) -> None:
        try:
            import torch
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover
            raise SystemExit(
                "ultralytics + torch are required for --engine ultralytics.\n"
                "  uv sync"
            ) from exc

        self._torch = torch
        if device != "cpu" and not torch.cuda.is_available():
            print("[warn] CUDA not available; falling back to CPU. This will be slow.")
            device, half = "cpu", False
        # fp16 on the CPU is emulated and slower than fp32, never faster.
        if device == "cpu" and half:
            print("[warn] --half ignored on CPU")
            half = False

        self.model = YOLO(weights)
        self.device = device
        self.half = bool(half)
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.max_det = int(max_det)
        self.batch = int(batch)
        self._gmc_hooked = False
        self._last_warp = None

        names = getattr(self.model, "names", None) or {}
        self.class_names = {int(k): str(v) for k, v in names.items()}

        self.info = EngineInfo(
            name=self.name,
            weights=weights,
            weights_sha16=sha256_short(weights),
            device=(
                f"cuda:{device} ({torch.cuda.get_device_name(int(device))})"
                if device != "cpu" else "cpu"
            ),
            imgsz=self.imgsz,
            half=self.half,
            batch=self.batch,
            conf=self.conf,
            iou=self.iou,
            max_det=self.max_det,
            class_names=self.class_names,
        )

    # ------------------------------------------------------------------ #
    def warmup(self, shape: tuple[int, int], n: int = 3) -> None:
        """Run a few throwaway batches.

        The first CUDA call pays kernel autotuning and cuDNN algorithm
        selection. Timing without a warmup measures that once and calls it the
        steady-state cost, which overstates it by an order of magnitude.
        """
        h, w = shape
        dummy = [np.zeros((h, w, 3), np.uint8)] * self.batch
        for _ in range(n):
            self._predict(dummy)

    def _predict(self, frames: list[np.ndarray]):
        return self.model.predict(
            frames,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            max_det=self.max_det,
            device=self.device,
            # ultralytics 8.4 replaced the boolean `half` with `quantize`, which
            # names the bit width; the predictor selects fp16 on `quantize == 16`.
            quantize=16 if self.half else None,
            verbose=False,
        )

    # ------------------------------------------------------------------ #
    def infer_batch(
        self, frames: list[np.ndarray]
    ) -> tuple[list[sv.Detections], list[Timing]]:
        if not frames:
            return [], []

        t0 = time.perf_counter()
        results = self._predict(frames)
        if self.device != "cpu":
            # predict() already moves results to host, but synchronise anyway so
            # the wall clock cannot be shortened by work still queued on the GPU.
            self._torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - t0) * 1000.0 / len(frames)

        dets, timings = [], []
        for r in results:
            dets.append(sv.Detections.from_ultralytics(r))
            # Ultralytics reports its own three-way split per image. Prefer it,
            # but reconcile against the wall clock: the difference is the batch
            # overhead its profiler does not see, and it belongs somewhere.
            sp = getattr(r, "speed", None) or {}
            pre = float(sp.get("preprocess", 0.0))
            inf = float(sp.get("inference", 0.0))
            post = float(sp.get("postprocess", 0.0))
            acc = pre + inf + post
            if acc <= 0.0:
                pre, inf, post = 0.0, wall_ms, 0.0
            elif wall_ms > acc:
                inf += wall_ms - acc
            timings.append(Timing(pre, inf, post))
        return dets, timings

    # ------------------------------------------------------------------ #
    def _hook_gmc(self) -> None:
        """Record the camera warp the tracker computes, instead of recomputing it.

        BoT-SORT, Deep OC-SORT and TrackTrack already run sparse optical flow
        every frame and warp their Kalman state with the result. The renderer
        needs the same transform for its trails, and there are two ways to get
        it: estimate it again with ``gmc.GlobalMotion`` (~3 ms/frame, and a
        second estimate that can disagree with the tracker's), or read back the
        one that was just used. This reads it back -- free, and guaranteed to be
        the transform the identities were actually associated with.

        ``GMC.apply`` returns a 2x3 affine mapping previous-frame coordinates to
        current-frame coordinates, with translation already rescaled out of its
        internal downscale, so it can be applied to full-resolution points as
        is. Trackers without camera compensation (ByteTrack, plain OC-SORT) have
        no ``gmc`` attribute; there is nothing to hook and ``last_warp`` stays
        None, which the caller reads as "no compensation available".
        """
        if self._gmc_hooked:
            return
        predictor = getattr(self.model, "predictor", None)
        trackers = getattr(predictor, "trackers", None) or []
        if not trackers:
            return                      # nothing built yet; try again next frame

        self._gmc_hooked = True
        gmc = getattr(trackers[0], "gmc", None)
        if gmc is None or getattr(gmc, "method", None) in (None, "none"):
            return

        inner = gmc.apply

        def recording_apply(*a, **kw):
            warp = inner(*a, **kw)
            self._last_warp = warp
            return warp

        gmc.apply = recording_apply

    @property
    def last_warp(self):
        """The most recent camera warp, or None if the tracker computes none."""
        return self._last_warp

    def track_frame(self, frame: np.ndarray, tracker: str) -> tuple[sv.Detections, Timing]:
        """Detect and track one frame using Ultralytics' persistent tracker.

        Tracking is deliberately frame-sequential. ``persist=True`` tells
        Ultralytics that consecutive calls belong to the same video; batching
        independent frames through a stateful MOT tracker would make ordering
        implicit and fragile.
        """
        # Cleared per frame, never carried over: Ultralytics falls back to an
        # identity warp if optical flow throws, and reusing the previous frame's
        # transform would compensate for motion that did not happen.
        self._last_warp = None

        t0 = time.perf_counter()
        results = self.model.track(
            frame,
            persist=True,
            tracker=tracker,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            max_det=self.max_det,
            device=self.device,
            quantize=16 if self.half else None,
            verbose=False,
        )
        if self.device != "cpu":
            self._torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - t0) * 1000.0
        if len(results) != 1:
            raise RuntimeError(f"tracker returned {len(results)} results for one frame")

        self._hook_gmc()

        result = results[0]
        sp = getattr(result, "speed", None) or {}
        pre = float(sp.get("preprocess", 0.0))
        inf = float(sp.get("inference", 0.0))
        post = float(sp.get("postprocess", 0.0))
        measured = pre + inf + post
        if measured <= 0.0:
            inf, track = wall_ms, 0.0
        else:
            # Ultralytics profiles detector stages, but tracker callbacks and
            # Python dispatch sit outside that split. Keep this overhead visible
            # rather than incorrectly charging it to model inference.
            track = max(0.0, wall_ms - measured)
        return sv.Detections.from_ultralytics(result), Timing(pre, inf, post, track)


# --------------------------------------------------------------------------- #
class OnnxEngine:
    """The repository's own ONNX path. Batch 1, imgsz fixed by the export."""

    name = "onnx"

    def __init__(
        self,
        weights: str,
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.45,
        max_det: int = 300,
        device: str = "cpu",
        half: bool = False,
        batch: int = 1,
        backend: str = "auto",
        pad_value: int = 114,
    ) -> None:
        from detector import Yolov8Detector          # 3-pipeline/detector.py
        from runtime import create_session           # 3-pipeline/runtime/__init__.py

        self.session = create_session(weights, backend=backend)
        self.det = Yolov8Detector(
            self.session,
            imgsz=imgsz,
            conf_thres=conf,
            iou_thres=iou,
            max_det=max_det,
            pad_value=pad_value,
        )
        self.imgsz = int(imgsz)
        self.batch = 1

        import yaml
        with open(os.path.join(ROOT, "configs", "visdrone.yaml"), encoding="utf-8") as f:
            self.class_names = {
                int(k): str(v) for k, v in yaml.safe_load(f)["names"].items()
            }

        self.info = EngineInfo(
            name=self.name,
            weights=weights,
            weights_sha16=sha256_short(weights),
            device=f"{self.session.backend_name} "
                   f"({', '.join(self.session.info.providers_active)})",
            imgsz=self.imgsz,
            half=False,
            batch=1,
            conf=conf,
            iou=iou,
            max_det=max_det,
            class_names=self.class_names,
        )

    def warmup(self, shape: tuple[int, int], n: int = 3) -> None:
        self.det.warmup(n=n)

    def infer_batch(
        self, frames: list[np.ndarray]
    ) -> tuple[list[sv.Detections], list[Timing]]:
        dets, timings = [], []
        for frame in frames:
            d = self.det(frame)
            dets.append(
                sv.Detections(
                    xyxy=d.xyxy.astype(np.float32).reshape(-1, 4),
                    confidence=d.conf.astype(np.float32).reshape(-1),
                    class_id=d.cls.astype(int).reshape(-1),
                )
            )
            timings.append(Timing(d.pre_ms, d.infer_ms, d.post_ms))
        return dets, timings


# --------------------------------------------------------------------------- #
def build_engine(args, mcfg: dict):
    """Construct the engine named by ``args.engine``.

    CLI flags override the YAML, never the reverse -- the same rule
    ``3-pipeline/run_pipeline.py`` follows.
    """
    conf = args.conf if args.conf is not None else float(mcfg.get("conf_thres", 0.25))
    iou = args.iou if args.iou is not None else float(mcfg.get("iou_thres", 0.45))
    max_det = int(mcfg.get("max_det", 300))

    if args.engine == "ultralytics":
        weights = args.model or os.path.join(ROOT, "models", "yolov8n_visdrone.pt")
        if not os.path.isfile(weights):
            raise SystemExit(f"[fatal] weights not found: {weights}")
        return UltralyticsEngine(
            weights, imgsz=args.imgsz, conf=conf, iou=iou, max_det=max_det,
            device=args.device, half=args.half, batch=args.batch,
        )

    weights = args.model or os.path.join(ROOT, str(mcfg.get("weights", "")))
    if not os.path.isfile(weights):
        raise SystemExit(f"[fatal] weights not found: {weights}")
    # The shipped ONNX is a static 1x3x640x640 graph; any other imgsz would be
    # rejected by the runtime with a shape error that says nothing useful.
    imgsz = int(mcfg.get("imgsz", 640))
    if args.imgsz != imgsz:
        print(f"[warn] --imgsz {args.imgsz} ignored: the ONNX graph is fixed at {imgsz}")
    return OnnxEngine(
        weights, imgsz=imgsz, conf=conf, iou=iou, max_det=max_det,
        backend=str(mcfg.get("backend", "auto")),
        pad_value=int(mcfg.get("pad_value", 114)),
    )
