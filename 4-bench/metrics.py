"""COCO-style detection metrics in pure numpy.

Written out rather than taking pycocotools because the whole robustness table
is "the same 548 images, ten times", and a dependency that needs a C compiler
is a bad thing to discover on an aarch64 board at hour 30. The implementation
follows the COCO definition closely enough that the numbers are comparable
*within this report*; it is not a drop-in replacement for the official
evaluator, and the report says so.

Implemented:
    AP        mean over IoU 0.50:0.05:0.95, 101-point interpolation
    AP50      IoU 0.50
    AP75      IoU 0.75
    APs/m/l   by GT area: <32^2, 32^2..96^2, >96^2

Matching rule (COCO): detections sorted by score descending; each detection
takes the highest-IoU unmatched GT of the same class above the IoU threshold.
Area range filtering applies to the GT, and detections matched to
out-of-range GT are ignored rather than counted as false positives.
"""

from __future__ import annotations

import numpy as np

IOU_THRS = np.arange(0.50, 0.96, 0.05)
REC_THRS = np.linspace(0.0, 1.0, 101)

AREA_RANGES = {
    "all": (0.0, 1e10),
    "small": (0.0, 32.0 ** 2),
    "medium": (32.0 ** 2, 96.0 ** 2),
    "large": (96.0 ** 2, 1e10),
}


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    area_a = np.maximum(0.0, a[:, 2] - a[:, 0]) * np.maximum(0.0, a[:, 3] - a[:, 1])
    area_b = np.maximum(0.0, b[:, 2] - b[:, 0]) * np.maximum(0.0, b[:, 3] - b[:, 1])
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:4], b[None, :, 2:4])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-12)


def _box_area(b: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, b[:, 2] - b[:, 0]) * np.maximum(0.0, b[:, 3] - b[:, 1])


class DetectionEvaluator:
    """Accumulate per-image predictions and ground truth, then compute AP.

    ``add(image_id, pred_boxes, pred_scores, pred_classes, gt_boxes, gt_classes)``
    """

    def __init__(self, num_classes: int) -> None:
        self.num_classes = int(num_classes)
        self._records: list[dict] = []

    def add(
        self,
        image_id,
        pred_boxes: np.ndarray,
        pred_scores: np.ndarray,
        pred_classes: np.ndarray,
        gt_boxes: np.ndarray,
        gt_classes: np.ndarray,
    ) -> None:
        self._records.append(
            {
                "image_id": image_id,
                "pb": np.asarray(pred_boxes, dtype=np.float64).reshape(-1, 4),
                "ps": np.asarray(pred_scores, dtype=np.float64).reshape(-1),
                "pc": np.asarray(pred_classes, dtype=np.int64).reshape(-1),
                "gb": np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 4),
                "gc": np.asarray(gt_classes, dtype=np.int64).reshape(-1),
            }
        )

    # ------------------------------------------------------------------ #
    def _ap_for(self, cls_id: int, area_rng: tuple[float, float]) -> np.ndarray:
        """Precision-interpolated AP per IoU threshold for one class. -1 if no GT."""
        lo, hi = area_rng

        scores_all, match_all = [], []
        n_gt_total = 0

        for rec in self._records:
            gm = rec["gc"] == cls_id
            gb = rec["gb"][gm]
            g_area = _box_area(gb)
            in_rng = (g_area >= lo) & (g_area < hi)
            n_gt_total += int(in_rng.sum())

            pm = rec["pc"] == cls_id
            pb, ps = rec["pb"][pm], rec["ps"][pm]
            if pb.shape[0] == 0:
                continue

            order = np.argsort(-ps)
            pb, ps = pb[order], ps[order]

            ious = iou_xyxy(pb, gb)                      # (n_pred, n_gt)
            # matched[t, d] = 1 tp, 0 fp, -1 ignore
            matched = np.zeros((len(IOU_THRS), pb.shape[0]), dtype=np.int8)

            if gb.shape[0] == 0:
                d_area = _box_area(pb)
                matched[:, :] = np.where(
                    (d_area >= lo) & (d_area < hi), 0, -1
                )[None, :]
            else:
                # The original implementation looped over ten thresholds and
                # every detection in Python.  VID has millions of hypotheses,
                # making that correct reference implementation need hours.
                # Keep the greedy detection order but associate all thresholds
                # together: one Python iteration per detection, with the ten
                # threshold states carried in a small vector.
                n_thr = len(IOU_THRS)
                gt_taken = np.zeros((n_thr, gb.shape[0]), dtype=bool)
                threshold_rows = np.arange(n_thr)
                d_area = _box_area(pb)
                outside_value = np.where(
                    (d_area >= lo) & (d_area < hi), 0, -1
                ).astype(np.int8)

                for di in range(pb.shape[0]):
                    candidates = np.broadcast_to(
                        ious[di], (n_thr, gb.shape[0])
                    ).copy()
                    candidates[gt_taken] = -1.0
                    best_gt = np.argmax(candidates, axis=1)
                    best_iou = candidates[threshold_rows, best_gt]
                    accepted = best_iou >= IOU_THRS

                    matched[:, di] = outside_value[di]
                    if accepted.any():
                        accepted_rows = threshold_rows[accepted]
                        accepted_gt = best_gt[accepted]
                        matched[accepted_rows, di] = np.where(
                            in_rng[accepted_gt], 1, -1
                        )
                        gt_taken[accepted_rows, accepted_gt] = True

            scores_all.append(ps)
            match_all.append(matched)

        if n_gt_total == 0:
            return np.full(len(IOU_THRS), -1.0)
        if not scores_all:
            return np.zeros(len(IOU_THRS))

        scores = np.concatenate(scores_all)
        matches = np.concatenate(match_all, axis=1)
        order = np.argsort(-scores)
        matches = matches[:, order]

        ap = np.zeros(len(IOU_THRS))
        for ti in range(len(IOU_THRS)):
            m = matches[ti]
            valid = m >= 0
            m = m[valid]
            tp = np.cumsum(m == 1)
            fp = np.cumsum(m == 0)
            rec = tp / max(n_gt_total, 1)
            prec = tp / np.maximum(tp + fp, 1e-12)

            # Make precision monotonically decreasing (COCO does this).
            prec = np.maximum.accumulate(prec[::-1])[::-1]
            idx = np.searchsorted(rec, REC_THRS, side="left")
            q = np.zeros_like(REC_THRS)
            ok = idx < len(prec)
            q[ok] = prec[idx[ok]]
            ap[ti] = q.mean()
        return ap

    # ------------------------------------------------------------------ #
    def evaluate(self) -> dict:
        out: dict[str, float] = {}
        per_class_all = {}

        for area_name, rng in AREA_RANGES.items():
            aps = []
            for c in range(self.num_classes):
                ap_iou = self._ap_for(c, rng)
                if ap_iou[0] < 0:      # class absent in this area range
                    continue
                aps.append(ap_iou)
                if area_name == "all":
                    per_class_all[c] = float(ap_iou.mean())
            if not aps:
                out[f"AP_{area_name}"] = float("nan")
                continue
            arr = np.stack(aps)                      # (n_cls, n_iou)
            out[f"AP_{area_name}"] = float(arr.mean())
            if area_name == "all":
                out["AP50"] = float(arr[:, 0].mean())
                out["AP75"] = float(arr[:, 5].mean())

        return {
            "AP": out.get("AP_all", float("nan")),
            "AP50": out.get("AP50", float("nan")),
            "AP75": out.get("AP75", float("nan")),
            "APs": out.get("AP_small", float("nan")),
            "APm": out.get("AP_medium", float("nan")),
            "APl": out.get("AP_large", float("nan")),
            "per_class_AP": per_class_all,
            "n_images": len(self._records),
        }
