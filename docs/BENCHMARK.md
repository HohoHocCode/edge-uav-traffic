# Benchmark protocol

The purpose of this document is to make every number in the report
reproducible by someone who has only the repository and a board.

## The three axes

| Axis | Values | Answers |
|---|---|---|
| Backend | `onnxruntime-cpu`, `onnxruntime-qnn` (Hexagon HTP) | What does the NPU actually buy over the Kryo cores? |
| Condition | `clean` + 9 degradations | Does the model survive the weather it will fly in? |
| Load | 50 / 150 / 300 / 600 surviving candidates | Where does the CPU postprocessing tail bite? |

Resolution is fixed at 640 for the quality table. The checkpoint was fine-tuned
at 640, and evaluating it elsewhere measures the resize, not the model.

## Metrics, and why each one is separate

| Metric | Meaning | Why it is not merged with the others |
|---|---|---|
| `pre_ms` | letterbox, normalise, HWC→NCHW | Scales with source resolution, not model cost |
| `infer_ms` | the forward pass only | The only number attributable to the compute unit |
| `post_ms` | decode + NMS + rescale | Runs on the CPU and scales with scene density, not with FLOPs |
| `track_ms` | association + Kalman | CPU, scales with object count |
| `total_ms` | sum of the above | The only number a deployment cares about |
| `AP / AP50 / APs` | COCO-style, IoU .50:.05:.95 | `APs` is the primary metric: 69 % of VisDrone val objects are small |
| `AP_retention` | condition AP ÷ clean AP | Comparable across models in a way absolute AP is not |

**`post_ms` is not optional.** On a dense aerial frame NMS runs over hundreds of
surviving candidates on the Kryo cores. A benchmark that reports only
`infer_ms` describes a system nobody can deploy.

### The postprocessing cost flips sign with the confidence threshold

Measured on VisDrone val at the AP protocol (`conf 0.001`), heavy rain raises
`post_ms` from 20.0 to 37.1 (+85%): rain streaks manufacture high-frequency
structure that scores above 0.001, so more candidates reach NMS.

Measured on the demo clip at the deployment threshold (`conf 0.25`), the same
condition *lowers* NMS cost from 3.1 to 1.8 ms (−37%), because rain pushes
confidences below 0.25 and the boxes never reach NMS at all. Detections fall
64.7 → 30.1 per frame over the same footage.

Both numbers are real and they are not in conflict — they measure different
operating points. The consequence for deployment is the uncomfortable one:

> In heavy rain the pipeline gets **faster because it sees less**. Latency and
> FPS telemetry look healthier at precisely the moment the detector is failing,
> and because the vehicle count also falls, a congestion monitor reads
> "normal" when the truth is "cannot see".

A health signal for this system therefore cannot be built from latency alone.
Any statement about the latency cost of degradation must name the confidence
threshold it was measured at; this repository does so on every row.

## Scene density is the hidden variable

Postprocessing cost tracks the number of surviving candidates, which tracks
scene density. VisDrone val averages **70.7 annotated objects per image**, and
the busiest frames exceed 300. `bench_latency.py` therefore drives NMS at four
explicit candidate loads instead of timing it on an empty frame, so the
reported tail is one that actually occurs.

## Degradation conditions

Ten conditions, defined in `2-augment/degradations.py` and frozen in
`CONDITIONS`. Each is seeded from the image index, so the degraded set is
byte-identical across runs and machines — this is what makes the robustness
table reproducible rather than merely repeatable.

| id | Models |
|---|---|
| `clean` | reference |
| `rain_light` / `rain_medium` / `rain_heavy` | streaks + wet-lens veil + contrast loss |
| `bright_down` / `bright_down_heavy` | dusk, heavy overcast (gamma, not linear scaling) |
| `bright_up` | direct sun on a bright surface |
| `blur_light` / `blur_medium` | UAV translation / gimbal shake during exposure |
| `fog_medium` | atmospheric veil with low-frequency spatial structure |

Severities were fixed **before** any results were seen, and calibrated so the
ladder is monotone and physically plausible — heavy rain shifts frame mean by
about +23 grey levels and drops contrast; blur leaves the mean unchanged and
drops standard deviation from 18.8 to 7.1. Conditions that destroy the image
were rejected, because a frame an operator would discard is not an operating
condition.

## Ignore regions — the policy that moves AP

VisDrone annotations contain two non-object categories:

- category `0` — **ignored region**: an area that is deliberately unannotated
- category `11` — **others**: a real object outside the 10-class taxonomy

The common conversion scripts drop both silently. The consequence is that every
detection landing in one of those areas is scored as a false positive, which
depresses AP by an amount that varies per image. VisDrone val contains **1410
ignore regions across 548 images**.

This repository keeps them and exposes the choice:

```
--ignore-policy mask   # drop detections >50% inside an ignore region (default)
--ignore-policy keep   # count them as false positives (what the naive scripts do)
```

Every result row records which policy produced it. Neither is "wrong"; quoting
a number without saying which one is.

Membership uses intersection over **detection area**, not IoU: a small detection
entirely inside a large ignored region has negligible IoU but should still be
excluded.

## Reproducing the tables

```bash
# latency ladder — no weights, no dataset needed
python 4-bench/bench_latency.py --model models/yolov8n_visdrone_640.onnx \
    --backends onnxruntime-cpu onnxruntime-qnn --iters 100 \
    --out results/latency.csv

# quality + robustness, all ten conditions, both ignore policies
python 4-bench/bench_quality.py --model models/yolov8n_visdrone_640.onnx \
    --all-conditions --ignore-policy mask --out results/quality_mask.csv
python 4-bench/bench_quality.py --model models/yolov8n_visdrone_640.onnx \
    --all-conditions --ignore-policy keep --out results/quality_keep.csv
```

Each row carries `model_sha256_16`, `backend`, `imgsz`, `conf_thres`,
`iou_thres`, `condition`, `ignore_policy` and `n_images`.

`conf_thres` defaults to `0.001` for AP measurement, which is the COCO
convention — a higher threshold truncates the precision-recall curve and
inflates nothing while quietly lowering AP. The deployment threshold (0.25) is
a separate row, not a substitute.

## What this benchmark does not establish

- **No architecture comparison.** One detector family. Statements about YOLOv8n
  versus anything else are not supported by this data.
- **No tracking accuracy.** No MOT ground truth is evaluated. The tracker is
  exercised, not scored. The words IDF1, IDSW and MOTA do not appear in the
  results.
- **No absolute leaderboard comparison.** The evaluator here is a faithful but
  independent implementation of the COCO definition, not `pycocotools`. Numbers
  are comparable within this report.
- **No watt figures unless the board exposes a rail.** `probe_power.py`
  reports absence as absence.
