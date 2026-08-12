# Pipeline architecture

```
                 ┌──────────── on the aircraft / edge box ────────────┐
                 │                                                    │
  camera ───► decode ──► letterbox ──► YOLOv8n ──► NMS ──► ByteTrack ──► analytics
   RTSP           CPU        CPU        NPU/HTP     CPU       CPU          CPU
   file                                                                     │
                 │                                              telemetry ◄─┘
                 │                                               │      │
                 └───────────────────────────────────────────────┼──────┘
                                                    SQLite + CSV │ HTTP POST
                                                     (always)    │ (best effort)
                                                                 ▼
                                                          command post
                                                       ingest API + dashboard
```

## Compute placement

| Stage | Unit | Why there |
|---|---|---|
| Decode | CPU / hardware codec | Not a tensor op |
| Letterbox + normalise | Kryo | Cheap; moving it to the NPU costs more in transfer than it saves |
| Backbone + head | **Hexagon HTP** | The only stage that is genuinely MAC-bound |
| NMS | Kryo | Sorting and gathering are not HTP-shaped work. Folding NMS into the graph forces a CPU partition anyway, and hides the cost |
| Tracking, analytics | Kryo | Sequential, branchy, tiny tensors |

The graph is deliberately cut at a **narrow tensor**: the head output after
score reduction, not in the middle of the neck where feature maps are megabytes
per frame. Cutting a graph at a wide tensor costs more in transfer than the
offload saves.

## Why the timing is split four ways

`total_ms = pre_ms + infer_ms + post_ms + track_ms`, reported separately at
every layer — `Detections` carries the first three, `run_pipeline` adds the
fourth, the HUD shows all four live, and the CSV records all four per frame.

The reason is that these four scale with completely different things:

- `infer_ms` — constant per resolution, independent of scene content
- `post_ms` — scales with scene density
- `track_ms` — scales with the number of live tracks
- `pre_ms` — scales with source resolution

A single number cannot be attributed, and therefore cannot be optimised.

## Tracker

ByteTrack, written out in `3-pipeline/tracker.py` with a 7-state constant
velocity Kalman filter in `kalman.py`. Pure numpy — no torch, no scipy, no
lap/lapx — because everything here has to install on an aarch64 board without a
compiler.

Two-stage association:

1. High-confidence detections (`conf ≥ 0.50`) against all tracks, IoU ≥ 0.20
2. Low-confidence leftovers (`0.10 ≤ conf < 0.50`) against still-unmatched
   tracks, IoU ≥ 0.50

Stage 2 is what keeps small, partially-occluded objects alive — and on aerial
footage most objects are small.

> **Threshold semantics.** Upstream ByteTrack writes these as `match_thresh:
> 0.8` and `0.5`, which are thresholds on the *cost* `1 − IoU`. Copying `0.8`
> across as an IoU bar rejects nearly every real association and silently
> produces a tracker that never confirms anything. This repository states them
> as IoU (`match_iou: 0.20`, `match_iou_low: 0.50`) so they read the way they
> behave.

Aspect ratio is modelled as constant with no velocity term: a car seen from a
UAV changes apparent area as the drone climbs but barely changes shape, and
giving the ratio a velocity mostly lets boxes drift.

## Analytics

- **Counting** — confirmed tracks only. A detection seen fewer than `min_hits`
  times is not an object yet.
- **ROI membership** — box centroid, not overlap. With overlap, a vehicle
  straddling an ROI edge belongs to two regions and the regional counts stop
  summing to the frame count.
- **Line crossing** — sign change of the point-to-directed-segment test,
  evaluated on the track's own previous recorded side. A track that appears
  already past the line never fires, and jitter on the line counts zero.
- **Congestion level** — hysteresis of 5 consecutive frames before the state
  moves. A bare threshold on a per-frame count flickers whenever the detector
  drops one vehicle, producing an alert stream nobody trusts.

## Telemetry

Offline-first. Every frame goes to CSV and SQLite synchronously; the HTTP POST
is a bounded queue drained by a daemon thread. When the queue fills — uplink
slower than the pipeline — telemetry is dropped and counted, never blocked,
because the complete history is already on disk and fresh state matters more
than a complete stream.

`sink.summary()` reports `posted` / `dropped` / `post_failures` at exit so a
run can state how much of it reached the command post.

## Configuration

`configs/pipeline.yaml` holds every value that changes a reported number:
thresholds, ROI and line geometry, congestion levels, telemetry paths.
`configs/visdrone.yaml` holds the frozen 10-class taxonomy and the class
groupings used for reporting.

Nothing that affects a result is hard-coded in a script.
