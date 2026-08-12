# SkySentry — UAV traffic-order monitoring on a Qualcomm edge NPU

One drone, one edge box, one detector. A UAV surveys a district, a Qualcomm
QCS8550 board processes every frame on-device, and only structured state —
counts, crossings, congestion level — goes back to the command post.

Built for the SOICT Summer School Edge-AI project. The repository contains both
halves of the work: **a deployable product** and **a benchmark that says
honestly how well it runs**.

<p align="center">
  <img src="docs/img/overlay.png" alt="On-device overlay: tracks, counts, latency split" width="820">
</p>

---

## Why this rather than a wall of cameras

A fixed-camera deployment needs one device per viewpoint, cabling, and a
permanent install per intersection. One UAV covers a district and moves to
where the problem is. What it cannot do is stream raw 1080p video back over a
flaky radio link — so the intelligence has to sit on the aircraft's companion
board. That is the whole design constraint, and it is why every number in this
repository is measured on the device rather than on a laptop.

## What it does

| VisDrone task | Implemented as | Where |
|---|---|---|
| Task 1 — detection in images | YOLOv8n fine-tuned on VisDrone, 10 classes | `3-pipeline/detector.py` |
| Task 2 — detection in video | the same detector driven frame-by-frame | `3-pipeline/run_pipeline.py` |
| Task 4 — multi-object tracking | ByteTrack + Kalman, dependency-free | `3-pipeline/tracker.py` |
| Task 5 — crowd / vehicle counting | per-frame, per-ROI, per-class counting | `3-pipeline/analytics.py` |

On top of the tasks: directional line-crossing counts, a congestion level with
hysteresis, an offline-first telemetry store, and a command-post dashboard.

## What it measures

The benchmark is not an afterthought — it is the part that makes the product
claim believable.

- **Latency split, never a single number.** `pre` / `infer` / `post` / `track`
  are reported separately. A detector with a 3 ms forward pass and a 15 ms NMS
  tail is not a 3 ms detector.
- **Robustness under degraded capture.** Ten conditions — rain (light/medium/
  heavy), brightness up/down, motion blur, fog — applied deterministically so
  the degraded set is byte-identical across machines. See `2-augment/`.
- **Resource cost.** Thermal, CPU/GPU frequency and any power rail the board
  actually exposes. If the board has no energy counter, the tool says so
  instead of reporting a zero. See `4-bench/probe_power.py`.

### Results — VisDrone-DET val, 548 images, all ten conditions

The detector was fine-tuned on clean imagery only. That is the point of the
table: it measures what a conventionally-trained model does when it actually
has to fly.

| condition | AP kept | APs kept | | condition | AP kept | APs kept |
|---|---:|---:|---|---|---:|---:|
| `bright_up` | 97 % | 99 % | | `rain_light` | 82 % | 78 % |
| `fog_medium` | 94 % | 95 % | | `blur_medium` | 73 % | **60 %** |
| `bright_down` | 93 % | 89 % | | `rain_medium` | 52 % | 51 % |
| `blur_light` | 93 % | 89 % | | `rain_heavy` | **17 %** | 20 % |
| `bright_down_heavy` | 86 % | 82 % | | | | |

Three findings, all of which contradict the obvious expectation:

**Only rain actually breaks it.** Every other condition stays above 73 %.
A clean-trained model turns out to be far more robust to exposure — the thing
augmentation recipes emphasise — than to precipitation, which they usually omit.
If you can only augment one thing, augment rain.

**Blur is what destroys small objects.** `blur_medium` keeps 73 % of AP but only
60 % of APs. At 640 input, a 20-pixel object under a 9-pixel kernel is gone.
69 % of objects in this split are small, so the headline AP understates it.

**The latency penalty flips sign with the confidence threshold.** At the AP
protocol (`conf 0.001`) rain raises CPU postprocessing 85 %, because streaks
manufacture candidates that clear the threshold. At the deployment threshold
(`conf 0.25`) the same rain *lowers* NMS cost 37 % — the pipeline gets faster
because it sees less. Operationally that is the worse of the two: latency
telemetry looks healthy exactly when the detector is going blind. See
[`docs/BENCHMARK.md`](docs/BENCHMARK.md).

<p align="center">
  <img src="docs/img/robustness.png" alt="AP and APs across ten degradation conditions" width="860">
</p>

---

## Repository layout

The directory names are the pipeline.

```
0-setup/      find the board, deploy to it, probe its capabilities
1-model/      export the fine-tuned checkpoint to ONNX
2-augment/    deterministic weather / exposure / blur degradations
3-pipeline/   detector -> tracker -> analytics -> telemetry -> overlay
4-bench/      latency ladder, quality + robustness table, power probe
5-server/     command post: ingest API + dashboard
scripts/      dataset fetch, demo clip synthesis
configs/      every value that changes a reported number
```

## Quick start

### Host

```bash
python -m venv .venv && .venv/Scripts/activate      # Linux: source .venv/bin/activate
pip install -r requirements-host.txt

# 1. get the validation split (548 images, ~78 MB)
python scripts/fetch_data.py

# 2. export your fine-tuned checkpoint
python 1-model/export_onnx.py --weights models/yolov8n_visdrone.pt \
    --imgsz 640 --out models/yolov8n_visdrone_640.onnx

# 3. make a demo clip and run the whole pipeline over it
python scripts/make_clip.py --seconds 20
python 3-pipeline/run_pipeline.py --source assets/demo_clip.mp4 \
    --save-video results/demo.mp4 --headless
```

### Command post

```bash
python 5-server/app.py --port 8000        # dashboard at http://localhost:8000
```

### Board

```powershell
.\0-setup\find_device.ps1                 # locate it over SSH or adb
.\0-setup\deploy.ps1 -DeviceIp 192.168.1.77
```

```bash
ssh root@192.168.1.77                     # password on the stock image: oelinux123
cd /opt/skysentry && bash device_setup.sh

python3 4-bench/bench_latency.py --model models/yolov8n_visdrone_640.onnx --iters 50
python3 3-pipeline/run_pipeline.py --source 0 --headless \
    --post-url http://<laptop-ip>:8000/ingest
```

## Benchmarks

```bash
# latency ladder — needs no trained weights, run it the moment the board is up
python 4-bench/bench_latency.py --model models/yolov8n_visdrone_640.onnx

# full robustness table: 10 conditions over 548 images
python 4-bench/bench_quality.py --model models/yolov8n_visdrone_640.onnx \
    --all-conditions --out results/quality_mask.csv

# tables + figures
python 4-bench/report.py --quality results/quality_mask.csv \
    --latency results/latency.csv --out results/report

# side-by-side demo video: same model, clean vs degraded
python scripts/make_comparison_video.py --condition rain_heavy \
    --out results/compare_rain_heavy.mp4
```

The NMS tail, measured at four candidate loads on the host CPU — this is the
cost that a FLOPs table never shows:

| surviving candidates | 50 | 150 | 300 | 600 |
|---|---:|---:|---:|---:|
| `post_ms` | 1.92 | 4.21 | 9.01 | 19.82 |

Slightly superlinear, and VisDrone val averages 70.7 annotated objects per
image with the busiest frames well past 300.

### Closing the loop

The table says rain is the failure. [`1-model/finetune_weather.py`](1-model/finetune_weather.py)
is the intervention: it injects the *same* degradation functions the benchmark
uses as a training-time callback, so the training and evaluation distributions
come from identical code and an improvement cannot be an artefact of two
different rain models.

```bash
python 1-model/finetune_weather.py --weights models/yolov8n_visdrone.pt \
    --data VisDrone.yaml --epochs 60 --degrade-prob 0.35
```

Re-running `bench_quality.py` on the result gives a before/after pair on an
identical protocol.

Every benchmark row carries the model hash, the backend, the resolution, the
condition id and the ignore-region policy. A number without those cannot be
compared to anything, so the tooling refuses to emit one.

See [`docs/BENCHMARK.md`](docs/BENCHMARK.md) for the protocol and
[`docs/RUNBOOK.md`](docs/RUNBOOK.md) for device bring-up including the failure
modes we actually hit.

## Limitations

Stated up front, because a benchmark that hides these is not worth reading.

- Evaluation is on **VisDrone-DET val, 548 images, 10 classes**. Absolute AP is
  not directly comparable to the official VisDrone leaderboard; the ignore-region
  policy is reported per row so the convention is at least explicit.
- The demo clip is synthesised by flying a virtual camera over VisDrone stills.
  Object motion within the scene is real; **camera motion is not**. No tracking
  accuracy claim is made from it.
- Only one detector family is benchmarked (YOLOv8n). This is a deployment
  study, not an architecture comparison.
- The NPU path requires `onnxruntime-qnn` plus the QAIRT libraries on the
  device. Where a row was measured on the Kryo CPU instead, the `backend`
  column says so. **Every number currently in this repository is a CPU
  number** — the board was not reachable at the time of measurement.
- **The congestion monitor inherits the detector's blindness.** In heavy rain
  the vehicle count falls because detection fails, not because traffic
  cleared, and the dashboard will report "normal" with healthy-looking
  latency. Nothing in the current pipeline distinguishes an empty road from a
  road it cannot see. A deployment would need a capture-quality signal
  independent of the detector before this alert could be trusted.

## License

AGPL-3.0 — see [LICENSE](LICENSE). The detector is derived from
[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics), which is
AGPL-3.0; a combined work distributed or served over a network must publish its
source under the same terms. This repository is public for exactly that reason.

## Acknowledgements

- VisDrone benchmark — Zhu et al., Tianjin University
- Ultralytics YOLOv8
- ByteTrack — Zhang et al., 2022
- Qualcomm AI Hub / QAIRT tooling, and the SOICT Summer School organisers for
  the hardware
