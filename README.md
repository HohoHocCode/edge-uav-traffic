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

Early result from the validation split, which is the reason the robustness axis
exists at all:

| condition | AP | APs (small objects) |
|---|---|---|
| clean | reference | reference |
| rain (medium) | −14 % | **−39 %** |

Rain costs small objects roughly three times what it costs the headline
number — and on aerial footage 69 % of all objects are small. A single AP
figure hides exactly the failure that matters.

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
    --all-conditions --out results/quality.csv
```

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
  column says so.

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
