# Runbook — getting a Qualcomm board to run this

Written from the failures we actually hit, in the order they hit us.

---

## 1. Find the board

```powershell
.\0-setup\find_device.ps1
```

It checks three paths: SSH on the local /24, adb over USB, and link-local RNDIS
peers.

### Failure: `adb devices` is empty

Check Device Manager (or `Get-PnpDevice`) before blaming the driver. What you
see tells you which problem you have:

| Symptom | Cause | Fix |
|---|---|---|
| No new USB device appears at all | The cable has no data pair, or it is in the power port | Use a cable known to carry data; use the port marked in the vendor docs |
| `Unknown USB Device (Device Descriptor Request Failed)` | The device is drawing power but not enumerating | Almost always the cable. Try another one before installing anything |
| Device appears as `QUSB_BULK` or similar, with a warning triangle | Driver missing | Install the [Qualcomm Userspace Driver](https://softwarecenter.qualcomm.com/catalog/item/Qualcomm_Userspace_Driver) |
| Device enumerates but `adb devices` is empty | `adb` not installed, or the server is stale | `adb kill-server && adb start-server` |

`adb` itself: unzip
[platform-tools](https://dl.google.com/android/repository/platform-tools-latest-windows.zip)
anywhere and point `-AdbPath` at it.

**Prefer Ethernet.** RJ45 + SSH is more stable than Type-C, gives faster file
transfer, and needs no vendor driver. Type-C is the fallback for when the board
has no network.

### Failure: the network scan finds nothing

The board's Ethernet must be on the **same** router or switch as your laptop.
Laptop on Wi-Fi and board on a different LAN will not see each other. Confirm
your own interface is up first:

```powershell
Get-NetAdapter | Where-Object Status -eq 'Up'
```

An `Ethernet` interface holding a `169.254.x.x` address means the cable is not
plugged in or the far end is dead — that is APIPA, not a working link.

Match the discovered MAC against the sticker on the device box. On a venue
network there will be other Linux hosts answering on port 22.

---

## 2. Log in

```bash
ssh root@<ip>          # stock HSPTEK / RB3 image password: oelinux123
```

Change it if the board is on a shared network.

---

## 3. Deploy

```powershell
.\0-setup\deploy.ps1 -DeviceIp <ip>
# or, over USB:
.\0-setup\deploy.ps1 -Adb
```

Only the runtime half is copied — pipeline, benchmarks, configs, model. The
dataset and the export toolchain stay on the laptop.

---

## 4. Bring-up

```bash
cd /opt/skysentry && bash device_setup.sh
```

This prints the SoC id, the Python packages present, the ONNX Runtime execution
providers, whether `libQnn*.so` exists, camera nodes, and the power/thermal
channels it can read.

### The decisive line

```
providers: ['CPUExecutionProvider']
warn QNN EP absent - inference will run on the Kryo CPU.
```

This is the difference between benchmarking a 48 TOPS NPU and benchmarking a
CPU. Fix it before recording any number:

```bash
pip3 install onnxruntime-qnn        # replaces plain onnxruntime; do not install both
```

`onnxruntime-qnn` needs the QAIRT runtime libraries on the device. If
`device_setup.sh` reports no `libQnn*.so`, install the QAIRT SDK on the board
(or copy the `aarch64-oe-linux` runtime libs from the SDK) and put them on
`LD_LIBRARY_PATH`:

```bash
export LD_LIBRARY_PATH=/opt/qairt/lib/aarch64-oe-linux-gcc11.2:$LD_LIBRARY_PATH
```

Verify the provider is genuinely active — requesting it is not the same as
getting it. `run_pipeline.py` prints `providers=[...]` from the live session at
startup for exactly this reason.

---

## 5. First measurements

```bash
python3 4-bench/probe_power.py --discover
python3 4-bench/bench_latency.py --model models/yolov8n_visdrone_640.onnx --iters 50
python3 3-pipeline/run_pipeline.py --source assets/demo_clip.mp4 --headless --max-frames 300
```

Run the latency ladder before anything else. It needs no trained weights and no
dataset, so it works the minute the board is reachable.

---

## 6. Live demo wiring

On the laptop:

```bash
python 5-server/app.py --host 0.0.0.0 --port 8000
```

On the board:

```bash
python3 3-pipeline/run_pipeline.py --source 0 \
    --post-url http://<laptop-ip>:8000/ingest --headless
```

Telemetry is written locally first and posted best-effort, so pulling the
network cable mid-demo degrades the dashboard but never stalls the pipeline.
The dashboard marks a node stale after 15 s rather than showing its last value
as if it were current.

If the laptop firewall blocks the board:

```powershell
New-NetFirewallRule -DisplayName "SkySentry ingest" -Direction Inbound `
    -LocalPort 8000 -Protocol TCP -Action Allow
```

---

## Disk layout — `data/` and `.venv/` are junctions

On this workstation the C: drive filled to zero bytes, so the two bulky
directories live on D: and are reached through NTFS junctions:

```
C:\Users\ASUS\edge-uav-traffic\data   ->  D:\edge-uav-traffic\data
C:\Users\ASUS\edge-uav-traffic\.venv  ->  D:\edge-uav-traffic\.venv
```

Every path in every script is unchanged — a junction is transparent, and
`sys.prefix` still reports the C: path. Verified after the move: both splits
load (548 val, 6471 train) and a 20-image `bench_quality.py` run reproduces.

Two consequences worth knowing:

- **`rm -rf` on the junction deletes the target's contents**, not just the
  link. Use `Remove-Item` on the *link* only if you mean to unlink; to move
  the data back, robocopy from D: first.
- **New datasets belong on D:.** C: has single-digit GB free; D: has ~330 GB.
  Point `--data` at `D:\...` directly rather than copying into the repo.

### Regenerating what was deleted to make room

All of these are reproducible; none is in git because of size.

| Artefact | Size | Rebuild with |
|---|---:|---|
| Weather-augmented dataset | 4.9 GB | `2-augment/build_dataset.py` — seeded per image index, so it reproduces exactly. The file list it produced is kept at `docs/results/weather_dataset_manifest/manifest.csv` |
| `results/calib/calibration.npy` | 1.5 GB | `1-model/make_calibration.py --n 64 --seed 0`; expected sha256 `d80e49589bd7ea0a` (recorded in QUANTIZATION.md §1) |
| `results/device_in/` | 0.9 GB | `4-bench/make_device_inputs.py --limit 200` |
| `results/device_out_*/` | 0.2 GB | re-run `qnn-net-run` on the board, ~1 min |
| `models/_ctx/*.bin` | 11 MB | re-download from AI Hub compile jobs `j5w4kzomg` / `jp81nrvq5` |

## Known-good invariants

Check these before recording a result:

- [ ] `providers` in the startup line contains `QNNExecutionProvider` — otherwise the row is a CPU row and must be labelled as one.
- [ ] `--iters` warm-up ran; the first QNN call pays graph finalisation and does not belong in a steady-state figure.
- [ ] The model hash in the benchmark row matches the artefact actually deployed.
- [ ] `probe_power.py --discover` reported an energy counter before any watt figure is quoted. If it did not, quote thermal and frequency instead.
