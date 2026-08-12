#!/usr/bin/env python3
"""Discover and sample the power / thermal / frequency telemetry a board exposes.

Runs on the device. Different Qualcomm boards expose different subsets of
sysfs, and a benchmark that assumes a rail exists will either crash or, worse,
report zeros as if they were measurements. So this tool has two modes:

    --discover   enumerate what is actually readable, print a report, exit
    --sample     log the readable channels at a fixed interval to CSV

Anything not present is reported as absent, never as 0. If no energy rail is
readable at all, the honest output of this script is "no energy counter on this
board", and the report should then quote thermal + frequency residency as the
power proxy rather than inventing a watt figure.

Typical use, alongside a pipeline run:

    python3 4-bench/probe_power.py --sample --interval 0.5 \
        --out results/power_run1.csv &
    python3 3-pipeline/run_pipeline.py --source clip.mp4 --headless
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import time

# Candidate sysfs locations, most specific first.
POWER_SUPPLY_GLOB = "/sys/class/power_supply/*"
THERMAL_GLOB = "/sys/class/thermal/thermal_zone*"
CPUFREQ_GLOB = "/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq"
GPU_FREQ_CANDIDATES = [
    "/sys/class/kgsl/kgsl-3d0/gpuclk",
    "/sys/class/kgsl/kgsl-3d0/devfreq/cur_freq",
]
GPU_BUSY_CANDIDATES = [
    "/sys/class/kgsl/kgsl-3d0/gpu_busy_percentage",
    "/sys/class/kgsl/kgsl-3d0/devfreq/gpu_load",
]
# Hexagon/NPU residency is not exposed as a public sysfs node on Qualcomm
# Linux; the QNN profiling output is the only supported source. Recorded here
# so nobody assumes the absence is an oversight.
NPU_NOTE = ("NPU utilisation is not exposed via sysfs on Qualcomm Linux; "
            "use qnn-profile-viewer or the AI Hub profile job instead.")


def _read(path: str) -> str | None:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()
    except OSError:
        return None


def _read_float(path: str) -> float | None:
    v = _read(path)
    if v is None:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def discover() -> dict:
    found: dict[str, dict] = {
        "power_supply": {}, "thermal": {}, "cpufreq": {},
        "gpu": {}, "notes": [NPU_NOTE],
    }

    for d in sorted(glob.glob(POWER_SUPPLY_GLOB)):
        name = os.path.basename(d)
        entry = {}
        for key, fname in (
            ("voltage_uV", "voltage_now"),
            ("current_uA", "current_now"),
            ("power_uW", "power_now"),
            ("energy_uWh", "energy_now"),
            ("capacity_pct", "capacity"),
            ("type", "type"),
        ):
            val = _read(os.path.join(d, fname))
            if val is not None:
                entry[key] = val
        if entry:
            found["power_supply"][name] = entry

    for d in sorted(glob.glob(THERMAL_GLOB)):
        t = _read_float(os.path.join(d, "temp"))
        if t is None:
            continue
        found["thermal"][os.path.basename(d)] = {
            "type": _read(os.path.join(d, "type")) or "?",
            # Kernel reports millidegrees on essentially every ARM SoC.
            "temp_C": t / 1000.0 if t > 1000 else t,
        }

    for p in sorted(glob.glob(CPUFREQ_GLOB)):
        cpu = p.split(os.sep)[-3]
        khz = _read_float(p)
        if khz is not None:
            found["cpufreq"][cpu] = khz / 1000.0   # MHz

    for p in GPU_FREQ_CANDIDATES:
        v = _read_float(p)
        if v is not None:
            found["gpu"]["freq_hz"] = v
            found["gpu"]["freq_source"] = p
            break
    for p in GPU_BUSY_CANDIDATES:
        v = _read(p)
        if v is not None:
            found["gpu"]["busy"] = v
            found["gpu"]["busy_source"] = p
            break

    return found


def has_energy_counter(d: dict) -> bool:
    for entry in d["power_supply"].values():
        if "power_uW" in entry or ("voltage_uV" in entry and "current_uA" in entry):
            return True
    return False


def sample_row(d_template: dict) -> dict:
    row: dict[str, float | str] = {"t": round(time.time(), 3)}

    for name in d_template["power_supply"]:
        base = f"/sys/class/power_supply/{name}"
        v = _read_float(f"{base}/voltage_now")
        i = _read_float(f"{base}/current_now")
        p = _read_float(f"{base}/power_now")
        if p is not None:
            row[f"{name}_W"] = p / 1e6
        elif v is not None and i is not None:
            row[f"{name}_W"] = abs(v * i) / 1e12   # uV * uA -> W
        if v is not None:
            row[f"{name}_V"] = v / 1e6

    for zone, meta in d_template["thermal"].items():
        t = _read_float(f"/sys/class/thermal/{zone}/temp")
        if t is not None:
            row[f"{meta['type']}_C"] = t / 1000.0 if t > 1000 else t

    for cpu in d_template["cpufreq"]:
        khz = _read_float(
            f"/sys/devices/system/cpu/{cpu}/cpufreq/scaling_cur_freq")
        if khz is not None:
            row[f"{cpu}_MHz"] = khz / 1000.0

    if "freq_source" in d_template.get("gpu", {}):
        v = _read_float(d_template["gpu"]["freq_source"])
        if v is not None:
            row["gpu_MHz"] = v / 1e6 if v > 1e6 else v
    if "busy_source" in d_template.get("gpu", {}):
        v = _read(d_template["gpu"]["busy_source"])
        if v is not None:
            row["gpu_busy"] = v

    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--discover", action="store_true")
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--interval", type=float, default=0.5)
    ap.add_argument("--duration", type=float, default=0.0, help="0 = until Ctrl-C")
    ap.add_argument("--out", default="results/power.csv")
    args = ap.parse_args()

    if not args.discover and not args.sample:
        args.discover = True

    d = discover()

    if args.discover:
        print("=== power supply rails ===")
        print("  (none readable)" if not d["power_supply"] else "")
        for k, v in d["power_supply"].items():
            print(f"  {k}: {v}")
        print("\n=== thermal zones ===")
        print("  (none readable)" if not d["thermal"] else "")
        for k, v in d["thermal"].items():
            print(f"  {k}: {v['type']} = {v['temp_C']:.1f} C")
        print("\n=== cpu frequencies (MHz) ===")
        print("  (none readable)" if not d["cpufreq"] else f"  {d['cpufreq']}")
        print("\n=== gpu ===")
        print(f"  {d['gpu'] or '(none readable)'}")
        print("\n=== notes ===")
        for n in d["notes"]:
            print(f"  - {n}")
        print(f"\nenergy counter available: {has_energy_counter(d)}")
        if not has_energy_counter(d):
            print("  -> report thermal + frequency as the power proxy, and say so. "
                  "Do not quote a watt figure this board cannot measure.")
        if not args.sample:
            return 0

    # ---- sampling ------------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    first = sample_row(d)
    if len(first) <= 1:
        print("[fatal] nothing sampleable on this device", flush=True)
        return 1

    t_end = time.time() + args.duration if args.duration > 0 else float("inf")
    n = 0
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(first.keys()), extrasaction="ignore")
        w.writeheader()
        w.writerow(first)
        f.flush()
        n += 1
        try:
            while time.time() < t_end:
                time.sleep(args.interval)
                w.writerow(sample_row(d))
                f.flush()
                n += 1
        except KeyboardInterrupt:
            pass
    print(f"[ok] {n} samples -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
