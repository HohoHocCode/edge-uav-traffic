#!/usr/bin/env python3
"""Generate notebooks/finetune_weather_colab.ipynb.

Kept as a generator rather than a hand-edited .ipynb so the cell text lives in
readable Python and diffs stay legible.
"""

import json
import os

MD = "markdown"
CODE = "code"

CELLS: list[tuple[str, str]] = [
    (MD, """# SkySentry — fine-tune YOLOv8n on weather-augmented VisDrone

Fine-tunes your clean-trained `best.pt` on rain / exposure / blur, then
re-measures on **the same protocol** as the baseline so the before/after pair
is comparable.

**Runtime → Change runtime type → GPU** before running anything.

### The one design decision worth understanding

`rain_heavy` is deliberately **held out of training**. Training on the exact
condition you then report on makes the improvement partly circular — you
taught the model the test. Holding it out turns the `rain_heavy` column into a
genuine generalisation result: *trained on light and medium rain, recovers
this much on heavy rain it has never seen*.

That is a weaker number and a much stronger claim. `bright_down_heavy`,
`blur_medium` and `fog_medium` are held out for the same reason.
"""),

    (CODE, """!nvidia-smi
import torch
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU :", torch.cuda.get_device_name(0))
    print("VRAM:", round(torch.cuda.get_device_properties(0).total_memory/1e9, 1), "GB")
import os
print("vCPU:", os.cpu_count(), "<- this, not the GPU, usually sets epoch time")"""),

    (MD, """## 1. Install"""),

    (CODE, """!pip install -q ultralytics opencv-python-headless onnxruntime
import ultralytics
ultralytics.checks()"""),

    (MD, """## 2. Mount Drive and locate your baseline checkpoint

Upload your clean-trained `best.pt` to Drive first, then fix the path below."""),

    (CODE, """from google.colab import drive
drive.mount('/content/drive')

BEST_PT = '/content/drive/MyDrive/best.pt'      # <-- edit if needed
OUT_DIR  = '/content/drive/MyDrive/skysentry'

import os
assert os.path.exists(BEST_PT), f"not found: {BEST_PT} — upload best.pt to Drive first"
os.makedirs(OUT_DIR, exist_ok=True)
print("baseline:", BEST_PT, round(os.path.getsize(BEST_PT)/1e6, 2), "MB")"""),

    (MD, """## 3. Clone the repo"""),

    (CODE, """%cd /content
!rm -rf edge-uav-traffic
!git clone -q https://github.com/HohoHocCode/edge-uav-traffic.git
%cd /content/edge-uav-traffic
!ls"""),

    (MD, """## 4. Fetch VisDrone

~1.5 GB train + 78 MB val, pulled over Google's network — far faster than
uploading a prepared dataset from home."""),

    (CODE, """!python scripts/fetch_data.py --split train
!python scripts/fetch_data.py --split val"""),

    (MD, """## 5. Build the weather-augmented dataset  (~15–25 min)

Two flags carry the reasoning:

- `--max-side 1024` — stores images downscaled. Training letterboxes to 640
  regardless, so an object ends up the same size either way; what this buys is
  ~3.5× cheaper JPEG decode. Mosaic loads **four** images per sample, so on a
  2-vCPU box decode, not the GPU, is the bottleneck.
- `--conditions` — holds `rain_heavy` out of training (see the note at the top).

Labels are normalised against the *original* dimensions, so the downscale does
not move a single box."""),

    (CODE, """!python 2-augment/build_dataset.py \\
    --train data/VisDrone2019-DET-train \\
    --val   data/VisDrone2019-DET-val \\
    --out   /content/visdrone_weather \\
    --mode pair --seed 0 --jpeg-quality 92 --max-side 1024 \\
    --conditions rain_light:0.34 rain_medium:0.34 bright_down:0.16 blur_light:0.16"""),

    (MD, """## 6. Check the dataset before spending two hours on it

Confirms the class order matches your checkpoint, that `path:` points at
`/content`, and that the condition mix is what you asked for. A wrong class
order still trains happily — it just learns the wrong labels."""),

    (CODE, """import csv, collections, yaml, glob, os

cfg = yaml.safe_load(open('/content/visdrone_weather/data.yaml'))
print("path :", cfg['path'])
print("nc   :", cfg['nc'])
print("names:", cfg['names'])

mix = collections.Counter(r['condition'] for r in
        csv.DictReader(open('/content/visdrone_weather/manifest.csv')))
print("\\ncondition mix:", dict(mix))

n_img = len(glob.glob('/content/visdrone_weather/images/train/*.jpg'))
n_lbl = len(glob.glob('/content/visdrone_weather/labels/train/*.txt'))
print(f"\\ntrain images {n_img} | labels {n_lbl}")
assert n_img == n_lbl, "image/label count mismatch"

import ultralytics, torch
ck = torch.load(BEST_PT, map_location='cpu', weights_only=False)
ckpt_names = ck['model'].names if hasattr(ck['model'], 'names') else None
print("\\ncheckpoint classes:", ckpt_names)
assert list(cfg['names'].values()) == list(ckpt_names.values()), \\
    "CLASS ORDER MISMATCH — do not train, fix data.yaml first"
print("\\nclass order matches the checkpoint.")"""),

    (MD, """## 7. Visual spot check

Renders the stored labels back onto an augmented image. If the boxes sit on
vehicles, the geometry survived augmentation and downscaling."""),

    (CODE, """import cv2, glob, os
import matplotlib.pyplot as plt

f = sorted(glob.glob('/content/visdrone_weather/images/train/*__*.jpg'))[3]
stem = os.path.splitext(os.path.basename(f))[0]
img = cv2.imread(f); h, w = img.shape[:2]
for line in open(f'/content/visdrone_weather/labels/train/{stem}.txt'):
    p = line.split()
    if len(p) != 5: continue
    _, cx, cy, bw, bh = int(p[0]), *map(float, p[1:])
    cv2.rectangle(img, (int((cx-bw/2)*w), int((cy-bh/2)*h)),
                       (int((cx+bw/2)*w), int((cy+bh/2)*h)), (0,255,0), 2)
plt.figure(figsize=(14,8)); plt.imshow(img[:,:,::-1]); plt.axis('off')
plt.title(stem.split('__')[1]); plt.show()"""),

    (MD, """## 8. Train

`batch=32` suits a 16 GB T4. Drop to 16 on OOM — but **do not lower `imgsz`**:
VisDrone is overwhelmingly small objects, and cutting resolution destroys
exactly what this exercise is trying to protect.

No weather augmentation is enabled here. It is already baked into the pixels,
and stacking Ultralytics' own on top would make the training distribution
undocumented. HSV and mosaic stay at their defaults so the only difference
from your baseline run is the weather."""),

    (CODE, """from ultralytics import YOLO

model = YOLO(BEST_PT)
results = model.train(
    data='/content/visdrone_weather/data.yaml',
    epochs=60,
    imgsz=640,
    batch=32,
    seed=0,
    workers=2,
    patience=15,
    project='/content/runs',
    name='visdrone_weather',
    exist_ok=True,
    plots=True,
)"""),

    (MD, """### If Colab disconnected

Run this instead of the cell above — it picks up from the last epoch."""),

    (CODE, """# from ultralytics import YOLO
# model = YOLO('/content/runs/visdrone_weather/weights/last.pt')
# model.train(resume=True)"""),

    (MD, """## 9. Save to Drive immediately

Colab reclaims the VM without warning. Do this before anything else."""),

    (CODE, """!mkdir -p {OUT_DIR}
!cp /content/runs/visdrone_weather/weights/best.pt {OUT_DIR}/best_weather.pt
!cp /content/runs/visdrone_weather/results.csv {OUT_DIR}/train_results.csv
!cp -r /content/runs/visdrone_weather {OUT_DIR}/run_visdrone_weather
print("saved to", OUT_DIR)"""),

    (MD, """## 10. Export to ONNX

opset 13, static shape, NMS left outside the graph — the settings the QNN /
Hexagon path needs."""),

    (CODE, """!python 1-model/export_onnx.py \\
    --weights /content/runs/visdrone_weather/weights/best.pt \\
    --imgsz 640 --opset 13 \\
    --out models/yolov8n_visdrone_weather_640.onnx

!cp models/yolov8n_visdrone_weather_640.onnx {OUT_DIR}/
!cp models/yolov8n_visdrone_weather_640.meta.json {OUT_DIR}/"""),

    (MD, """## 11. Re-measure on the identical protocol  (~30–40 min)

The point of the whole exercise. Same 548 val images, same ten conditions,
same ignore policy, same confidence threshold as the baseline table — so the
two are directly comparable.

Note the val split is **clean**: the degradation is applied by the benchmark
at evaluation time, not baked in."""),

    (CODE, """!python 4-bench/bench_quality.py \\
    --model models/yolov8n_visdrone_weather_640.onnx \\
    --data data/VisDrone2019-DET-val \\
    --all-conditions --ignore-policy mask \\
    --out results/quality_weather.csv --tag weather-aug

!cp results/quality_weather.csv {OUT_DIR}/"""),

    (MD, """## 12. Before / after

`baseline` is the committed clean-trained result; `weather` is what you just
produced. Held-out conditions are marked — those are the honest generalisation
numbers."""),

    (CODE, """import pandas as pd

TRAINED_ON = {'rain_light', 'rain_medium', 'bright_down', 'blur_light'}

base = pd.read_csv('docs/results/quality_mask.csv').drop_duplicates('condition', keep='last')
new  = pd.read_csv('results/quality_weather.csv').drop_duplicates('condition', keep='last')

m = base.merge(new, on='condition', suffixes=('_base', '_new'))
m['AP_delta']  = m['AP_new']  - m['AP_base']
m['APs_delta'] = m['APs_new'] - m['APs_base']
m['seen'] = m['condition'].apply(lambda c: 'trained' if c in TRAINED_ON
                                 else ('—' if c == 'clean' else 'HELD OUT'))

cols = ['condition', 'seen', 'AP_base', 'AP_new', 'AP_delta',
        'APs_base', 'APs_new', 'APs_delta']
print(m[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

print("\\nclean AP change:",
      f"{float(m.loc[m.condition=='clean','AP_delta'].iloc[0]):+.4f}",
      "  <- the price paid for robustness; a large drop means too much rain")"""),

    (MD, """## 13. Figures

Writes the same tables and plots the repo generates for the baseline."""),

    (CODE, """!python 4-bench/report.py \\
    --quality results/quality_weather.csv \\
    --out results/report_weather

!cp -r results/report_weather {OUT_DIR}/
from IPython.display import Image, display
display(Image('results/report_weather/fig_robustness.png'))"""),

    (MD, """---

### Reading the result

- **`rain_heavy` (held out)** is the headline. The baseline retains 17 % of
  clean AP there. Any recovery is generalisation, not recall.
- **`clean`** is the cost. A drop of 1–3 points is normal and worth it. More
  than ~5 points means the rain share is too high — rebuild with
  `--mode mixed --degrade-frac 0.4` and retrain.
- **`fog_medium` and `blur_medium`** were never trained on either. If they move
  much, the model learned something general about degraded input rather than
  three specific effects.

### What this still does not prove

The rain is synthetic. It has no lens droplets, no wet-road reflections, and
no sensor noise in the dark frames. The result is robustness to *this* rain
model — a controlled, reproducible stress test, not evidence the drone works
in real weather. Say that in the report; a reviewer will ask.
"""),
]


def main() -> None:
    nb = {
        "cells": [
            {
                "cell_type": kind,
                "metadata": {},
                "source": text.splitlines(keepends=True),
                **({"outputs": [], "execution_count": None} if kind == CODE else {}),
            }
            for kind, text in CELLS
        ],
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "notebooks", "finetune_weather_colab.ipynb")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"wrote {out} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
