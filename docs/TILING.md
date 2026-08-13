# Tiled inference — measured, not argued

Objects in drone footage are small enough that the input resolution, not the
model, is the binding constraint. This document records what cutting the frame
into overlapping tiles is actually worth, on real validation data.

---

## 1. The arithmetic that motivates it

A 1920×1080 frame letterboxed to 640 is scaled by **0.33**. Mosaic augmentation
during training halves it again.

| object | in the frame | at 640 | after mosaic |
|---|---:|---:|---:|
| pedestrian | 15 px | 5.0 px | **2.5 px** |
| motorbike | 25 px | 8.3 px | 4.2 px |
| car | 45 px | 15.0 px | 7.5 px |

A stride-8 feature map cannot represent a 2.5 px object. Those labels are
asking the network for something the architecture has no way to express, and no
amount of additional training changes that.

Cutting 2×2 with 20% overlap gives 1067×600 tiles, which letterbox at **0.60**
instead of 0.33 — every object arrives **1.80× larger**, independent of size.

---

## 2. What it actually buys

Same model, same images, same AP code; the only difference is the inference
strategy. `yolov8n_visdrone_640.onnx`, VisDrone-DET val, conf 0.001, IoU 0.65,
maxDets 500, ignore-regions masked.

| | AP | AP50 | **APs** | det/img | ms/img |
|---|---:|---:|---:|---:|---:|
| untiled | 0.2112 | 0.3372 | 0.0612 | 256.1 | 62.8 |
| **tiled 2×2, overlap 0.20** | **0.2468** | **0.3943** | **0.1038** | 311.2 | 248.4 |
| | **+16.8 %** | +16.9 % | **+69.8 %** | | **3.95×** |

60 images. `docs/results/tiling.csv`.

**The gain is concentrated in APs, which is what makes the result believable.**
The stated mechanism is "small objects arrive larger", and small objects are
where nearly all of it lands: +69.8 % on APs against +16.8 % overall. A uniform
improvement across all sizes would have meant something other than the claimed
mechanism was responsible, and the claim would need rewriting.

The cost is **3.95×** against 4 forward passes, so the merge itself is close to
free. This is the trade that was accepted deliberately: latency for recall on
the objects the platform exists to find.

---

## 3. Three details that are silent when wrong

### Overlap must exceed the largest object

At 20% the shared band is 214 px on a 1920-wide frame. An object wider than the
band can be cut by every tile that sees it, leaving no tile holding it whole.
For denser or closer footage the overlap has to grow with the objects.

### Truncated detections must be dropped, not merged

A car bisected by a tile edge yields a confident half-car box. NMS will not
remove it: its IoU against the whole-car box from the neighbouring tile is
around 0.5, below any usable threshold. So detections touching an *interior*
tile edge are dropped outright — the overlap guarantees a neighbour saw the
object whole. Detections touching the *image* edge are kept, because there is
no neighbour and the truncation is real.

Measured on the same 40 images:

| | AP | APs | det/img |
|---|---:|---:|---:|
| keep truncated | 0.2071 | 0.0939 | 313.0 |
| **drop truncated** | **0.2141** | **0.1019** | 290.4 |
| | +3.4 % | **+8.5 %** | −22.6 |

The 22.6 detections per image that disappear are duplicate half-objects. Worth
+8.5 % of APs on its own.

### Merging must be per class

Class-agnostic NMS across tiles deletes a motorbike that legitimately overlaps
a car. The implementation offsets boxes by class into disjoint coordinate
bands, so a single NMS pass behaves as one pass per class.

---

## 4. Training on tiles

`notebooks/finetune_1model_colab.ipynb` generates the tiled dataset as well, so
the model is trained on the same distribution it will be served on. Boxes cut
by a tile edge are kept only when at least `MIN_BOX_VISIBLE` (0.40) of their
area survives; below that they are dropped, because teaching the model that a
sliver of a car is a car produces false positives on every fragment of metal.

The numbers in section 2 come from a model trained **without** tiles, so they
understate what a tile-trained model should reach.

Two side effects of tiling the training set worth knowing:

- **4× the images**, so 4× the epoch time.
- **Fewer objects per image**, which relaxes the `TaskAlignedAssigner` memory
  bound — its tensors are `(batch, n_max_boxes, n_anchors)` — so the batch cap
  computed in the notebook rises. See `docs/RUNBOOK.md`.

---

## 5. Reproducing

```bash
python 4-bench/bench_tiled.py --model models/yolov8n_visdrone_640.onnx \
    --limit 60 --grid 2 --overlap 0.20
```

`--keep-truncated` disables the edge rule, which is how the section 3 table was
produced. Implementation: `3-pipeline/tiled_detector.py`.
