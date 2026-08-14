# Tracker configurations

Ultralytics resolves a bare `botsort.yaml` to *its own* shipped default, so a
tuned value cannot live in code — it has to live in a file passed by path. These
are those files. Each one changes exactly one thing against its baseline, so a
row in `results/tracking_quality.csv` attributes a delta to a cause.

    python 4-bench/bench_tracking.py --trackers \
        botsort configs/trackers/botsort_nogmc.yaml \
        configs/trackers/botsort_reid.yaml \
        ocsort configs/trackers/deepocsort_gmc_reid.yaml tracktrack

| file | baseline | single change | question it answers |
|---|---|---|---|
| *(builtin)* `botsort` | — | — | reference: what the showcase renders today |
| `botsort_nogmc.yaml` | botsort | `gmc_method: none` | is optical-flow camera compensation earning its ~ms? |
| `botsort_reid.yaml` | botsort | `with_reid: True` | does appearance rescue the identities that motion loses? |
| `botsort_buffer.yaml` | botsort | `track_buffer: 60` | are lost tracks dying before the occlusion ends? |
| *(builtin)* `ocsort` | — | — | motion-only, **no GMC** — the control for the GMC question |
| `deepocsort_gmc_reid.yaml` | deepocsort | `gmc_method` + `with_reid` on | OC-SORT's momentum *plus* everything BoT-SORT has |
| *(builtin)* `tracktrack` | — | — | the strongest option installed |

Two traps in the shipped defaults, which is why the last two rows exist:

* `ocsort.yaml` has **no** `gmc_method` key at all, and `oc_sort.py` never
  imports `GMC`. Selecting OC-SORT on drone footage silently removes camera
  compensation.
* `deepocsort.yaml` ships `gmc_method: none` and `with_reid: False`, so plain
  `--tracker deepocsort` is *weaker* than BoT-SORT, not stronger. Only
  `deepocsort_gmc_reid.yaml` is a fair comparison.

`with_reid: True` with `model: auto` costs almost nothing here: Ultralytics
hooks the `Detect` layer of `models/yolov8n_visdrone.pt` and reuses the features
the forward pass already computed. It requires a torch `.pt` — pointed at an
ONNX graph it falls back to downloading a separate classifier.
