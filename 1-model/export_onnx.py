#!/usr/bin/env python3
"""Export a fine-tuned Ultralytics checkpoint to ONNX for the edge runtime.

Run on the host (needs torch + ultralytics). The board never sees torch.

Choices made here, and why:

``opset=13``
    The QNN execution provider and the AI Hub converters are conservative.
    Newer opsets introduce ops that get partitioned back to the CPU, which
    shows up as a large ``n_ops_fallback`` and a latency cliff.

``dynamic=False``, fixed 1x3xHxW
    A static shape is a hard requirement for producing a QNN context binary.
    A dynamic axis forces the graph to be re-finalised, or refused.

``nms=False``
    NMS stays on the Kryo cores and is timed separately. Folding it into the
    graph would hide the single most important CPU cost in the pipeline, and
    the NMS ops themselves fall back to CPU anyway.

``simplify=True``
    Removes the shape-inference clutter that otherwise becomes dozens of
    unsupported nodes on the HTP.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys


def sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sanitise_onnx(path: str) -> list[str]:
    """Remove graph IO tensors that are also declared in ``value_info``.

    Ultralytics' export followed by onnxslim can leave the output tensor
    listed twice: once in ``graph.output`` and again in ``graph.value_info``.
    The ONNX spec says value_info carries *intermediate* tensors only, so this
    is malformed. ONNX Runtime tolerates it silently, which is why it survives
    local testing -- but the Qualcomm AI Hub converter rejects the model
    outright:

        Tensors {'output0'} occur in value_info but also in model IO.

    Stripping the duplicates changes no semantics: the shapes are already
    declared authoritatively in graph.input / graph.output.
    """
    try:
        import onnx
    except ImportError:
        return []

    model = onnx.load(path)
    io_names = {t.name for t in model.graph.input} | \
               {t.name for t in model.graph.output}
    dupes = [vi.name for vi in model.graph.value_info if vi.name in io_names]
    if not dupes:
        return []

    keep = [vi for vi in model.graph.value_info if vi.name not in io_names]
    del model.graph.value_info[:]
    model.graph.value_info.extend(keep)
    onnx.save(model, path)
    return dupes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="path to best.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--opset", type=int, default=13)
    ap.add_argument("--out", default=None, help="output .onnx path")
    ap.add_argument("--half", action="store_true", help="fp16 (GPU targets only)")
    args = ap.parse_args()

    if not os.path.exists(args.weights):
        print(f"[fatal] weights not found: {args.weights}", file=sys.stderr)
        return 2

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[fatal] pip install ultralytics onnx onnxslim", file=sys.stderr)
        return 2

    model = YOLO(args.weights)
    names = model.names
    print(f"[info] checkpoint classes ({len(names)}): {names}")

    exported = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        dynamic=False,
        simplify=True,
        nms=False,
        half=args.half,
        batch=1,
    )
    exported = str(exported)

    out = args.out
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        if os.path.abspath(exported) != os.path.abspath(out):
            os.replace(exported, out)
    else:
        out = exported

    dupes = sanitise_onnx(out)
    if dupes:
        print(f"[fix] removed {len(dupes)} duplicate value_info entr"
              f"{'y' if len(dupes) == 1 else 'ies'}: {dupes}")
        print("      (ONNX Runtime ignores these; the AI Hub converter "
              "rejects the model because of them)")

    meta = {
        "source_weights": os.path.abspath(args.weights),
        "source_sha256": sha256(args.weights),
        "onnx": os.path.abspath(out),
        "onnx_sha256": sha256(out),
        "imgsz": args.imgsz,
        "opset": args.opset,
        "nc": len(names),
        "names": {int(k): v for k, v in names.items()},
    }
    meta_path = os.path.splitext(out)[0] + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    size_mb = os.path.getsize(out) / 1e6
    print(f"[ok] {out}  ({size_mb:.2f} MB)")
    print(f"[ok] {meta_path}")
    print(f"[ok] onnx sha256 {meta['onnx_sha256'][:16]}...  "
          f"— record this in the benchmark table")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
