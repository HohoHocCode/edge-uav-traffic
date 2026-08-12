"""ONNX Runtime session wrapper, CPU and QNN (Hexagon NPU).

The wrapper exists for three reasons the raw ORT API does not give us:

1. It times *only* the ``run()`` call, so ``infer_ms`` is the compute-unit
   number and never includes preprocessing or NMS. Those are timed separately
   by the caller. Merging them is the single most common way an edge benchmark
   becomes unreadable.
2. It records which execution provider actually got used, not which one was
   requested. ORT silently falls back per-node; asking afterwards is the only
   honest answer.
3. It reports how many nodes were assigned away from the accelerator, which is
   the ``n_ops_fallback`` column of the benchmark.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np
import onnxruntime as ort


@dataclass
class SessionInfo:
    backend_name: str
    providers_requested: list[str]
    providers_active: list[str]
    input_name: str
    input_shape: tuple
    output_names: list[str]
    n_nodes_total: int = -1
    n_nodes_fallback: int = -1
    notes: list[str] = field(default_factory=list)


class OrtSession:
    """Minimal, well-instrumented ONNX Runtime session."""

    def __init__(
        self,
        model_path: str,
        backend_name: str = "onnxruntime-cpu",
        intra_threads: int | None = None,
        qnn_backend_path: str = "libQnnHtp.so",
        qnn_performance_mode: str = "burst",
        profile_dir: str | None = None,
    ) -> None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)

        self.model_path = model_path
        self.backend_name = backend_name

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if intra_threads:
            so.intra_op_num_threads = int(intra_threads)
        if profile_dir:
            os.makedirs(profile_dir, exist_ok=True)
            so.enable_profiling = True
            so.profile_file_prefix = os.path.join(profile_dir, "ort_profile")

        if backend_name == "onnxruntime-qnn":
            providers = [
                (
                    "QNNExecutionProvider",
                    {
                        "backend_path": qnn_backend_path,
                        "htp_performance_mode": qnn_performance_mode,
                        "htp_graph_finalization_optimization_mode": "3",
                    },
                ),
                "CPUExecutionProvider",
            ]
        elif backend_name == "onnxruntime-cpu":
            providers = ["CPUExecutionProvider"]
        else:
            raise ValueError(f"unsupported backend {backend_name!r}")

        self.session = ort.InferenceSession(
            model_path, sess_options=so, providers=providers
        )

        inp = self.session.get_inputs()[0]
        self.info = SessionInfo(
            backend_name=backend_name,
            providers_requested=[p if isinstance(p, str) else p[0] for p in providers],
            providers_active=list(self.session.get_providers()),
            input_name=inp.name,
            input_shape=tuple(inp.shape),
            output_names=[o.name for o in self.session.get_outputs()],
        )
        self._probe_partitioning()

        # Timing bookkeeping
        self.last_infer_ms: float = 0.0
        self._times: list[float] = []

    # ------------------------------------------------------------------ #
    def _probe_partitioning(self) -> None:
        """Count nodes that did not land on the accelerator.

        ORT does not expose the partitioning map through the Python API, so we
        read it back from the session's own profiling/verbose channel when
        available. When it is not, we record -1 rather than guessing — an
        invented number here would silently corrupt the benchmark's most
        interesting column.
        """
        try:
            import onnx

            model = onnx.load(self.model_path, load_external_data=False)
            self.info.n_nodes_total = len(model.graph.node)
        except Exception as exc:  # onnx not installed on the board, etc.
            self.info.notes.append(f"node count unavailable: {type(exc).__name__}")
            return

        if self.backend_name != "onnxruntime-qnn":
            self.info.n_nodes_fallback = 0
            return

        # Filled in by tools/parse_qnn_partition.py from the ORT verbose log;
        # left at -1 here so nobody mistakes a guess for a measurement.
        self.info.notes.append(
            "n_nodes_fallback requires ORT verbose log; run with "
            "ORT_LOG_SEVERITY=0 and parse with 4-bench/parse_partition.py"
        )

    # ------------------------------------------------------------------ #
    def run(self, x: np.ndarray) -> list[np.ndarray]:
        """Run one forward pass. Times only the ORT call."""
        t0 = time.perf_counter()
        out = self.session.run(self.info.output_names, {self.info.input_name: x})
        self.last_infer_ms = (time.perf_counter() - t0) * 1000.0
        self._times.append(self.last_infer_ms)
        return out

    def warmup(self, shape: tuple[int, int, int, int], n: int = 5) -> None:
        """Run ``n`` dummy passes and discard their timings.

        Mandatory before any measurement: the first QNN call pays graph
        finalization and the first CPU call pays arena allocation, and both are
        one-off costs that do not belong in a steady-state latency figure.
        """
        dummy = np.zeros(shape, dtype=np.float32)
        for _ in range(n):
            self.run(dummy)
        self.reset_timers()

    def reset_timers(self) -> None:
        self._times.clear()

    def stats(self) -> dict:
        """avg / p50 / p95 over the timings recorded since the last reset."""
        if not self._times:
            return {"n": 0, "avg_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}
        a = np.asarray(self._times, dtype=np.float64)
        return {
            "n": int(a.size),
            "avg_ms": float(a.mean()),
            "p50_ms": float(np.percentile(a, 50)),
            "p95_ms": float(np.percentile(a, 95)),
        }

    def describe(self) -> dict:
        d = dict(self.info.__dict__)
        d["input_shape"] = list(self.info.input_shape)
        return d
