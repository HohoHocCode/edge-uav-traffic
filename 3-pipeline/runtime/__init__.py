"""Backend resolution for the inference runtime.

One import surface, several execution targets. The pipeline code never asks
"am I on the board?" — it asks the registry for a session and reads
``session.backend_name`` when it needs to label a benchmark row.

Resolution order for ``backend="auto"``:
    1. onnxruntime with QNNExecutionProvider   (Hexagon NPU)
    2. onnxruntime with CPUExecutionProvider   (Kryo / x86 host)

The QNN path is only available on an aarch64 device with onnxruntime-qnn
installed; on a Windows/x86 host it silently falls back to CPU, which is what
lets the exact same script be developed on the laptop and run on the board.
"""

from __future__ import annotations

from .ort_backend import OrtSession

__all__ = ["OrtSession", "create_session", "available_backends"]


def available_backends() -> list[str]:
    """Backends actually usable in this process, most-preferred first."""
    import onnxruntime as ort

    providers = ort.get_available_providers()
    out = []
    if "QNNExecutionProvider" in providers:
        out.append("onnxruntime-qnn")
    out.append("onnxruntime-cpu")
    return out


def create_session(model_path: str, backend: str = "auto", **kwargs) -> OrtSession:
    """Build an inference session for ``model_path``.

    ``backend`` is one of ``auto``, ``onnxruntime-qnn``, ``onnxruntime-cpu``.
    Passing an explicit backend that is unavailable raises rather than falling
    back, because a benchmark row silently measured on the wrong compute unit
    is worse than no row at all.
    """
    avail = available_backends()

    if backend == "auto":
        chosen = avail[0]
    elif backend in avail:
        chosen = backend
    else:
        raise RuntimeError(
            f"backend {backend!r} is not available in this process. "
            f"Available: {avail}. On the board, install onnxruntime-qnn; "
            f"on the host, use 'onnxruntime-cpu' or 'auto'."
        )

    return OrtSession(model_path, backend_name=chosen, **kwargs)
