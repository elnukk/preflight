import os

from .base import GpuSample, TelemetryAdapter
from .mock import MockAdapter

__all__ = ["GpuSample", "TelemetryAdapter", "MockAdapter", "select_adapter"]


def _nvml():
    from .nvml import NvmlAdapter
    return NvmlAdapter()


def _nvidia_smi():
    from .nvidia_smi import NvidiaSmiAdapter
    return NvidiaSmiAdapter()


def _mock():
    return MockAdapter(os.environ.get("PREFLIGHT_MOCK_TRACE"))


FACTORIES = {"nvml": _nvml, "nvidia-smi": _nvidia_smi, "mock": _mock}


def select_adapter(name: str | None = None) -> TelemetryAdapter:
    """PREFLIGHT_ADAPTER=auto (default) tries nvml, then nvidia-smi, then falls back to mock."""
    name = name or os.environ.get("PREFLIGHT_ADAPTER", "auto")
    if name != "auto":
        if name not in FACTORIES:
            raise ValueError(f"unknown telemetry adapter {name!r} (available: auto, {', '.join(FACTORIES)})")
        return FACTORIES[name]()
    errors = []
    for key in ("nvml", "nvidia-smi"):
        try:
            return FACTORIES[key]()
        except Exception as e:  # missing library, driver, binary or GPU
            errors.append(f"{key}: {e}")
    adapter = _mock()
    adapter.fallback_reason = "; ".join(errors)
    return adapter
