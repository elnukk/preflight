import json
from pathlib import Path

from .base import GpuSample

DEFAULT_TRACE = Path(__file__).resolve().parent.parent / "fixtures" / "dataloader_bound.json"


class MockAdapter:
    """Replays a recorded GPU trace in a loop. Lets the full pipeline run on machines without GPUs."""

    name = "mock"
    measured = False

    def __init__(self, trace_path: str | Path | None = None):
        self.trace_path = Path(trace_path or DEFAULT_TRACE)
        trace = json.loads(self.trace_path.read_text())
        self.gpu_name = trace.get("gpu_name", "unknown")
        self.frames = trace["samples"]
        self.fallback_reason: str | None = None
        self._i = 0

    def capabilities(self) -> dict:
        caps = {
            "adapter": self.name,
            "measured": self.measured,
            "trace": str(self.trace_path),
            "gpu_name": self.gpu_name,
            "metrics": ["util_pct", "mem_used_mb", "mem_total_mb", "power_w"],
        }
        if self.fallback_reason:
            caps["fallback_reason"] = f"no real GPU telemetry ({self.fallback_reason})"
        return caps

    def sample(self) -> list[GpuSample]:
        frame = self.frames[self._i % len(self.frames)]
        self._i += 1
        return [GpuSample(**g) for g in frame]
