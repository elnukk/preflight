from dataclasses import dataclass
from typing import Protocol


@dataclass
class GpuSample:
    gpu: int
    util_pct: float
    mem_used_mb: float
    mem_total_mb: float
    power_w: float | None = None


class TelemetryAdapter(Protocol):
    name: str
    measured: bool  # False when numbers are not read from real hardware

    def capabilities(self) -> dict: ...

    def sample(self) -> list[GpuSample]: ...
