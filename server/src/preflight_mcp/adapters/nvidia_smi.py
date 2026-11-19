import shutil
import subprocess

from .base import GpuSample
from .visible import visible_gpus

QUERY = "index,uuid,name,utilization.gpu,memory.used,memory.total,power.draw"


def _num(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None  # "[N/A]", "[Not Supported]"


def parse_query(text: str) -> list[dict]:
    rows = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 7:
            continue
        idx, uuid, name, util, used, total, power = parts
        rows.append({"index": int(idx), "uuid": uuid, "name": name, "util_pct": _num(util),
                     "mem_used_mb": _num(used), "mem_total_mb": _num(total), "power_w": _num(power)})
    return rows


class NvidiaSmiAdapter:
    """Fallback when NVML bindings fail: polls `nvidia-smi --query-gpu`. Slower per sample than NVML."""

    name = "nvidia-smi"
    measured = True

    def __init__(self, binary: str | None = None):
        self.binary = binary or shutil.which("nvidia-smi")
        if not self.binary:
            raise RuntimeError("nvidia-smi not found")
        rows = self._query()
        if not rows:
            raise RuntimeError("nvidia-smi reported no GPUs")
        keep = set(visible_gpus([r["uuid"] for r in rows]))
        self.indices = [r["index"] for r in rows if r["index"] in keep]
        self.names = {r["index"]: r["name"] for r in rows}

    def _query(self) -> list[dict]:
        out = subprocess.run([self.binary, f"--query-gpu={QUERY}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10, check=True)
        return parse_query(out.stdout)

    def capabilities(self) -> dict:
        return {
            "adapter": self.name,
            "measured": True,
            "gpus": [{"index": i, "name": self.names[i]} for i in self.indices],
            "metrics": ["util_pct", "mem_used_mb", "mem_total_mb", "power_w"],
            "notes": ["utilization is per device and includes other processes sharing the GPU"],
        }

    def sample(self) -> list[GpuSample]:
        try:
            rows = self._query()
        except (subprocess.SubprocessError, OSError):
            return []
        return [GpuSample(r["index"], r["util_pct"] or 0.0, r["mem_used_mb"] or 0.0, r["mem_total_mb"] or 0.0,
                          r["power_w"]) for r in rows if r["index"] in self.indices]
