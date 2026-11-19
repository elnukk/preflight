from .base import GpuSample
from .visible import visible_gpus

METRICS = ["util_pct", "mem_used_mb", "mem_total_mb", "power_w"]


class NvmlAdapter:
    """Reads device-level telemetry through NVML. Utilization covers every process on the GPU, not just this job."""

    name = "nvml"
    measured = True

    def __init__(self):
        import pynvml

        pynvml.nvmlInit()
        self.nv = pynvml
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
        uuids = [pynvml.nvmlDeviceGetUUID(h) for h in handles]
        self.devices = [(i, handles[i]) for i in visible_gpus(uuids)]
        if not self.devices:
            raise RuntimeError("NVML found no GPUs")
        self.names = [pynvml.nvmlDeviceGetName(h) for _, h in self.devices]

    def capabilities(self) -> dict:
        return {
            "adapter": self.name,
            "measured": True,
            "gpus": [{"index": i, "name": n} for (i, _), n in zip(self.devices, self.names)],
            "metrics": METRICS,
            "notes": ["utilization is per device and includes other processes sharing the GPU"],
        }

    def sample(self) -> list[GpuSample]:
        out = []
        for i, h in self.devices:
            try:
                util = self.nv.nvmlDeviceGetUtilizationRates(h).gpu
                mem = self.nv.nvmlDeviceGetMemoryInfo(h)
            except self.nv.NVMLError:
                continue
            try:
                power = self.nv.nvmlDeviceGetPowerUsage(h) / 1000
            except self.nv.NVMLError:
                power = None
            out.append(GpuSample(i, float(util), mem.used / 2**20, mem.total / 2**20, power))
        return out
