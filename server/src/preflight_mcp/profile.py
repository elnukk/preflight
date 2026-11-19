import os
import re
import signal
import statistics
import subprocess
import threading
import time

import psutil

from .adapters import TelemetryAdapter

# Scripts report progress with lines like: [preflight] total_items=50000 epochs=10  /  [preflight] step=3 items=192
MARKER = re.compile(r"\[preflight\]\s+(.*)")
KV = re.compile(r"(\w+)=([-+\d.eE]+)")
TAIL = 30


def parse_marker(line: str) -> dict | None:
    m = MARKER.search(line)
    if not m:
        return None
    return {k: float(v) for k, v in KV.findall(m.group(1))} or None


def summarize_markers(markers: list[tuple[float, dict]]) -> dict:
    out: dict = {"steps_seen": 0}
    steps = []
    for t, m in markers:
        if "total_items" in m:
            out["total_items"] = int(m["total_items"])
        if "epochs" in m:
            out["epochs"] = int(m["epochs"])
        if "items" in m:
            steps.append((t, m["items"]))
    out["steps_seen"] = len(steps)
    if steps:
        out["time_to_first_step_s"] = round(steps[0][0], 3)
    if len(steps) >= 2 and steps[-1][0] > steps[0][0]:
        out["throughput_items_per_s"] = (steps[-1][1] - steps[0][1]) / (steps[-1][0] - steps[0][0])
        rates = [(b[1] - a[1]) / (b[0] - a[0]) for a, b in zip(steps, steps[1:]) if b[0] > a[0]]
        if len(rates) >= 2:
            q = statistics.quantiles(rates, n=4)
            out["throughput_p25"], out["throughput_p75"] = q[0], q[2]
    return out


def summarize_gpu(samples, adapter: TelemetryAdapter) -> dict:
    out = {"adapter": adapter.name, "measured": adapter.measured, "n_samples": len(samples)}
    if not samples:
        return out
    utils = [s.util_pct for s in samples]
    powers = [s.power_w for s in samples if s.power_w is not None]
    out.update(
        mean_util_pct=round(statistics.fmean(utils), 1),
        median_util_pct=round(statistics.median(utils), 1),
        peak_mem_mb=max(s.mem_used_mb for s in samples),
        mem_total_mb=max(s.mem_total_mb for s in samples),
        n_gpus=len({s.gpu for s in samples}),
    )
    if powers:
        out["mean_power_w"] = round(statistics.fmean(powers), 1)
    return out


class _TreeUsage:
    def __init__(self, pid: int):
        self.root = psutil.Process(pid)
        self.procs: dict[int, psutil.Process] = {}

    def sample(self) -> tuple[float, float]:
        try:
            tree = [self.root, *self.root.children(recursive=True)]
        except psutil.NoSuchProcess:
            return 0.0, 0.0
        cpu = rss = 0.0
        for p in tree:
            p = self.procs.setdefault(p.pid, p)
            try:
                cpu += p.cpu_percent(None)
                rss += p.memory_info().rss
            except psutil.NoSuchProcess:
                pass
        return cpu, rss / 2**20


def _pump(stream, sink: list, markers: list | None, start: float):
    for line in stream:
        sink.append(line.rstrip("\n"))
        if markers is not None and (m := parse_marker(line)):
            markers.append((time.monotonic() - start, m))


def run_profile(cmd: str, cwd: str, subset: int, max_seconds: float,
                adapter: TelemetryAdapter, interval: float = 0.2) -> dict:
    env = {**os.environ, "PREFLIGHT_SUBSET": str(subset), "PREFLIGHT_PROFILE": "1", "PYTHONUNBUFFERED": "1"}
    start = time.monotonic()
    proc = subprocess.Popen(cmd, shell=True, cwd=cwd, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    out_lines, err_lines, markers = [], [], []
    pumps = [threading.Thread(target=_pump, args=(proc.stdout, out_lines, markers, start), daemon=True),
             threading.Thread(target=_pump, args=(proc.stderr, err_lines, None, start), daemon=True)]
    for t in pumps:
        t.start()

    usage = _TreeUsage(proc.pid)
    gpu, cpu, rss = [], [], []  # (t, value) pairs
    timed_out = False
    while proc.poll() is None:
        if time.monotonic() - start > max_seconds:
            timed_out = True
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
            break
        now = time.monotonic() - start
        gpu.extend((now, s) for s in adapter.sample())
        c, r = usage.sample()
        cpu.append((now, c))
        rss.append(r)
        time.sleep(interval)
    proc.wait()
    for t in pumps:
        t.join(2)

    markers_info = summarize_markers(markers)
    t0 = markers_info.get("time_to_first_step_s")
    steady = lambda pairs: [v for t, v in pairs if t0 is None or t >= t0]
    gpu_window, cpu_window = steady(gpu), steady(cpu[1:])
    window = "steady_state"
    if len(cpu_window) < 3:
        gpu_window, cpu_window, window = [v for _, v in gpu], [v for _, v in cpu[1:]], "all"

    return {
        "cmd": cmd,
        "cwd": cwd,
        "subset": subset,
        "exit_code": proc.returncode,
        "timed_out": timed_out,
        "wall_s": round(time.monotonic() - start, 2),
        **markers_info,
        "window": window,
        "gpu": summarize_gpu(gpu_window, adapter),
        "cpu": {
            "mean_pct": round(statistics.fmean(cpu_window), 1) if cpu_window else None,
            "peak_rss_mb": round(max(rss), 1) if rss else None,
        },
        "stdout_tail": out_lines[-TAIL:],
        "stderr_tail": err_lines[-TAIL:],
    }
