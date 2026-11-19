import getpass
import re
import shutil
import statistics
import subprocess
from collections import defaultdict

FIELDS = ["JobID", "JobName", "State", "ExitCode", "ElapsedRaw", "TimelimitRaw", "AllocCPUS", "TotalCPU",
          "ReqMem", "MaxRSS", "AllocTRES", "TRESUsageInAve", "Submit", "Start", "WorkDir"]
UNITS = {"K": 1 / 1024, "M": 1, "G": 1024, "T": 1024**2}
EARLY_FAILURE_S = 600
FAILED_STATES = ("FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL")


def parse_mem_mb(value: str, cpus: int = 1, nodes: int = 1) -> float | None:
    m = re.fullmatch(r"([\d.]+)([KMGT]?)([nc]?)", value.strip())
    if not m:
        return None
    mb = float(m.group(1)) * UNITS.get(m.group(2) or "M", 1)
    return mb * cpus if m.group(3) == "c" else mb * nodes if m.group(3) == "n" else mb


def parse_cpu_seconds(value: str) -> float | None:
    """sacct TotalCPU: [D-][HH:]MM:SS[.mmm]"""
    if not value:
        return None
    days = 0
    if "-" in value:
        d, value = value.split("-", 1)
        days = int(d)
    parts = [float(p) for p in value.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    h, m, s = parts
    return days * 86400 + h * 3600 + m * 60 + s


def parse_tres(value: str) -> dict:
    out = {}
    for item in filter(None, value.split(",")):
        k, _, v = item.partition("=")
        out[k] = v
    return out


def _int(value: str) -> int | None:
    return int(value) if value.isdigit() else None


def parse_sacct(text: str) -> list[dict]:
    """Group allocation and step rows (123, 123.batch, 123.0) into one record per job."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for line in text.strip().splitlines():
        parts = line.split("|")
        if len(parts) != len(FIELDS):
            continue
        row = dict(zip(FIELDS, parts))
        groups[row["JobID"].split(".")[0]].append(row)

    jobs = []
    for job_id, rows in groups.items():
        alloc = next((r for r in rows if r["JobID"] == job_id), rows[0])
        steps = [r for r in rows if r is not alloc]
        tres = parse_tres(alloc["AllocTRES"])
        cpus = _int(alloc["AllocCPUS"]) or 1
        nodes = _int(tres.get("node", "1")) or 1
        gpus = _int(tres.get("gres/gpu", "0")) or 0
        elapsed = _int(alloc["ElapsedRaw"]) or 0
        limit_min = _int(alloc["TimelimitRaw"])
        total_cpu = parse_cpu_seconds(alloc["TotalCPU"])
        req_mem = parse_mem_mb(alloc["ReqMem"], cpus, nodes) if alloc["ReqMem"] else None
        rss = [m for r in steps if (m := parse_mem_mb(r["MaxRSS"])) is not None]
        usage = [parse_tres(r["TRESUsageInAve"]) for r in steps]
        gpu_util = [float(u["gres/gpuutil"]) for u in usage if u.get("gres/gpuutil", "").replace(".", "").isdigit()]

        job = {
            "job_id": job_id,
            "name": alloc["JobName"],
            "state": alloc["State"].split()[0],
            "exit_code": alloc["ExitCode"],
            "submit": alloc["Submit"],
            "start": alloc["Start"],
            "workdir": alloc["WorkDir"],
            "elapsed_h": round(elapsed / 3600, 4),
            "timelimit_h": round(limit_min / 60, 4) if limit_min else None,
            "cpus": cpus,
            "gpus": gpus,
            "gpu_hours": round(elapsed / 3600 * gpus, 4),
            "cpu_eff": round(total_cpu / (elapsed * cpus), 3) if total_cpu is not None and elapsed else None,
            "mem_req_mb": req_mem,
            "mem_max_mb": max(rss) if rss else None,
            "gpu_util_pct": max(gpu_util) if gpu_util else None,
        }
        if req_mem and job["mem_max_mb"] is not None:
            job["mem_eff"] = round(job["mem_max_mb"] / req_mem, 3)
        if job["timelimit_h"]:
            job["time_used_frac"] = round(job["elapsed_h"] / job["timelimit_h"], 3)
        job["early_failure"] = job["state"].startswith(FAILED_STATES) and elapsed < EARLY_FAILURE_S
        jobs.append(job)
    return jobs


def _median(values):
    values = [v for v in values if v is not None]
    return round(statistics.median(values), 3) if values else None


def summarize(jobs: list[dict]) -> dict:
    failed = [j for j in jobs if j["state"].startswith(FAILED_STATES)]
    return {
        "n_jobs": len(jobs),
        "gpu_hours": round(sum(j["gpu_hours"] for j in jobs), 3),
        "gpu_hours_failed": round(sum(j["gpu_hours"] for j in failed), 3),
        "n_failed": len(failed),
        "n_early_failures": sum(j["early_failure"] for j in jobs),
        "n_timeouts": sum(j["state"].startswith("TIMEOUT") for j in jobs),
        "n_oom": sum(j["state"].startswith("OUT_OF_MEMORY") for j in jobs),
        "n_cancelled": sum(j["state"].startswith("CANCELLED") for j in jobs),
        "median_cpu_eff": _median(j["cpu_eff"] for j in jobs),
        "median_mem_eff": _median(j.get("mem_eff") for j in jobs),
        "median_time_used_frac": _median(j.get("time_used_frac") for j in jobs if not j["early_failure"]),
        "median_gpu_util_pct": _median(j["gpu_util_pct"] for j in jobs if j["gpus"]),
        "gpu_util_available": any(j["gpu_util_pct"] is not None for j in jobs),
    }


def sacct_available() -> bool:
    return shutil.which("sacct") is not None


def fetch(user: str | None = None, days: int = 30, name: str | None = None, run=subprocess.run) -> list[dict]:
    cmd = ["sacct", "-u", user or getpass.getuser(), "-S", f"now-{days}days", "--parsable2", "--noheader",
           "-o", ",".join(FIELDS)]
    if name:
        cmd += ["--name", name]
    out = run(cmd, capture_output=True, text=True, timeout=60, check=True)
    return parse_sacct(out.stdout)


def history(user: str | None = None, days: int = 30, name: str | None = None, limit: int = 50, run=None) -> dict:
    if run is None and not sacct_available():
        return {"available": False, "message": "sacct not found; Slurm history is only available on a cluster login node."}
    jobs = fetch(user, days, name, run or subprocess.run)
    jobs.sort(key=lambda j: j["submit"], reverse=True)
    return {"available": True, "days": days, "summary": summarize(jobs), "jobs": jobs[:limit]}
