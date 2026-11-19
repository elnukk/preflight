from pathlib import Path
from types import SimpleNamespace

import pytest

from preflight_mcp import slurm

SACCT = (Path(__file__).parent / "fixtures" / "sacct.txt").read_text()


@pytest.mark.parametrize("value,cpus,mb", [
    ("32G", 1, 32768), ("4000M", 1, 4000), ("4Gc", 8, 32768), ("2.5G", 1, 2560), ("123456K", 1, 120.5625), ("", 1, None),
])
def test_parse_mem(value, cpus, mb):
    assert slurm.parse_mem_mb(value, cpus) == (pytest.approx(mb) if mb else None)


@pytest.mark.parametrize("value,secs", [("01:00:00", 3600), ("12:34.500", 754.5), ("2-00:00:00", 172800)])
def test_parse_cpu(value, secs):
    assert slurm.parse_cpu_seconds(value) == secs


def test_parse_sacct_jobs():
    jobs = {j["job_id"]: j for j in slurm.parse_sacct(SACCT)}
    assert set(jobs) == {"101", "102", "103", "104"}
    j = jobs["101"]
    assert (j["gpus"], j["cpu_eff"], j["mem_eff"], j["gpu_util_pct"], j["time_used_frac"]) == (1, 0.125, 0.125, 22, 0.1)
    assert jobs["102"]["early_failure"] and not jobs["104"]["early_failure"]
    assert jobs["103"]["mem_req_mb"] == 32768 and jobs["103"]["gpu_hours"] == 20
    assert jobs["104"]["state"] == "CANCELLED" and jobs["104"]["mem_max_mb"] is None


def test_summary():
    s = slurm.summarize(slurm.parse_sacct(SACCT))
    assert s["n_jobs"] == 4 and s["n_early_failures"] == 1 and s["n_timeouts"] == 1 and s["n_cancelled"] == 1
    assert s["gpu_hours"] == pytest.approx(21.033, abs=1e-3)
    assert s["gpu_hours_failed"] == pytest.approx(20.033, abs=1e-3)
    assert s["median_mem_eff"] == 0.125 and s["median_gpu_util_pct"] == 22


def test_history_with_fake_runner():
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        return SimpleNamespace(stdout=SACCT)

    h = slurm.history(user="u", days=7, name="train", limit=2, run=run)
    assert h["available"] and len(h["jobs"]) == 2 and h["jobs"][0]["job_id"] == "104"
    assert calls[0][:5] == ["sacct", "-u", "u", "-S", "now-7days"] and "--name" in calls[0]


def test_history_without_sacct(monkeypatch):
    monkeypatch.setattr(slurm, "sacct_available", lambda: False)
    assert slurm.history()["available"] is False
