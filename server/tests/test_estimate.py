import pytest

from preflight_mcp import study
from preflight_mcp.estimate import estimate, parse_sbatch, parse_time_limit, static_hints


@pytest.mark.parametrize("value,hours", [
    ("30", 0.5), ("10:00", 10 / 60), ("02:00:00", 2), ("1-00", 24), ("1-12:30", 36.5), ("2-01:00:00", 49),
])
def test_time_limit(value, hours):
    assert parse_time_limit(value) == pytest.approx(hours)


def test_parse_sbatch():
    text = "#SBATCH --gres=gpu:a100:2\n#SBATCH -t 04:00:00\n#SBATCH --nodes 2\n#SBATCH -c 8\npython x.py"
    assert parse_sbatch(text) == {"gpus": 4, "time_limit_h": 4, "nodes": 2, "cpus_per_task": 8}
    assert parse_sbatch("#SBATCH --gpus=3")["gpus"] == 3


def test_static_hints():
    src = 'p.add_argument("--epochs", type=int, default=7)\np.add_argument("--num-samples", default=1000)\n'
    assert static_hints(src) == {"epochs": 7, "total_items": 1000}
    assert static_hints("def broken(") == {}


def profile(**kw):
    base = {"id": "p1", "exit_code": 0, "timed_out": False, "throughput_items_per_s": 1000.0,
            "throughput_p25": 800.0, "throughput_p75": 1250.0, "time_to_first_step_s": 0.0,
            "total_items": 3_600_000, "epochs": 1,
            "gpu": {"adapter": "mock", "measured": False, "mean_power_w": 200.0,
                    "peak_mem_mb": 10_000, "mem_total_mb": 40_000}}
    base.update(kw)
    return base


def test_measured_estimate_math():
    s = study.load(__import__("pathlib").Path("/nonexistent"))
    r = estimate(s, profile=profile(), gpus=2)
    assert r["runtime_h"] == {"low": 0.8, "point": 1.0, "high": 1.25}
    assert r["gpu_hours"]["point"] == 2.0
    assert r["usd"]["point"] == 2.0 * s["rates"]["usd_per_gpu_hour"]
    assert r["kwh"]["point"] == pytest.approx(2.0 * 200 * 1.3 / 1000)
    assert any("mock" in w for w in r["warnings"])


def test_warnings(tmp_path):
    sb = tmp_path / "job.sbatch"
    sb.write_text("#SBATCH --gres=gpu:1\n#SBATCH --time=00:30:00\n")
    s = study.load(tmp_path)
    s["budget"]["usd"] = 1
    gpu = {"adapter": "nvml", "measured": True, "peak_mem_mb": 39_000, "mem_total_mb": 40_000}
    r = estimate(s, sbatch_path=str(sb), profile=profile(exit_code=1, gpu=gpu))
    text = " ".join(r["warnings"])
    for needle in ("time limit", "exited with code 1", "out of memory", "dollar budget"):
        assert needle in text


def test_upper_bound_and_insufficient(tmp_path):
    sb = tmp_path / "job.sbatch"
    sb.write_text("#SBATCH --gpus=4\n#SBATCH --time=2:00:00\n")
    s = study.load(tmp_path)
    r = estimate(s, sbatch_path=str(sb))
    assert r["gpu_hours"] == {"low": None, "point": None, "high": 8.0}
    assert estimate(s)["basis"] == "insufficient"
