import sys

from preflight_mcp.adapters import MockAdapter
from preflight_mcp.profile import parse_marker, run_profile


def test_parse_marker():
    assert parse_marker("[preflight] step=3 items=192") == {"step": 3, "items": 192}
    assert parse_marker("loss 0.3") is None


def test_profile_toy(toy):
    cmd = f"{sys.executable} train.py --io-ms 5"
    r = run_profile(cmd, str(toy), subset=640, max_seconds=30, adapter=MockAdapter(), interval=0.05)
    assert r["exit_code"] == 0 and not r["timed_out"]
    assert r["total_items"] == 1_200_000 and r["epochs"] == 5
    assert r["steps_seen"] == 10
    assert r["throughput_items_per_s"] > 0
    assert r["gpu"]["measured"] is False and r["gpu"]["n_samples"] > 0


def test_profile_timeout(tmp_path):
    r = run_profile(f"{sys.executable} -c 'import time; time.sleep(30)'", str(tmp_path), 10, 1, MockAdapter(), 0.1)
    assert r["timed_out"] and r["wall_s"] < 10


def test_profile_failure(tmp_path):
    r = run_profile(f"{sys.executable} -c 'raise SystemExit(\"boom\")'", str(tmp_path), 10, 10, MockAdapter(), 0.1)
    assert r["exit_code"] == 1 and "boom" in r["stderr_tail"][-1]


def test_steady_state_window(tmp_path):
    script = tmp_path / "slow_start.py"
    script.write_text("import time\ntime.sleep(1.0)\n"
                      "for i in range(1, 8):\n    print(f'[preflight] step={i} items={i * 10}', flush=True)\n"
                      "    time.sleep(0.15)\n")
    r = run_profile(f"{sys.executable} {script}", str(tmp_path), 10, 30, MockAdapter(), 0.05)
    assert r["window"] == "steady_state" and r["time_to_first_step_s"] >= 1.0
    assert r["gpu"]["n_samples"] < r["wall_s"] / 0.05
