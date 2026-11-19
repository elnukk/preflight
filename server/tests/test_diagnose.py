from preflight_mcp.diagnose import dataloader_calls, diagnose

LOADER_SRC = """
import torch
from torch.utils.data import DataLoader
loader = DataLoader(ds, batch_size=64, shuffle=True)
eval_loader = DataLoader(ds, batch_size=64, num_workers=4, pin_memory=True)
model.to(device)
"""


def prof(util=70.0, cpu=50.0, peak=20000, total=40960, measured=True, **kw):
    base = {"exit_code": 0, "timed_out": False, "steps_seen": 20, "time_to_first_step_s": 2.0, "wall_s": 30,
            "gpu": {"adapter": "nvml" if measured else "mock", "measured": measured, "mean_util_pct": util,
                    "median_util_pct": util, "peak_mem_mb": peak, "mem_total_mb": total, "n_gpus": 1},
            "cpu": {"mean_pct": cpu}}
    base.update(kw)
    return base


def ids(result):
    return [f["id"] for f in result["findings"]]


def find(result, fid):
    return next(f for f in result["findings"] if f["id"] == fid)


def test_dataloader_calls():
    calls = dataloader_calls(LOADER_SRC)
    assert [(c["num_workers"], c["pin_memory"]) for c in calls] == [(None, None), (4, True)]


def test_dataloader_bound_from_source():
    r = diagnose(prof(util=25), source=LOADER_SRC)
    f = find(r, "dataloader_bound")
    assert f["risk"] == "safe" and f["confidence"] == "high"
    assert any("line 4" in e for e in f["evidence"])


def test_io_bound_and_simulated_confidence():
    f = find(diagnose(prof(util=20, cpu=10, measured=False)), "io_bound")
    assert f["confidence"] == "low" and any("simulated" in e for e in f["evidence"])


def test_memory_findings():
    assert "memory_near_limit" in ids(diagnose(prof(peak=39000)))
    r = diagnose(prof(peak=8000))
    assert find(r, "memory_headroom")["risk"] == "may_change_results"
    assert "oversized_gpu" in ids(r)


def test_profile_failed_classifies_oom():
    p = prof(exit_code=1, stderr_tail=["Traceback ...", "torch.OutOfMemoryError: CUDA out of memory."])
    f = find(diagnose(p), "profile_failed")
    assert f["risk"] == "failure_risk" and "out of memory" in f["title"]


def test_slow_startup_mentions_uncached_features():
    src = "feats = model.encode(texts)\nclf.fit(feats, y)\n"
    r = diagnose(prof(time_to_first_step_s=120, wall_s=150), source=src)
    assert any("caching" in e for e in find(r, "slow_startup")["evidence"])
    assert "uncached_features" not in ids(r)


def test_source_and_estimate_findings():
    est = {"runtime_h": {"low": 3.0, "point": 3.5, "high": 4.0}, "sbatch": {"time_limit_h": 2.0},
           "inputs": {"gpus": 2}, "warnings": ["exceeds the study's dollar budget ($5)"]}
    r = diagnose(estimate=est, source="for x in data:\n    loss.backward()\n")
    assert {"no_checkpoint", "no_subset_flag", "time_limit_too_short", "over_budget"} <= set(ids(r))
    assert find(r, "time_limit_too_short")["est_savings"]["gpu_hours_at_risk"] == 4.0


def test_mixed_precision_and_ordering():
    r = diagnose(prof(util=90, peak=39000), source=LOADER_SRC)
    assert find(r, "no_mixed_precision")["risk"] == "may_change_results"
    risks = [f["risk"] for f in r["findings"]]
    assert risks == sorted(risks, key=["failure_risk", "safe", "may_change_results"].index)
    assert "no_mixed_precision" not in ids(diagnose(prof(util=90), source=LOADER_SRC + "torch.autocast('cuda')"))


def test_history_findings():
    from pathlib import Path
    from preflight_mcp import slurm

    summary = slurm.summarize(slurm.parse_sacct((Path(__file__).parent / "fixtures" / "sacct.txt").read_text()))
    got = set(ids(diagnose(history_summary=summary)))
    assert {"early_failures", "timeouts", "cpus_overrequested", "memory_overrequested",
            "time_overrequested", "low_gpu_util_history"} <= got


def test_markers_missing():
    assert "no_progress_markers" in ids(diagnose(prof(steps_seen=0)))


def test_num_workers_resolved_from_argparse():
    src = ('import torch\np.add_argument("--num-workers", type=int, default=0)\n'
           'loader = DataLoader(ds, batch_size=8, num_workers=args.num_workers)\n')
    assert dataloader_calls(src)[0]["num_workers"] == 0
    assert "dataloader_bound" in ids(diagnose(prof(util=20, cpu=30), source=src))


CPU_SRC = "import random, time\nfor i in range(10):\n    time.sleep(0.01)\n"


def test_cpu_script_ignores_gpu_telemetry():
    r = diagnose(prof(util=20, cpu=1, peak=8000, measured=False), source=CPU_SRC)
    got = ids(r)
    assert "io_bound" in got
    assert not {"memory_headroom", "oversized_gpu", "dataloader_bound", "gpu_underutilized"} & set(got)
    assert any("GPU telemetry was ignored" in e for e in find(r, "io_bound")["evidence"])


def test_gpu_requested_unused():
    est = {"gpu_hours": {"point": 0.75}, "sbatch": {"gpus": 1}, "runtime_h": {}, "inputs": {"gpus": 1}}
    f = find(diagnose(estimate=est, source=CPU_SRC), "gpu_requested_unused")
    assert f["risk"] == "safe" and f["est_savings"] == {"gpu_hours": 0.75}
    assert "gpu_requested_unused" not in ids(diagnose(estimate=est, source=LOADER_SRC))


def test_memory_overrequested_single_run():
    p = prof(cpu=50)
    p["cpu"]["peak_rss_mb"] = 13
    f = find(diagnose(p, source=CPU_SRC, sbatch={"mem": "16G"}), "memory_overrequested")
    assert f["confidence"] == "low" and "16384 MB" in f["evidence"][0]
    p["cpu"]["peak_rss_mb"] = 9000
    assert "memory_overrequested" not in ids(diagnose(p, source=CPU_SRC, sbatch={"mem": "16G"}))
