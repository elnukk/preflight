from pathlib import Path

from mcp.server.mcpserver import MCPServer

from . import diagnose as diag
from . import estimate as est
from . import slurm, store, study
from .adapters import select_adapter
from .profile import run_profile

mcp = MCPServer("preflight")


@mcp.tool()
def init_study(project_dir: str, name: str | None = None, outputs: list[dict] | None = None,
               budget: dict | None = None, rates: dict | None = None, hook_mode: str | None = None) -> dict:
    """Create or update .preflight/study.yaml. Only the fields you pass are changed.

    outputs: [{"path": "results/metrics.json", "metrics": {"val_acc": {"abs": 0.002}}}] (abs or rel tolerance)
    budget: {"gpu_hours": 10, "usd": 50}
    rates: {"usd_per_gpu_hour", "gpu_tdp_watts", "pue", "kg_co2_per_kwh"}
    hook_mode: "warn" (ask before unestimated launches) or "block" (refuse them)
    """
    root = Path(project_dir).resolve()
    existed = study.study_path(root).is_file()
    if not existed and not name:
        return {"error": "name is required when creating a study"}
    try:
        path = study.update(root, name=name, outputs=outputs, budget=budget, rates=rates, hook_mode=hook_mode)
    except ValueError as e:
        return {"error": str(e)}
    saved = study.load(root)
    saved.pop("_found")
    return {"path": str(path), "created": not existed, "study": saved}


@mcp.tool()
def capabilities() -> dict:
    """Which telemetry adapter is active and what it can measure, and whether Slurm accounting is available.

    Adapter selection: PREFLIGHT_ADAPTER=auto (nvml, then nvidia-smi, then mock), or nvml / nvidia-smi / mock.
    Mock adapters do not read real hardware.
    """
    return {**select_adapter().capabilities(), "slurm_accounting": slurm.sacct_available()}


@mcp.tool()
def profile_run(cmd: str, cwd: str, subset: int = 2000, max_seconds: int = 120) -> dict:
    """Run a short profile of a job on a data subset and record throughput, GPU/CPU utilization and memory.

    The command runs with PREFLIGHT_SUBSET=<subset> set; the script should honor it (or pass the script's own
    subset flag in `cmd`). Throughput is measured from stdout lines like
    `[preflight] total_items=50000 epochs=10` and `[preflight] step=3 items=192` (items is cumulative).
    """
    root = store.find_root(Path(cwd))
    result = run_profile(cmd, str(Path(cwd).resolve()), subset, max_seconds, select_adapter())
    return store.save(root, "profiles", result)


@mcp.tool()
def estimate_job(script_path: str, sbatch_path: str | None = None, profile_id: str | None = None,
                 total_items: int | None = None, epochs: int | None = None, gpus: int | None = None) -> dict:
    """Estimate runtime, GPU-hours, cost, energy and CO2 for a full job, as low/point/high ranges.

    Pass a profile_id from profile_run for a measured estimate. Pass total_items/epochs/gpus when you have read
    them from the script or launch command; they override values the script reported or argparse defaults.
    Recording an estimate also lets the launch hook allow this job.
    """
    script = Path(script_path).resolve()
    root = store.find_root(script.parent)
    profile = store.load(root, "profiles", profile_id) if profile_id else None
    result = est.estimate(study.load(root), str(script), sbatch_path, profile, total_items, epochs, gpus)
    if result["basis"] == "insufficient":
        return result
    result["keys"] = [str(script)] + ([str(Path(sbatch_path).resolve())] if sbatch_path else [])
    return store.save(root, "estimates", result)


@mcp.tool()
def slurm_history(user: str | None = None, days: int = 30, name: str | None = None, limit: int = 50) -> dict:
    """Past Slurm jobs from sacct: requested vs used CPUs, memory and time, GPU-hours, failures and timeouts.

    `name` filters by job name. GPU utilization appears only if the cluster records gres/gpuutil in accounting.
    """
    return slurm.history(user, days, name, limit)


@mcp.tool()
def diagnose(script_path: str | None = None, sbatch_path: str | None = None, profile_id: str | None = None,
             estimate_id: str | None = None, include_history: bool = False, job_name: str | None = None,
             days: int = 30) -> dict:
    """Run rule-based detectors over a profile, an estimate, the script, the sbatch file and optionally Slurm history.

    Each finding has a risk class: failure_risk (job likely fails or loses work), safe (does not change results),
    or may_change_results (must be verified against the study's tolerances before recommending), plus evidence,
    a suggestion and a playbook page in the compute-efficiency skill.
    """
    start = Path(script_path).resolve().parent if script_path else Path.cwd()
    root = store.find_root(start)
    profile = store.load(root, "profiles", profile_id) if profile_id else None
    estimate = store.load(root, "estimates", estimate_id) if estimate_id else None
    source = Path(script_path).read_text() if script_path and script_path.endswith(".py") else None
    notes = []
    summary = None
    if include_history:
        hist = slurm.history(days=days, name=job_name, limit=0)
        if hist["available"]:
            summary = hist["summary"]
        else:
            notes.append(hist["message"])
    sbatch = est.parse_sbatch(Path(sbatch_path).read_text()) if sbatch_path else None
    result = diag.diagnose(profile, estimate, source, summary, sbatch)
    result["notes"] = notes
    return result


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
