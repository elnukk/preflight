import ast
import re
import shlex
from pathlib import Path

HINT_NAMES = {
    "epochs": "epochs", "num_epochs": "epochs", "n_epochs": "epochs",
    "num_samples": "total_items", "n_samples": "total_items", "max_samples": "total_items",
    "dataset_size": "total_items",
    "batch_size": "batch_size", "max_steps": "max_steps",
}
SBATCH = re.compile(r"^#SBATCH\s+(.*)$", re.M)


def parse_time_limit(value: str) -> float | None:
    """Slurm time formats: M, M:S, H:M:S, D-H, D-H:M, D-H:M:S. Returns hours."""
    days = 0
    if "-" in value:
        d, value = value.split("-", 1)
        days = int(d)
        parts = [int(p) for p in value.split(":")]
        h, m, s = (parts + [0, 0])[:3]
    else:
        parts = [int(p) for p in value.split(":")]
        if len(parts) == 1:
            h, m, s = 0, parts[0], 0
        elif len(parts) == 2:
            h, m, s = 0, parts[0], parts[1]
        else:
            h, m, s = parts[:3]
    return days * 24 + h + m / 60 + s / 3600


def _sbatch_flags(line: str):
    tokens = shlex.split(line, comments=True)
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("-") and "=" in tok:
            yield tuple(tok.split("=", 1))
        elif tok.startswith("-") and i + 1 < len(tokens) and not tokens[i + 1].startswith("-"):
            yield tok, tokens[i + 1]
            i += 1
        elif tok.startswith("-"):
            yield tok, ""
        i += 1


def parse_sbatch(text: str) -> dict:
    out: dict = {}
    for line in SBATCH.findall(text):
        for key, val in _sbatch_flags(line):
            if key == "--gres" and (m := re.search(r"gpu(?::[\w-]+)?:(\d+)", val)):
                out["gpus"] = int(m.group(1))
            elif key in ("--gpus", "-G", "--gpus-per-node", "--gpus-per-task") and (m := re.search(r"(\d+)$", val)):
                out["gpus"] = int(m.group(1))
            elif key in ("--time", "-t") and val:
                out["time_limit_h"] = parse_time_limit(val)
            elif key in ("--nodes", "-N") and val.isdigit():
                out["nodes"] = int(val)
            elif key in ("--cpus-per-task", "-c") and val.isdigit():
                out["cpus_per_task"] = int(val)
            elif key == "--mem" and val:
                out["mem"] = val
    if "gpus" in out and out.get("nodes", 1) > 1:
        out["gpus"] *= out["nodes"]
    return out


def static_hints(source: str) -> dict:
    """argparse defaults for well-known size/length arguments."""
    hints = {}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return hints
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "add_argument"):
            continue
        flags = [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
        name = next((f.lstrip("-").replace("-", "_") for f in flags if f.startswith("--")), None)
        default = next((k.value for k in node.keywords if k.arg == "default"), None)
        if name in HINT_NAMES and isinstance(default, ast.Constant) and isinstance(default.value, (int, float)):
            hints[HINT_NAMES[name]] = default.value
    return hints


def _range(low, point, high, sig=3) -> dict:
    r = lambda v: None if v is None else float(f"{v:.{sig}g}")
    return {"low": r(low), "point": r(point), "high": r(high)}


def _scale(rng: dict, factor: float) -> dict:
    return _range(*(None if rng[k] is None else rng[k] * factor for k in ("low", "point", "high")))


def estimate(study: dict, script_path: str | None = None, sbatch_path: str | None = None,
             profile: dict | None = None, total_items: int | None = None,
             epochs: int | None = None, gpus: int | None = None) -> dict:
    assumptions, warnings = [], []
    sb = parse_sbatch(Path(sbatch_path).read_text()) if sbatch_path else {}
    hints = static_hints(Path(script_path).read_text()) if script_path and script_path.endswith(".py") else {}
    profile = profile or {}

    def pick(name, explicit):
        if explicit is not None:
            return explicit
        if profile.get(name) is not None:
            assumptions.append(f"{name}={profile[name]} as reported by the script during profiling")
            return profile[name]
        if name in hints:
            assumptions.append(f"{name}={hints[name]} from the argparse default; launch args may override it")
            return hints[name]
        return None

    total_items = pick("total_items", total_items)
    epochs = pick("epochs", epochs) or 1
    if gpus is None:
        gpus = sb.get("gpus", 1)
        assumptions.append(f"gpus={gpus} " + ("from sbatch" if "gpus" in sb else "(default; no sbatch given)"))

    tp = profile.get("throughput_items_per_s")
    limit = sb.get("time_limit_h")
    if tp and total_items:
        work = total_items * epochs
        startup = profile.get("time_to_first_step_s", 0) / 3600
        p25, p75 = profile.get("throughput_p25"), profile.get("throughput_p75")
        point = work / tp / 3600 + startup
        if p25 and p75:
            low, high = work / p75 / 3600 + startup, work / p25 / 3600 + startup
        else:
            low, high = point * 0.7, point * 1.3
            assumptions.append("too few progress steps for a measured spread; using ±30%")
        runtime = _range(low, point, high)
        basis = "measured throughput on a data subset"
        assumptions.append("subset throughput is representative of the full run with the same GPU count")
    elif limit:
        runtime = _range(None, None, limit)
        basis = "sbatch time limit (upper bound only)"
        warnings.append("no measured throughput; run profile_run for a real estimate")
    else:
        return {
            "basis": "insufficient",
            "message": "Need a profile with progress markers plus total_items, or an sbatch time limit.",
            "hints": hints,
            "sbatch": sb,
        }

    rates = study["rates"]
    gpu_hours = _scale(runtime, gpus)
    gpu_info = profile.get("gpu", {})
    power_w = gpu_info.get("mean_power_w") or rates["gpu_tdp_watts"]
    if "mean_power_w" not in gpu_info:
        assumptions.append(f"GPU power assumed at TDP ({rates['gpu_tdp_watts']} W)")
    kwh = _scale(gpu_hours, power_w * rates["pue"] / 1000)

    if gpu_info and not gpu_info.get("measured", True):
        warnings.append(f"GPU telemetry is from the '{gpu_info.get('adapter')}' adapter, not real hardware")
    if profile.get("exit_code") not in (None, 0) and not profile.get("timed_out"):
        warnings.append(f"profile run exited with code {profile['exit_code']}; fix this before launching")
    if limit and runtime["high"] and runtime["high"] > limit:
        warnings.append(f"estimated runtime (up to {runtime['high']:.2f} h) may exceed the sbatch time limit ({limit:.2f} h)")
    if gpu_info.get("peak_mem_mb") and gpu_info["peak_mem_mb"] > 0.9 * gpu_info["mem_total_mb"]:
        warnings.append("GPU memory above 90% on the subset; full-size batches may run out of memory")
    budget = study["budget"]
    usd = _scale(gpu_hours, rates["usd_per_gpu_hour"])
    if budget.get("gpu_hours") and (gpu_hours["point"] or gpu_hours["high"]) > budget["gpu_hours"]:
        warnings.append(f"exceeds the study's GPU-hour budget ({budget['gpu_hours']})")
    if budget.get("usd") and (usd["point"] or usd["high"]) > budget["usd"]:
        warnings.append(f"exceeds the study's dollar budget (${budget['usd']})")

    return {
        "basis": basis,
        "inputs": {"total_items": total_items, "epochs": epochs, "gpus": gpus,
                   "throughput_items_per_s": tp, "profile_id": profile.get("id")},
        "runtime_h": runtime,
        "gpu_hours": gpu_hours,
        "usd": usd,
        "kwh": kwh,
        "kg_co2": _scale(kwh, rates["kg_co2_per_kwh"]),
        "rates": rates,
        "assumptions": assumptions,
        "warnings": warnings,
        "hints": hints,
        "sbatch": sb,
    }
