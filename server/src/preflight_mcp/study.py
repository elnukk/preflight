import copy
from pathlib import Path

import yaml

DEFAULTS = {
    "name": None,
    "outputs": [],
    "budget": {"gpu_hours": None, "usd": None},
    "rates": {
        "usd_per_gpu_hour": 2.0,
        "gpu_tdp_watts": 400,
        "pue": 1.3,
        "kg_co2_per_kwh": 0.2,
    },
    "hook_mode": "warn",
}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def study_path(root: Path) -> Path:
    return root / ".preflight" / "study.yaml"


def load(root: Path) -> dict:
    path = study_path(root)
    data = yaml.safe_load(path.read_text()) if path.is_file() else {}
    study = _merge(DEFAULTS, data or {})
    study["_found"] = path.is_file()
    return study


HEADER = """\
# Preflight study spec. outputs: files that must not change when an optimization is applied, with tolerances
# per metric ({abs: x} or {rel: x}). rates: replace the defaults with your cluster's numbers.
"""


def validate(data: dict) -> list[str]:
    errors = []
    for i, out in enumerate(data.get("outputs") or []):
        if not isinstance(out, dict) or not isinstance(out.get("path"), str):
            errors.append(f"outputs[{i}] needs a 'path'")
            continue
        for metric, tol in (out.get("metrics") or {}).items():
            if not (isinstance(tol, dict) and set(tol) <= {"abs", "rel"} and tol
                    and all(isinstance(v, (int, float)) for v in tol.values())):
                errors.append(f"outputs[{i}].metrics.{metric} must be {{abs: number}} or {{rel: number}}")
    for k, v in (data.get("budget") or {}).items():
        if k not in DEFAULTS["budget"] or not (v is None or isinstance(v, (int, float))):
            errors.append(f"budget.{k} is not a known numeric field")
    for k, v in (data.get("rates") or {}).items():
        if k not in DEFAULTS["rates"] or not isinstance(v, (int, float)):
            errors.append(f"rates.{k} is not a known numeric field")
    if data.get("hook_mode") not in (None, "warn", "block"):
        errors.append("hook_mode must be warn or block")
    return errors


def update(root: Path, **fields) -> Path:
    """Create the study file if needed and merge the given top-level fields into it."""
    path = study_path(root)
    current = yaml.safe_load(path.read_text()) if path.is_file() else {}
    merged = _merge(DEFAULTS, current or {})
    merged = _merge(merged, {k: v for k, v in fields.items() if v is not None})
    if errors := validate(merged):
        raise ValueError("; ".join(errors))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(HEADER + yaml.safe_dump(merged, sort_keys=False))
    return path
