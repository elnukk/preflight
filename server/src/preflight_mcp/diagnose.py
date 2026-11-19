"""Rule-based waste and failure detectors. They report evidence; Claude reads the code and proposes the diff."""
import ast
import re
from dataclasses import asdict, dataclass, field

from .slurm import parse_mem_mb

FAILURE_RISK, SAFE, MAY_CHANGE = "failure_risk", "safe", "may_change_results"
RISK_ORDER = {FAILURE_RISK: 0, SAFE: 1, MAY_CHANGE: 2}

LOW_UTIL_PCT = 40
COMPUTE_BOUND_UTIL_PCT = 60
CPU_PEGGED_PCT = 85
MEM_HEADROOM_FRAC = 0.5
MEM_NEAR_LIMIT_FRAC = 0.9
MEM_OVERREQUEST_FRAC = 0.25
SLOW_STARTUP_S = 30
LONG_JOB_H = 2

CHECKPOINT = re.compile(r"torch\.save|save_pretrained|save_checkpoint|ModelCheckpoint|save_state|\.save\(")
MIXED_PRECISION = re.compile(r"autocast|GradScaler|bf16|fp16|bfloat16|float16|mixed_precision")
SUBSET = re.compile(r"PREFLIGHT_SUBSET|--(subset|debug|limit|max[-_]samples|n[-_]samples|num[-_]samples|smoke)")
FEATURES = re.compile(r"\b(embed\w*|encode|extract_features|feature_extractor|get_features|hidden_states)\b")
CACHE = re.compile(r"np\.save|np\.load|torch\.load|to_parquet|read_parquet|pickle|joblib|\.exists\(|lru_cache|"
                   r"cache_dir|load_from_disk|save_to_disk|h5py|memmap")
USES_GPU = re.compile(r"\bcuda\b|\.to\(device\)|device_map|accelerator")
GPU_FRAMEWORK = re.compile(r"^\s*(import|from)\s+(torch|jax|flax|tensorflow|keras|cupy|cudf|cuml|vllm|transformers|"
                           r"accelerate|deepspeed|lightning|pytorch_lightning|sentence_transformers|numba|triton|"
                           r"xgboost|lightgbm|onnxruntime)\b|\bcuda\b", re.M)
STDERR_PATTERNS = [
    (re.compile(r"CUDA out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED|MemoryError|Killed"), "out of memory"),
    (re.compile(r"FileNotFoundError|No such file or directory"), "missing file or path"),
    (re.compile(r"ModuleNotFoundError|ImportError"), "missing Python package"),
    (re.compile(r"CUDA error|NCCL error|device-side assert"), "CUDA/NCCL error"),
    (re.compile(r"KeyError|IndexError|TypeError|ValueError|AttributeError"), "bug in the script"),
]


@dataclass
class Finding:
    id: str
    title: str
    risk: str
    confidence: str
    evidence: list[str]
    suggestion: str
    playbook: str
    est_savings: dict = field(default_factory=dict)


def argparse_defaults(tree: ast.AST) -> dict:
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "add_argument":
            flag = next((a.value for a in node.args if isinstance(a, ast.Constant) and str(a.value).startswith("--")), None)
            default = next((k.value for k in node.keywords if k.arg == "default"), None)
            if flag and isinstance(default, ast.Constant):
                out[flag.lstrip("-").replace("-", "_")] = default.value
    return out


def dataloader_calls(source: str) -> list[dict]:
    """DataLoader(...) calls with their num_workers / pin_memory settings.

    None = not set, "expr" = set from something we cannot resolve; `args.x` resolves to the argparse default.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    defaults = argparse_defaults(tree)

    def value(n):
        if n is None:
            return None
        if isinstance(n, ast.Constant):
            return n.value
        if isinstance(n, ast.Attribute) and n.attr in defaults:
            return defaults[n.attr]
        return "expr"

    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fname = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if fname == "DataLoader":
                kw = {k.arg: k.value for k in node.keywords if k.arg}
                calls.append({"line": node.lineno, "num_workers": value(kw.get("num_workers")),
                              "pin_memory": value(kw.get("pin_memory"))})
    return calls


def uses_gpu(source: str | None) -> bool | None:
    """None when there is no source to read."""
    return None if source is None else bool(GPU_FRAMEWORK.search(source))


def _gpu_evidence(gpu: dict) -> tuple[str, str]:
    """Confidence ceiling and a caveat for GPU-based findings."""
    if not gpu.get("measured", True):
        return "low", f"GPU numbers are simulated ({gpu.get('adapter')} adapter)"
    return "high", ""


def _with_caveat(evidence: list[str], caveat: str) -> list[str]:
    return evidence + ([caveat] if caveat else [])


def from_profile(profile: dict, source: str | None) -> list[Finding]:
    out = []
    gpu, cpu = profile.get("gpu", {}), profile.get("cpu", {})
    ceiling, caveat = _gpu_evidence(gpu)

    if profile.get("exit_code") not in (None, 0) and not profile.get("timed_out"):
        stderr = "\n".join(profile.get("stderr_tail", []))
        cause = next((label for rx, label in STDERR_PATTERNS if rx.search(stderr)), "unknown error")
        last = next((l for l in reversed(profile.get("stderr_tail", [])) if l.strip()), "")
        out.append(Finding("profile_failed", f"Profile run failed: {cause}", FAILURE_RISK, "high",
                           [f"exit code {profile['exit_code']} after {profile.get('wall_s')} s", f"stderr: {last[:300]}"],
                           "Fix the error before launching; the full job would fail the same way after queueing.",
                           "playbook/failures.md"))

    if profile.get("steps_seen", 0) == 0:
        out.append(Finding("no_progress_markers", "Throughput could not be measured", SAFE, "high",
                           ["no `[preflight] step=… items=…` lines in stdout"],
                           "Print `[preflight] total_items=N epochs=E` once and `[preflight] step=S items=I` per batch "
                           "(cumulative items). Printing only, so results are unaffected.",
                           "playbook/instrumentation.md"))

    util, cpu_pct = gpu.get("mean_util_pct"), cpu.get("mean_pct")
    if uses_gpu(source) is False:
        if cpu_pct is not None and cpu_pct < 50:
            out.append(Finding("io_bound", "Job mostly waiting: CPU idle", SAFE, "medium",
                               [f"job CPU {cpu_pct}% (100% = one core)",
                                "script uses no GPU framework, so GPU telemetry was ignored"],
                               "Find what each step waits on (disk, network, sleeps, remote calls) and overlap or "
                               "cache it.", "playbook/dataloading.md"))
        return out + _startup(profile, source)

    loaders = dataloader_calls(source) if source else []
    if util is not None and util < LOW_UTIL_PCT:
        unset_workers = [c for c in loaders if c["num_workers"] in (None, 0)]
        evidence = [f"mean GPU utilization {util}% (median {gpu.get('median_util_pct')}%)"]
        if cpu_pct is not None:
            evidence.append(f"job CPU {cpu_pct}% (100% = one core)")
        if unset_workers:
            evidence += [f"DataLoader at line {c['line']} has num_workers={c['num_workers'] or 0}" for c in unset_workers]
        if unset_workers or (cpu_pct or 0) >= CPU_PEGGED_PCT:
            out.append(Finding("dataloader_bound", "GPU waits on CPU-side data loading", SAFE,
                               "medium" if ceiling == "low" else "high", _with_caveat(evidence, caveat),
                               "Set DataLoader num_workers to the allocated CPUs minus one, pin_memory=True, "
                               "persistent_workers=True; move per-item preprocessing out of the training loop.",
                               "playbook/dataloading.md"))
        elif cpu_pct is not None and cpu_pct < 50:
            out.append(Finding("io_bound", "GPU and CPU both mostly idle: waiting on I/O or network", SAFE,
                               "medium" if ceiling == "high" else "low", _with_caveat(evidence, caveat),
                               "Stage data to local/node scratch, cache decoded data, prefetch, or batch remote calls.",
                               "playbook/dataloading.md"))
        else:
            out.append(Finding("gpu_underutilized", "GPU underutilized, cause unclear", SAFE, "low",
                               _with_caveat(evidence, caveat),
                               "Profile one step with torch.profiler to see where time goes.",
                               "playbook/dataloading.md"))

    peak, total = gpu.get("peak_mem_mb"), gpu.get("mem_total_mb")
    if peak and total:
        frac = peak / total
        if frac > MEM_NEAR_LIMIT_FRAC:
            out.append(Finding("memory_near_limit", "GPU memory nearly full on the subset", FAILURE_RISK, ceiling,
                               _with_caveat([f"peak {peak:.0f} of {total:.0f} MB ({frac:.0%})"], caveat),
                               "Longer sequences or larger batches at full scale may run out of memory. Reduce batch "
                               "size with gradient accumulation, or enable activation checkpointing.",
                               "playbook/memory.md"))
        elif frac < MEM_HEADROOM_FRAC and (util is None or util < 90):
            out.append(Finding("memory_headroom", "Room for a larger batch", MAY_CHANGE, ceiling,
                               _with_caveat([f"peak {peak:.0f} of {total:.0f} MB ({frac:.0%})"], caveat),
                               "A larger batch can raise throughput, but changes optimization dynamics (learning-rate "
                               "scaling). Only recommend after verifying metrics within the study's tolerances.",
                               "playbook/memory.md"))
        if total and peak < 0.25 * total and gpu.get("n_gpus", 1) == 1:
            out.append(Finding("oversized_gpu", "Job may fit on a smaller GPU or a MIG slice", SAFE, ceiling,
                               _with_caveat([f"uses {peak:.0f} MB of a {total:.0f} MB GPU"], caveat),
                               "Request a smaller GPU type or MIG partition if the cluster offers one; usually shorter "
                               "queues and the same results.", "playbook/slurm-requests.md"))

    return out + _startup(profile, source)


def _startup(profile: dict, source: str | None) -> list[Finding]:
    out = []
    ttfs, wall = profile.get("time_to_first_step_s"), profile.get("wall_s") or 0
    if ttfs is not None and ttfs > max(SLOW_STARTUP_S, 0.25 * wall):
        evidence = [f"{ttfs:.0f} s before the first training step ({ttfs / max(wall, 1e-9):.0%} of the profile)"]
        if source and FEATURES.search(source) and not CACHE.search(source):
            evidence.append("script computes features/embeddings with no sign of caching them to disk")
        out.append(Finding("slow_startup", "Expensive setup before training starts", SAFE, "medium", evidence,
                           "Cache extracted features or preprocessed data to disk and reuse them across runs "
                           "(key the cache on model + data version).", "playbook/caching.md"))
    return out


def from_source(source: str, estimate: dict | None) -> list[Finding]:
    out = []
    long_job = estimate and ((estimate.get("runtime_h") or {}).get("high") or 0) > LONG_JOB_H
    if long_job and not CHECKPOINT.search(source):
        out.append(Finding("no_checkpoint", "Long job without checkpointing", FAILURE_RISK, "medium",
                           [f"estimated up to {estimate['runtime_h']['high']} h", "no checkpoint-saving call found"],
                           "Save model/optimizer state periodically and resume from the last checkpoint, so a "
                           "timeout or preemption does not discard the run.", "playbook/failures.md"))
    if not SUBSET.search(source):
        out.append(Finding("no_subset_flag", "No way to run on a small subset", SAFE, "medium",
                           ["no subset/debug/limit argument and PREFLIGHT_SUBSET is not read"],
                           "Read PREFLIGHT_SUBSET (or add --subset) so debugging and profiling do not process the "
                           "full dataset.", "playbook/instrumentation.md"))
    if FEATURES.search(source) and not CACHE.search(source):
        out.append(Finding("uncached_features", "Features/embeddings recomputed every run", SAFE, "low",
                           ["feature/embedding computation found, no disk caching found"],
                           "Compute features once, save them (np.save / parquet) and load them on later runs.",
                           "playbook/caching.md"))
    return out


def from_estimate(estimate: dict) -> list[Finding]:
    out = []
    runtime, limit = estimate.get("runtime_h") or {}, (estimate.get("sbatch") or {}).get("time_limit_h")
    if limit and runtime.get("high") and runtime["high"] > limit:
        out.append(Finding("time_limit_too_short", "Job will likely hit its time limit", FAILURE_RISK, "high",
                           [f"estimated {runtime.get('low')}–{runtime['high']} h vs --time {limit:.2f} h"],
                           f"Raise --time to about {runtime['high'] * 1.2:.2f} h (estimate +20%), or checkpoint and "
                           "resume across jobs.", "playbook/failures.md",
                           {"gpu_hours_at_risk": round(limit * estimate["inputs"]["gpus"], 3)}))
    for w in estimate.get("warnings", []):
        if "budget" in w:
            out.append(Finding("over_budget", "Estimate exceeds the study budget", FAILURE_RISK, "medium", [w],
                               "Check the configuration with the researcher before launching.", "playbook/failures.md"))
    return out


def from_history(summary: dict) -> list[Finding]:
    out = []
    n = summary.get("n_jobs", 0)
    if n == 0:
        return out
    if summary["n_early_failures"]:
        out.append(Finding("early_failures", "Jobs failing in their first minutes", FAILURE_RISK, "high",
                           [f"{summary['n_early_failures']} of {n} jobs failed within 10 minutes",
                            f"{summary['gpu_hours_failed']} GPU-hours spent in failed jobs"],
                           "Run a short profile (profile_run) before each submission; most early failures are "
                           "path, package or config errors visible in seconds.", "playbook/failures.md",
                           {"gpu_hours_failed": summary["gpu_hours_failed"]}))
    if summary["n_timeouts"]:
        out.append(Finding("timeouts", "Jobs hitting their time limit", FAILURE_RISK, "high",
                           [f"{summary['n_timeouts']} of {n} jobs ended in TIMEOUT"],
                           "Estimate runtime before launching and checkpoint long jobs.", "playbook/failures.md"))
    if n >= 3 and (eff := summary.get("median_cpu_eff")) is not None and eff < 0.25:
        out.append(Finding("cpus_overrequested", "CPUs requested but mostly idle", SAFE, "medium",
                           [f"median CPU efficiency {eff:.0%} across {n} jobs"],
                           "Request fewer CPUs (--cpus-per-task), unless DataLoader workers should be using them.",
                           "playbook/slurm-requests.md"))
    if n >= 3 and (eff := summary.get("median_mem_eff")) is not None and eff < 0.25:
        out.append(Finding("memory_overrequested", "Memory requested but unused", SAFE, "medium",
                           [f"median memory efficiency {eff:.0%} across {n} jobs"],
                           "Lower --mem to about 1.5× the observed peak; smaller requests schedule sooner.",
                           "playbook/slurm-requests.md"))
    if n >= 3 and (frac := summary.get("median_time_used_frac")) is not None and frac < 0.25:
        out.append(Finding("time_overrequested", "Time limits far above actual runtime", SAFE, "medium",
                           [f"median job used {frac:.0%} of its time limit"],
                           "Set --time closer to the estimate (+20%); shorter limits fit backfill and start sooner.",
                           "playbook/slurm-requests.md"))
    if (util := summary.get("median_gpu_util_pct")) is not None and util < LOW_UTIL_PCT:
        out.append(Finding("low_gpu_util_history", "Past GPU jobs mostly idle", SAFE, "high",
                           [f"median GPU utilization {util}% across GPU jobs (Slurm accounting)"],
                           "Profile the most expensive job and check data loading.", "playbook/dataloading.md"))
    return out


def from_requests(sbatch: dict, profile: dict | None, source: str | None, estimate: dict | None) -> list[Finding]:
    """Compare what the sbatch file requests with what one profiled run used."""
    out = []
    gpus = sbatch.get("gpus")
    if gpus and uses_gpu(source) is False:
        gpu_hours = ((estimate or {}).get("gpu_hours") or {}).get("point")
        out.append(Finding("gpu_requested_unused", "GPU requested but the script never uses one", SAFE, "medium",
                           [f"sbatch requests {gpus} GPU(s)",
                            "the script imports no GPU framework and never mentions CUDA"],
                           "Drop --gres/--gpus and run on a CPU partition, unless the sbatch script also runs other "
                           "GPU programs.", "playbook/slurm-requests.md",
                           {"gpu_hours": gpu_hours} if gpu_hours else {}))
    req = parse_mem_mb(sbatch["mem"]) if sbatch.get("mem") else None
    peak = ((profile or {}).get("cpu") or {}).get("peak_rss_mb")
    if req and peak and peak < MEM_OVERREQUEST_FRAC * req:
        out.append(Finding("memory_overrequested", "Memory requested but unused", SAFE, "low",
                           [f"--mem={sbatch['mem']} ({req:.0f} MB) vs {peak:.0f} MB peak in the profile",
                            "measured on a subset; the full run may need more"],
                           "Lower --mem to about 1.5× the full-run peak; confirm with MaxRSS from sacct after the first "
                           "full run.", "playbook/slurm-requests.md"))
    return out


def mixed_precision_finding(profile: dict, source: str) -> Finding | None:
    gpu = profile.get("gpu", {})
    util = gpu.get("mean_util_pct")
    if not (util and util >= COMPUTE_BOUND_UTIL_PCT and USES_GPU.search(source) and not MIXED_PRECISION.search(source)):
        return None
    ceiling, caveat = _gpu_evidence(gpu)
    return Finding("no_mixed_precision", "Compute-bound in full precision", MAY_CHANGE, ceiling,
                   _with_caveat([f"mean GPU utilization {util}%", "no autocast/bf16 found"], caveat),
                   "bf16 autocast often gives 1.5–2× on recent GPUs but changes numerics; verify metrics against "
                   "the study's tolerances first.", "playbook/precision.md")


def diagnose(profile: dict | None = None, estimate: dict | None = None, source: str | None = None,
             history_summary: dict | None = None, sbatch: dict | None = None) -> dict:
    findings: list[Finding] = []
    if profile:
        findings += from_profile(profile, source)
        if source and (f := mixed_precision_finding(profile, source)):
            findings.append(f)
    if source:
        findings += from_source(source, estimate)
    if estimate:
        findings += from_estimate(estimate)
    if history_summary:
        findings += from_history(history_summary)
    if sbatch := sbatch or (estimate or {}).get("sbatch"):
        findings += from_requests(sbatch, profile, source, estimate)

    seen, unique = set(), []
    for f in findings:
        if f.id not in seen:
            seen.add(f.id)
            unique.append(f)
    if any(f.id == "slow_startup" for f in unique):
        unique = [f for f in unique if f.id != "uncached_features"]
    unique.sort(key=lambda f: RISK_ORDER[f.risk])
    return {"findings": [asdict(f) for f in unique], "counts": {r: sum(f.risk == r for f in unique) for r in RISK_ORDER}}
