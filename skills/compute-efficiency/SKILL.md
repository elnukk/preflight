---
name: compute-efficiency
user-invocable: false
description: Use before launching training, fine-tuning, feature-extraction, embedding or evaluation jobs on GPUs or Slurm, or when a researcher asks what a job will cost or why it is slow. Runs the Preflight loop of measuring, diagnosing, proposing, verifying and reporting with the preflight MCP tools.
---

# Compute efficiency with Preflight

Preflight sits between the researcher, their coding agent and the cluster. Its job is to make compute cost visible and
waste actionable **without changing research results**.

## The loop

1. **Measure.** `capabilities` shows what telemetry is real. `profile_run` runs a short profile on a data subset.
   `estimate_job` turns the profile into full-run ranges.
2. **Diagnose.** Call `diagnose` (profile, estimate, script, and `include_history` on Slurm clusters) for
   rule-based findings, then read the code to confirm or reject each one. `slurm_history` shows past requested vs
   used resources. Each finding links to a page in `playbook/`; read it before proposing a fix. Typical signals:
   - mean GPU utilization under 40% with busy CPUs: data-loading bound (`num_workers`, `pin_memory`, caching);
   - peak GPU memory far below capacity: room for a larger batch (this can change results);
   - long `time_to_first_step_s`: expensive setup such as re-extracting features or loading the full dataset;
   - non-zero exit or stderr in the profile: the job will fail at scale, and fixing it costs nothing.
3. **Propose.** Show changes as diffs, each with its evidence and whether it can change results. Never apply them
   unless the researcher asks.
4. **Verify.** A change that can affect results must be compared against the tolerances in `.preflight/study.yaml`
   before you recommend it. If you cannot verify a change, say so and do not recommend it.
5. **Report.** Give numbers as ranges with their basis. Never invent numbers the tools did not return.

## Playbook

- [dataloading.md](playbook/dataloading.md): low GPU utilization, CPU- or I/O-bound input pipelines
- [memory.md](playbook/memory.md): GPU memory headroom or near-OOM
- [caching.md](playbook/caching.md): slow startup, recomputed features and embeddings
- [failures.md](playbook/failures.md): failed profiles, early failures, timeouts, checkpointing, budget
- [slurm-requests.md](playbook/slurm-requests.md): over-requested CPUs, memory, time or GPU size
- [precision.md](playbook/precision.md): mixed precision (may change results)
- [instrumentation.md](playbook/instrumentation.md): progress markers and subset support

## Rules

- Always use absolute paths with the tools.
- When `gpu.measured` is false, say the GPU numbers are simulated.
- The launch hook asks about or blocks jobs that have no up-to-date estimate. Re-estimate after editing a script.
  Use `PREFLIGHT_SKIP=1` only when the researcher says to.
