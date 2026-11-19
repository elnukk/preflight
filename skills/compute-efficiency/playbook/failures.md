# Failures and lost work (profile_failed, early_failures, timeouts, time_limit_too_short, no_checkpoint, over_budget)

A failed job costs its own GPU-hours and the time spent in the queue, and coding agents often resubmit it again and again.

- **Failure in profiling.** Fix it before submitting; it reproduces in seconds. The usual causes are paths that differ on compute nodes, modules or environments not activated in the sbatch script, and package mismatches.
- **Early failures in history.** Make `profile_run` a required step before `sbatch`; the launch hook enforces this.
- **Time limit too short.** Set `--time` to the high estimate +20%. If the job is longer than the partition allows, checkpoint and resubmit with `--dependency=afterany:<jobid>`.
- **No checkpointing on long jobs.** Save `{model, optimizer, scheduler, epoch, rng_state}` every N minutes and resume on start when a checkpoint exists. With RNG state restored, resuming does not change results.
- **Over budget.** Stop and confirm the configuration with the researcher before running anything.
