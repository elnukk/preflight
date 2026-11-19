# Slurm requests (cpus_overrequested, memory_overrequested, time_overrequested, oversized_gpu)

Over-requesting doesn't change results, but it makes your jobs wait longer in the queue and blocks other people's.

- **CPUs.** Request about the number of DataLoader workers + 1. If CPU efficiency is low *and* the GPU is starved, the fix is more workers, not fewer CPUs.
- **Memory.** About 1.5× the observed MaxRSS peak.
- **Time.** The estimate +20%. Shorter limits qualify for backfill scheduling.
- **GPU type.** Jobs using under 25% of GPU memory at moderate utilization often fit a smaller GPU or a MIG slice (`--gres=gpu:1g.10gb:1` or whatever the cluster calls it).

Check the cluster's partition and GRES names (`sinfo -o "%P %G"`) before suggesting a change.
