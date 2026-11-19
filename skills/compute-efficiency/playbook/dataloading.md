# Data loading (dataloader_bound, io_bound, gpu_underutilized, low_gpu_util_history)

**Signal.** Mean GPU utilization under 40%.
- CPU near or above 100% (one core pegged): preprocessing runs in the main process.
- CPU low as well: the job is waiting on disk, a network filesystem or remote calls.

**Check.** Look at the `DataLoader(...)` arguments, the per-item work in `__getitem__` (decoding, tokenizing, augmentation), and where the data lives (NFS or home directory vs node-local scratch).

**Fixes. These do not change results when the seeds and data order are kept.**
```python
DataLoader(ds, batch_size=bs, shuffle=True,
           num_workers=max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", 4)) - 1),
           pin_memory=True, persistent_workers=True, prefetch_factor=4)
```
- Tokenize or decode once ahead of time and save the result (see `caching.md`).
- Copy the dataset to `$TMPDIR` or `/scratch` at job start.
- For API or remote calls, batch them or send them concurrently.

**Verify.** Re-profile and confirm utilization went up. Workers can change the random augmentation stream unless `worker_init_fn` seeds each worker, so compare metrics if augmentation is random.
