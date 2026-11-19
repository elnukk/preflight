# GPU memory (memory_headroom, memory_near_limit)

**Headroom (peak memory under 50% of the GPU).** A larger batch can raise throughput. **This may change results**: optimization dynamics change, so the learning rate usually needs scaling. Recommend it only after a baseline-vs-change run on the same subset matches within the study tolerances. If the researcher wants identical optimization, keep the effective batch the same and use the memory for something else, such as fewer gradient-accumulation steps.

**Near the limit (peak above 90% on the subset).** The full run may run out of memory once it hits longer sequences, larger images or evaluation batches. Options:
- reduce the per-step batch and use gradient accumulation (equivalent up to numerics);
- activation checkpointing (`model.gradient_checkpointing_enable()`), which is slower but uses less memory;
- `torch.cuda.empty_cache()` does not fix this.

**Verify.** Profile with the largest input the dataset contains, not a random subset.
