# Mixed precision (no_mixed_precision)

**This may change results.** bf16 autocast usually speeds up compute-bound training 1.5–2× on Ampere and newer GPUs, but it changes numerics.

```python
with torch.autocast("cuda", dtype=torch.bfloat16):
    loss = model(x).loss
```
Prefer bf16 over fp16, since it needs no GradScaler. Do not apply it to evaluation code whose outputs are compared bit-for-bit.

**Verify.** Train baseline and bf16 on the same subset with the same seed and compare the study's metrics within tolerance. If you can't run that comparison, say so and do not recommend the change.
