# Instrumentation (no_progress_markers, no_subset_flag)

Printing-only changes do not alter results.

```python
import os
SUBSET = int(os.environ.get("PREFLIGHT_SUBSET", 0)) or None
if SUBSET:
    dataset = torch.utils.data.Subset(dataset, range(min(SUBSET, len(dataset))))
print(f"[preflight] total_items={full_len} epochs={args.epochs}", flush=True)
...
seen += batch_size
print(f"[preflight] step={step} items={seen}", flush=True)
```
`total_items` is the size of the **full** dataset even when running a subset. `items` is cumulative.
