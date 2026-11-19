"""Stand-in training job (stdlib only). Deliberately data-loading bound: each batch waits on simulated I/O."""
import argparse
import json
import math
import os
import random
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--epochs", type=int, default=5)
p.add_argument("--batch-size", type=int, default=64)
p.add_argument("--num-samples", type=int, default=1_200_000)
p.add_argument("--io-ms", type=float, default=25.0, help="simulated per-batch loading latency")
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

random.seed(args.seed)
n, epochs = args.num_samples, args.epochs
if subset := os.environ.get("PREFLIGHT_SUBSET"):
    n, epochs = min(n, int(subset)), 1

print(f"[preflight] total_items={args.num_samples} epochs={args.epochs}")

w, loss, seen, step = 0.0, 0.0, 0, 0
for epoch in range(epochs):
    for start in range(0, n, args.batch_size):
        bs = min(args.batch_size, n - start)
        time.sleep(args.io_ms / 1000)
        xs = [random.gauss(0, 1) for _ in range(bs)]
        grad = sum(2 * (w * x - 3 * x) * x for x in xs) / bs
        w -= 0.05 * grad
        loss = sum((w * x - 3 * x) ** 2 for x in xs) / bs
        seen += bs
        step += 1
        print(f"[preflight] step={step} items={seen}")
    print(f"epoch {epoch} loss={loss:.6f} w={w:.4f}")

out = Path("results")
out.mkdir(exist_ok=True)
(out / "metrics.json").write_text(json.dumps({"loss": loss, "w": w, "rmse": math.sqrt(loss)}))
