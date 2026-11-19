"""Small CNN on synthetic images. Deliberately data-loading bound: heavy per-item CPU work, num_workers=0."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

p = argparse.ArgumentParser()
p.add_argument("--epochs", type=int, default=10)
p.add_argument("--batch-size", type=int, default=128)
p.add_argument("--num-samples", type=int, default=200_000)
p.add_argument("--num-workers", type=int, default=0)
p.add_argument("--image-size", type=int, default=64)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()


class SyntheticImages(Dataset):
    def __init__(self, n, size):
        self.n, self.size = n, size

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng(i)
        label = i % 10
        img = rng.normal(label / 10, 1.0, (3, self.size, self.size)).astype(np.float32)
        for _ in range(4):  # stand-in for decode + augmentation
            img = np.clip(img + rng.normal(0, 0.05, img.shape).astype(np.float32), -3, 3)
            img = (img - img.mean()) / (img.std() + 1e-6)
        return torch.from_numpy(img), label


torch.manual_seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
full = SyntheticImages(args.num_samples, args.image_size)
data, epochs = full, args.epochs
if subset := int(os.environ.get("PREFLIGHT_SUBSET", 0)):
    data, epochs = Subset(full, range(min(subset, len(full)))), 1
loader = DataLoader(data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

model = nn.Sequential(
    nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
    nn.Flatten(), nn.Linear(256, 10),
).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
loss_fn = nn.CrossEntropyLoss()

print(f"[preflight] total_items={len(full)} epochs={args.epochs}", flush=True)
step = seen = 0
for epoch in range(epochs):
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        correct += (logits.argmax(1) == y).sum().item()
        total += len(y)
        step += 1
        seen += len(y)
        print(f"[preflight] step={step} items={seen}", flush=True)
    print(f"epoch {epoch} loss={loss.item():.4f} acc={correct / total:.4f}", flush=True)

Path("results").mkdir(exist_ok=True)
Path("results/metrics.json").write_text(json.dumps({"train_acc": correct / total, "loss": loss.item()}))
