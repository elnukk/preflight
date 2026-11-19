"""Record a GPU telemetry trace in the mock-adapter fixture format.

    preflight-record --out trace.json --pattern dataloader_bound -- python train.py
    preflight-record --out trace.json --seconds 60        # sample whatever is running
"""
import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict

from .adapters import select_adapter


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="preflight-record")
    p.add_argument("--out", required=True)
    p.add_argument("--seconds", type=float, default=60, help="duration when no command is given")
    p.add_argument("--interval", type=float, default=0.5)
    p.add_argument("--pattern", default="unlabeled", help="label for the waste pattern this trace shows")
    p.add_argument("--adapter", default=None, help="nvml or nvidia-smi (default: auto, never mock)")
    p.add_argument("cmd", nargs=argparse.REMAINDER)
    args = p.parse_args(argv)

    adapter = select_adapter(args.adapter)
    if not adapter.measured:
        print(f"no real GPU telemetry available: {getattr(adapter, 'fallback_reason', '')}", file=sys.stderr)
        return 1

    cmd = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd
    proc = subprocess.Popen(cmd) if cmd else None
    end = time.monotonic() + args.seconds
    samples = []
    while (proc.poll() is None) if proc else time.monotonic() < end:
        samples.append([asdict(s) for s in adapter.sample()])
        time.sleep(args.interval)

    caps = adapter.capabilities()
    trace = {"gpu_name": ", ".join(g["name"] for g in caps.get("gpus", [])), "pattern": args.pattern,
             "adapter": adapter.name, "interval_s": args.interval, "samples": samples}
    with open(args.out, "w") as f:
        json.dump(trace, f, indent=1)
    print(f"wrote {len(samples)} samples to {args.out}", file=sys.stderr)
    return proc.returncode if proc else 0


if __name__ == "__main__":
    sys.exit(main())
