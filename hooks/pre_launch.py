#!/usr/bin/env python3
"""PreToolUse hook: ask (warn mode) or deny (block mode) job launches that have no fresh Preflight estimate."""
import json
import re
import sys
from pathlib import Path

LAUNCH = re.compile(
    r"\b(sbatch|torchrun|deepspeed|accelerate\s+launch)\b"
    r"|\bpython3?\s+(-\S+\s+)*\S*(train|finetune|fit|extract|embed|eval)\w*\.py"
)
TARGET = re.compile(r"[\w./~-]+\.(?:py|sh|sbatch|slurm)\b")
MODE = re.compile(r"^hook_mode:\s*(\w+)", re.M)


def find_root(start: Path) -> Path | None:
    for d in [start, *start.parents]:
        if (d / ".preflight").is_dir():
            return d
    return None


def has_fresh_estimate(target: Path, root: Path) -> bool:
    est_dir = root / ".preflight" / "estimates"
    if not est_dir.is_dir():
        return False
    mtime = target.stat().st_mtime
    for f in est_dir.glob("*.json"):
        try:
            rec = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if str(target) in rec.get("keys", []) and rec.get("created_at", 0) >= mtime:
            return True
    return False


def hook_mode(root: Path | None) -> str:
    if root:
        study = root / ".preflight" / "study.yaml"
        if study.is_file() and (m := MODE.search(study.read_text())):
            return m.group(1)
    return "warn"


def decide(data: dict) -> dict | None:
    if data.get("tool_name") != "Bash":
        return None
    cmd = data.get("tool_input", {}).get("command", "")
    if "PREFLIGHT_SKIP=1" in cmd or "PREFLIGHT_SUBSET" in cmd or not LAUNCH.search(cmd):
        return None

    cwd = Path(data.get("cwd") or ".")
    targets = [p.resolve() for t in TARGET.findall(cmd) if (p := (cwd / Path(t).expanduser())).is_file()]
    root = next((r for t in targets if (r := find_root(t.parent))), None) or find_root(cwd.resolve())
    if root and any(has_fresh_estimate(t, root) for t in targets):
        return None

    mode = hook_mode(root)
    names = ", ".join(t.name for t in targets) or "this job"
    reason = (
        f"Preflight: no up-to-date estimate for {names}. Run /preflight:preflight (profile_run + estimate_job) "
        "before launching, or prefix the command with PREFLIGHT_SKIP=1 to launch anyway."
    )
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny" if mode == "block" else "ask",
            "permissionDecisionReason": reason,
        }
    }


def main() -> None:
    try:
        data = json.load(sys.stdin)
    except json.JSONDecodeError:
        return
    out = decide(data)
    if out:
        print(json.dumps(out))


if __name__ == "__main__":
    main()
