import json
import os
import subprocess
import sys
import time

from tests.conftest import REPO

HOOK = REPO / "hooks" / "pre_launch.py"


def run_hook(cmd, cwd):
    payload = {"tool_name": "Bash", "tool_input": {"command": cmd}, "cwd": str(cwd)}
    out = subprocess.run([sys.executable, str(HOOK)], input=json.dumps(payload), capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)["hookSpecificOutput"] if out.stdout.strip() else None


def test_ignores_non_launch(toy):
    assert run_hook("ls -la", toy) is None
    assert run_hook("PREFLIGHT_SKIP=1 sbatch train.sbatch", toy) is None


def test_asks_without_estimate(toy):
    out = run_hook("sbatch train.sbatch", toy)
    assert out["permissionDecision"] == "ask"
    assert "train.sbatch" in out["permissionDecisionReason"]
    assert run_hook("python train.py --epochs 5", toy)["permissionDecision"] == "ask"


def test_block_mode(toy):
    (toy / ".preflight").mkdir()
    (toy / ".preflight" / "study.yaml").write_text("name: t\nhook_mode: block\n")
    assert run_hook("sbatch train.sbatch", toy)["permissionDecision"] == "deny"


def write_estimate(toy, target, created_at):
    d = toy / ".preflight" / "estimates"
    d.mkdir(parents=True, exist_ok=True)
    (d / "e.json").write_text(json.dumps({"keys": [str(target)], "created_at": created_at}))


def test_allows_with_fresh_estimate(toy):
    write_estimate(toy, toy / "train.sbatch", time.time() + 1)
    assert run_hook("sbatch train.sbatch", toy) is None


def test_stale_estimate_after_edit(toy):
    write_estimate(toy, toy / "train.sbatch", time.time() - 100)
    os.utime(toy / "train.sbatch")
    assert run_hook("sbatch train.sbatch", toy)["permissionDecision"] == "ask"
