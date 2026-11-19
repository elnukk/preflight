import json
import stat

import pytest

from preflight_mcp import record
from preflight_mcp.adapters import MockAdapter, select_adapter
from preflight_mcp.adapters.nvidia_smi import NvidiaSmiAdapter, parse_query
from preflight_mcp.adapters.visible import visible_gpus

SMI_OUT = ("0, GPU-aaa, NVIDIA A100-SXM4-40GB, 35, 11240, 40960, 210.5\n"
           "1, GPU-bbb, NVIDIA A100-SXM4-40GB, 0, 4, 40960, [N/A]\n")


def test_visible_gpus(monkeypatch):
    uuids = ["GPU-aaa", "GPU-bbb", "GPU-ccc"]
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert visible_gpus(uuids) == [0, 1, 2]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,0")
    assert visible_gpus(uuids) == [2, 0]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-bbb")
    assert visible_gpus(uuids) == [1]


def test_parse_query():
    rows = parse_query(SMI_OUT)
    assert rows[0]["util_pct"] == 35 and rows[0]["power_w"] == 210.5 and rows[1]["power_w"] is None


@pytest.fixture
def fake_smi(tmp_path):
    path = tmp_path / "nvidia-smi"
    path.write_text(f"#!/bin/sh\ncat <<'X'\n{SMI_OUT}X\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


def test_nvidia_smi_adapter(fake_smi, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    a = NvidiaSmiAdapter(binary=fake_smi)
    assert a.measured and [g["index"] for g in a.capabilities()["gpus"]] == [1]
    [s] = a.sample()
    assert (s.gpu, s.util_pct, s.mem_used_mb, s.power_w) == (1, 0.0, 4.0, None)


def test_auto_falls_back_to_mock(monkeypatch):
    import preflight_mcp.adapters as ad

    def broken():
        raise RuntimeError("no driver")

    monkeypatch.setitem(ad.FACTORIES, "nvml", broken)
    monkeypatch.setitem(ad.FACTORIES, "nvidia-smi", broken)
    a = select_adapter("auto")
    assert isinstance(a, MockAdapter) and "no driver" in a.capabilities()["fallback_reason"]
    with pytest.raises(ValueError):
        select_adapter("bogus")


def test_record_refuses_mock(tmp_path, monkeypatch):
    monkeypatch.setenv("PREFLIGHT_ADAPTER", "mock")
    assert record.main(["--out", str(tmp_path / "t.json"), "--seconds", "0"]) == 1


def test_record_writes_fixture_format(tmp_path, fake_smi, monkeypatch):
    import preflight_mcp.record as rec

    monkeypatch.setattr(rec, "select_adapter", lambda name: NvidiaSmiAdapter(binary=fake_smi))
    out = tmp_path / "t.json"
    assert rec.main(["--out", str(out), "--interval", "0.05", "--pattern", "test", "--", "sleep", "0.3"]) == 0
    trace = json.loads(out.read_text())
    assert trace["pattern"] == "test" and len(trace["samples"]) >= 2
    MockAdapter(out).sample()
