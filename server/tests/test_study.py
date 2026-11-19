import pytest
import yaml

from preflight_mcp import study


def test_update_creates_and_merges(tmp_path):
    study.update(tmp_path, name="toy", budget={"gpu_hours": 10})
    study.update(tmp_path, outputs=[{"path": "results/metrics.json", "metrics": {"loss": {"abs": 1e-6}}}],
                 rates={"usd_per_gpu_hour": 1.5})
    s = study.load(tmp_path)
    assert s["name"] == "toy" and s["budget"] == {"gpu_hours": 10, "usd": None}
    assert s["rates"]["usd_per_gpu_hour"] == 1.5 and s["rates"]["pue"] == 1.3
    assert s["outputs"][0]["metrics"]["loss"] == {"abs": 1e-6}
    assert yaml.safe_load(study.study_path(tmp_path).read_text())["name"] == "toy"


@pytest.mark.parametrize("fields,msg", [
    ({"outputs": [{"metrics": {}}]}, "needs a 'path'"),
    ({"outputs": [{"path": "m.json", "metrics": {"acc": 0.01}}]}, "abs: number"),
    ({"budget": {"hours": 3}}, "budget.hours"),
    ({"rates": {"usd_per_gpu_hour": "cheap"}}, "rates.usd_per_gpu_hour"),
    ({"hook_mode": "loud"}, "hook_mode"),
])
def test_update_rejects_bad_fields(tmp_path, fields, msg):
    with pytest.raises(ValueError, match=msg):
        study.update(tmp_path, name="x", **fields)
    assert not study.study_path(tmp_path).exists()
