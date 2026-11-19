import shutil
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TOY = REPO / "benchmarks" / "toy_train"


@pytest.fixture
def toy(tmp_path) -> Path:
    dest = tmp_path / "toy"
    shutil.copytree(TOY, dest, ignore=shutil.ignore_patterns("results", ".preflight"))
    return dest.resolve()
