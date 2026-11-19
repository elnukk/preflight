import json
import time
import uuid
from pathlib import Path


def find_root(start: Path) -> Path:
    """Nearest ancestor holding a .preflight/ directory, else `start` itself."""
    start = start.resolve()
    for d in [start, *start.parents]:
        if (d / ".preflight").is_dir():
            return d
    return start


def save(root: Path, kind: str, record: dict) -> dict:
    record.setdefault("id", f"{kind[:-1]}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}")
    record.setdefault("created_at", time.time())
    d = root / ".preflight" / kind
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{record['id']}.json").write_text(json.dumps(record, indent=2))
    return record


def load(root: Path, kind: str, record_id: str) -> dict:
    path = root / ".preflight" / kind / f"{record_id}.json"
    if not path.is_file():
        raise FileNotFoundError(f"no {kind[:-1]} {record_id!r} under {root}")
    return json.loads(path.read_text())
