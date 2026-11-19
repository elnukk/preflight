import os


def visible_gpus(uuids: list[str]) -> list[int]:
    """Device indices this process may use, honoring CUDA_VISIBLE_DEVICES (indices or UUIDs)."""
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None or raw.strip() == "":
        return list(range(len(uuids)))
    out = []
    for item in (s.strip() for s in raw.split(",")):
        if item.isdigit() and int(item) < len(uuids):
            out.append(int(item))
        elif item.startswith(("GPU-", "MIG-")):
            out.extend(i for i, u in enumerate(uuids) if u.startswith(item))
    return out or list(range(len(uuids)))
