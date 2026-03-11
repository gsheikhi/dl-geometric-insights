import json
from pathlib import Path

def load_json(path: str | Path) -> dict[str, list[float]]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def _exists(*paths) -> bool:
    return all(Path(p).exists() for p in paths)