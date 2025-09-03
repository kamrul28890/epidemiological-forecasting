"""IO helpers stub."""

"""
IO & config helpers.
"""

from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(p: str | Path) -> Path:
    pth = Path(p)
    pth.mkdir(parents=True, exist_ok=True)
    return pth


def resolve_paths(cfg: Dict[str, Any]) -> Dict[str, Path]:
    paths = cfg.get("paths", {})
    out = {k: Path(v) for k, v in paths.items()}
    # ensure writeable dirs exist
    for k in ["external", "interim", "processed", "artifacts"]:
        if k in out:
            ensure_dir(out[k])
    return out
