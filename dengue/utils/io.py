"""IO helpers: YAML loading, path resolution, and dir creation."""

from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def resolve_paths(cfg: Dict[str, Any]) -> Dict[str, Path]:
    """
    Returns a dict of canonical project paths and ensures key directories exist.

    Keys returned:
      root, data, raw, external, interim, processed,
      artifacts, models, forecasts, figures, tables
    """
    root = Path(cfg.get("project_root", ".")).resolve()
    data_root = root / "data"
    art_root = root / "artifacts"

    paths: Dict[str, Path] = {
        "root": root,
        "data": data_root,
        "raw": data_root / "raw",
        "external": data_root / "external",
        "interim": data_root / "interim",
        "processed": data_root / "processed",
        "artifacts": art_root,
        "models": art_root / "models",
        "forecasts": art_root / "forecasts",
        "figures": art_root / "figures",
        "tables": art_root / "tables",
    }

    # Ensure these exist on disk
    for k in (
        "raw",
        "external",
        "interim",
        "processed",
        "artifacts",
        "models",
        "forecasts",
        "figures",
        "tables",
    ):
        ensure_dir(paths[k])

    return paths
