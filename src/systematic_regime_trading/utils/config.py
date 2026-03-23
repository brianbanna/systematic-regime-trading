"""
Config loader that reads YAML files from configs/ directory.
All parameters accessed via config object -- no hard-coded values in source.
"""

import yaml
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


def load_config(name: str) -> dict:
    """
    Load a YAML config file by name (without extension).

    Args:
        name: Config file name (e.g., 'models', 'data', 'features')

    Returns:
        Dictionary with config values
    """
    path = PROJECT_ROOT / "configs" / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def get_path(relative: str) -> Path:
    """
    Resolve a path relative to project root.

    Args:
        relative: Relative path string (e.g., 'data/processed')

    Returns:
        Absolute Path object
    """
    return PROJECT_ROOT / relative


def get_project_root() -> Path:
    """Return the project root directory."""
    return PROJECT_ROOT
