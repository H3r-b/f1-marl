"""Configuration loading utilities."""

from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file and return its top-level mapping.

    Args:
        path: Path to a YAML file.

    Raises:
        ValueError: If the YAML document is not a mapping.
        FileNotFoundError: If the path does not exist.
    """
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as config_file:
        data = yaml.safe_load(config_file)

    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}")

    return data
