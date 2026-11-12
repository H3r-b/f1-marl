
from pathlib import Path

import pytest

from f1_marl.config import load_yaml_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_load_environment_config() -> None:
    config = load_yaml_config(PROJECT_ROOT / "configs" / "environment.yaml")

    assert config["environment"]["num_agents"] == 4
    assert config["vehicle"]["model"] == "kinematic_bicycle"


def test_reject_non_mapping_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("- item\n- item\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Expected a YAML mapping"):
        load_yaml_config(config_path)
