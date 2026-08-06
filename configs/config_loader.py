"""Project configuration loader."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def _expand_environment_variables(value: Any) -> Any:
    """Recursively replace ${VAR_NAME} with environment-variable values."""
    if isinstance(value, dict):
        return {
            key: _expand_environment_variables(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [_expand_environment_variables(item) for item in value]

    if isinstance(value, str):
        return os.path.expandvars(value)

    return value


def load_config(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    env_path: str | Path = DEFAULT_ENV_PATH,
) -> dict[str, Any]:
    """Load .env and YAML configuration."""
    config_path = Path(config_path)
    env_path = Path(env_path)

    if not env_path.is_file():
        raise FileNotFoundError(f".env file not found: {env_path}")

    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    load_dotenv(dotenv_path=env_path, override=False)

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("The root of config.yaml must be a mapping.")

    config = _expand_environment_variables(config)
    config["_project_root"] = str(PROJECT_ROOT)

    return config
