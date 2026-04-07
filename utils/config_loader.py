"""
utils/config_loader.py – Optional YAML override for config.py values.

If a file ``config_override.yaml`` exists in the project root it is loaded
and its key–value pairs are patched into the ``config`` module at import time.

Usage:
    from utils.config_loader import load_overrides
    load_overrides()          # call once at startup
"""

import os
from typing import Any, Dict

import yaml  # type: ignore

import config
from utils.logging_utils import get_logger

logger = get_logger(__name__)

_OVERRIDE_FILE = os.path.join(config.PROJECT_ROOT, "config_override.yaml")


def load_overrides(path: str | None = None) -> Dict[str, Any]:
    """Load YAML overrides and patch ``config`` module attributes.

    Returns the dict of overridden keys for inspection / logging.
    """
    path = path or _OVERRIDE_FILE
    if not os.path.isfile(path):
        logger.debug("No config override file found at %s", path)
        return {}

    with open(path, "r", encoding="utf-8") as fh:
        overrides: Dict[str, Any] = yaml.safe_load(fh) or {}

    applied: Dict[str, Any] = {}
    for key, value in overrides.items():
        upper_key = key.upper()
        if hasattr(config, upper_key):
            setattr(config, upper_key, value)
            applied[upper_key] = value
            logger.info("Config override: %s = %s", upper_key, value)
        else:
            logger.warning("Unknown config key in override file: %s", key)

    return applied
