"""
utils/logging_utils.py – Centralised logging for every module.

Usage:
    from utils.logging_utils import get_logger
    logger = get_logger(__name__)
    logger.info("message")
"""

import logging
import os
import sys
from datetime import datetime

import config

_LOG_INITIALISED = False


def _init_root_logger() -> None:
    """Configure the root logger once."""
    global _LOG_INITIALISED
    if _LOG_INITIALISED:
        return

    root = logging.getLogger()
    root.setLevel(getattr(logging, config.LOG_LEVEL.upper(), logging.INFO))

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)-30s │ %(message)s",
        datefmt="%H:%M:%S",
    )
    console.setFormatter(fmt)
    root.addHandler(console)

    # Optional file handler
    if config.LOG_TO_FILE:
        os.makedirs(config.LOG_DIR, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(
            os.path.join(config.LOG_DIR, f"mtp_{stamp}.log"),
            encoding="utf-8",
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        root.addHandler(fh)

    _LOG_INITIALISED = True


def get_logger(name: str) -> logging.Logger:
    """Return a named child logger; initialises root on first call."""
    _init_root_logger()
    return logging.getLogger(name)
