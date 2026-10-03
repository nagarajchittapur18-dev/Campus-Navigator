"""
Campus Navigator: Centralized Logging System (Phase 18)
======================================================
Module: utils/logger.py

Provides robust, thread-safe, and UTF-8 encoded logging for the Campus Navigator application.
Automatically creates and manages the `logs/` directory and writes persistent runtime
diagnostic logs to `logs/campus_navigator.log`.

Features:
- FileHandler with UTF-8 encoding (avoids Windows cp1252 unicode crash).
- Standard academic logging format: Timestamp, Log Level, Module, Line, Message.
- Traceback capture for unhandled or caught exceptions.
- Convenience helper functions for standardized error and warning reporting.
"""

import logging
import os
import sys
import traceback
from typing import Optional

# Ensure project root is determined accurately
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE_PATH = os.path.join(LOGS_DIR, "campus_navigator.log")

# Guarantee logs directory exists
os.makedirs(LOGS_DIR, exist_ok=True)

# Common Formatter
LOG_FORMAT = "%(asctime)s [%(levelname)s] [%(name)s:%(filename)s:%(lineno)d] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def setup_logger(
    name: str = "CampusNavigator",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    enable_console: bool = False,
) -> logging.Logger:
    """
    Configure and return a standardized logger instance with persistent file output.

    Args:
        name: Name of the logger module.
        log_file: Target log file path. Defaults to logs/campus_navigator.log.
        level: Minimum logging level (default: logging.INFO).
        enable_console: Whether to also stream log messages to sys.stderr (default: False).

    Returns:
        logger: Configured logging.Logger instance.
    """
    global _configured
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers if setup_logger is called repeatedly
    if not logger.handlers:
        target_path = log_file or LOG_FILE_PATH
        os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)

        formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)

        # File Handler (UTF-8 encoded, append mode)
        try:
            file_handler = logging.FileHandler(target_path, mode="a", encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            print(f"[!] Warning: Could not initialize log file at {target_path}: {e}")

        # Optional Console Handler
        if enable_console:
            console_handler = logging.StreamHandler(sys.stderr)
            console_handler.setLevel(logging.WARNING)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

    return logger


# Global application logger instance
logger = setup_logger("CampusNavigator")


def get_logger(name: str = "CampusNavigator") -> logging.Logger:
    """Retrieve a named child logger under the CampusNavigator hierarchy."""
    if name == "CampusNavigator":
        return logger
    return setup_logger(name)


def log_exception(exc: Exception, context: str = "") -> None:
    """
    Log an exception with its full stack trace and descriptive context.

    Args:
        exc: Exception instance.
        context: Explanatory context string describing where the failure occurred.
    """
    tb = traceback.format_exc()
    msg = f"Exception in {context}: {type(exc).__name__}: {str(exc)}\n{tb}" if context else f"Exception: {type(exc).__name__}: {str(exc)}\n{tb}"
    logger.error(msg)


def log_error(message: str, exc_info: bool = False) -> None:
    """Log an error message to the persistent log file."""
    logger.error(message, exc_info=exc_info)


def log_warning(message: str) -> None:
    """Log a warning message to the persistent log file."""
    logger.warning(message)


def log_info(message: str) -> None:
    """Log an informational message to the persistent log file."""
    logger.info(message)
