"""Runtime backends for executing compiled pipelines."""

from __future__ import annotations

from .executor import Executor, available, get_executor, register

# Importing the pandas executor registers it as the default backend.
from . import pandas_executor  # noqa: F401

__all__ = ["Executor", "available", "get_executor", "register"]
