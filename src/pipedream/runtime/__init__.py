"""Runtime backends for executing compiled pipelines."""

from __future__ import annotations

from .executor import Executor, available, get_executor, register

# Importing the executors registers them. pandas is the default backend;
# duckdb registers its class but imports duckdb lazily at run time.
from . import pandas_executor  # noqa: F401
from . import duckdb_executor  # noqa: F401

__all__ = ["Executor", "available", "get_executor", "register"]
