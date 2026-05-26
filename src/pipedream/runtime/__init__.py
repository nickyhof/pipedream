"""Runtime backends for executing compiled pipelines."""

from __future__ import annotations

from .executor import Executor, available, get_executor, register
from .model import RuntimeModel, get_model
from .model import available as available_models
from .model import register as register_model

# Importing the executors registers them. pandas is the default backend;
# duckdb registers its class but imports duckdb lazily at run time.
from . import pandas_executor  # noqa: F401
from . import duckdb_executor  # noqa: F401

__all__ = [
    "Executor",
    "available",
    "get_executor",
    "register",
    "RuntimeModel",
    "get_model",
    "available_models",
    "register_model",
]
