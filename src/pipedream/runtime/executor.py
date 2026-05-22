"""The pluggable executor interface and registry.

An :class:`Executor` knows how to run a validated :class:`~pipedream.ir.Pipeline`
against some backend. The default is the in-memory pandas executor, but the
registry lets other backends (DuckDB, a SQL warehouse, Spark) be added without
touching the compiler or the IR — they only need to implement ``run``.

Executors are responsible for nothing but execution: by the time ``run`` is
called the pipeline has already passed semantic analysis, so an executor may
assume references resolve and the graph is acyclic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

from ..ir import Pipeline


class Executor(ABC):
    """Base class for pipeline backends."""

    name: str = "executor"

    @abstractmethod
    def run(self, pipeline: Pipeline) -> Any:
        """Execute ``pipeline`` and return its output table."""


_REGISTRY: dict[str, Callable[[], Executor]] = {}


def register(name: str, factory: Callable[[], Executor]) -> None:
    _REGISTRY[name] = factory


def available() -> list[str]:
    return sorted(_REGISTRY)


def get_executor(name: str = "pandas") -> Executor:
    try:
        factory = _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown executor {name!r}; available: {', '.join(available()) or 'none'}"
        ) from None
    return factory()
