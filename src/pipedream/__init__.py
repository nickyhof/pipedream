"""PipeDream: an LLM-based compiler and runtime for data pipelines.

Author a pipeline in plain English; the compiler frontend (Claude) turns it into
a typed, executor-agnostic IR, deterministic passes validate and optimize it,
and a pluggable runtime executes it (in-memory pandas by default).
"""

from __future__ import annotations

from .compiler import Compiler, load_pipeline, parse_pipeline, save_pipeline
from .errors import CompileError, ExecutionError, ExpressionError, PipeDreamError
from .ir import Pipeline
from .runtime import available, get_executor

__version__ = "0.1.0"

__all__ = [
    "Compiler",
    "Pipeline",
    "load_pipeline",
    "parse_pipeline",
    "save_pipeline",
    "get_executor",
    "available",
    "CompileError",
    "ExecutionError",
    "ExpressionError",
    "PipeDreamError",
    "__version__",
]
