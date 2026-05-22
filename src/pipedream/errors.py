"""Error types raised across the compiler and runtime phases."""

from __future__ import annotations


class PipeDreamError(Exception):
    """Base class for every error PipeDream raises."""


class CompileError(PipeDreamError):
    """Raised when natural-language source cannot be turned into a valid IR.

    Covers LLM/transport failures, schema-validation failures of the model's
    output, and semantic-analysis failures (unknown references, cycles).
    """


class ExecutionError(PipeDreamError):
    """Raised when a validated pipeline fails at runtime."""


class ExpressionError(PipeDreamError):
    """Raised when a row expression is malformed or uses a disallowed construct."""
