"""A small, safe row-expression evaluator.

Filter predicates and derived-column expressions are authored as strings (by the
LLM frontend or a human writing IR directly). Rather than ``eval`` them — which
would let a compiled pipeline run arbitrary code — we parse them with Python's
``ast`` module and walk a strict allowlist of node types. Anything outside the
grammar (attribute access, calls to unknown functions, comprehensions, lambdas,
indexing, etc.) raises :class:`ExpressionError`.

Expressions evaluate against a single row represented as a ``dict`` of
column name to value. The grammar supports:

* column references (bare names) and the literals ``True``/``False``/``None``
* number and string literals
* boolean ops ``and``/``or``, unary ``not``/``-``/``+``
* arithmetic ``+ - * / // % **``
* comparisons ``== != < <= > >=`` and membership ``in`` / ``not in``
* a fixed set of helper functions (see ``_FUNCTIONS``)

The same grammar is engine-agnostic in spirit: the pandas executor evaluates it
row-wise here, while a future SQL executor would translate the AST to a WHERE/
SELECT expression. Keeping it tiny is what makes that translation tractable.
"""

from __future__ import annotations

import ast
from typing import Any, Callable

from .errors import ExpressionError


def _fn_contains(haystack: Any, needle: Any) -> bool:
    if haystack is None:
        return False
    return needle in haystack


def _fn_coalesce(*values: Any) -> Any:
    for v in values:
        if v is not None and not _is_nan(v):
            return v
    return None


def _is_nan(v: Any) -> bool:
    return isinstance(v, float) and v != v


def _fn_is_null(v: Any) -> bool:
    return v is None or _is_nan(v)


_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "lower": lambda s: s.lower() if isinstance(s, str) else s,
    "upper": lambda s: s.upper() if isinstance(s, str) else s,
    "len": len,
    "abs": abs,
    "round": round,
    "startswith": lambda s, p: isinstance(s, str) and s.startswith(p),
    "endswith": lambda s, p: isinstance(s, str) and s.endswith(p),
    "contains": _fn_contains,
    "is_null": _fn_is_null,
    "coalesce": _fn_coalesce,
}

_BIN_OPS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a**b,
}

_CMP_OPS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Eq: lambda a, b: a == b,
    ast.NotEq: lambda a, b: a != b,
    ast.Lt: lambda a, b: a < b,
    ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b,
    ast.GtE: lambda a, b: a >= b,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}

_CONSTANTS: dict[str, Any] = {"True": True, "False": False, "None": None}


class Expression:
    """A compiled, reusable row expression."""

    __slots__ = ("source", "_tree")

    def __init__(self, source: str) -> None:
        self.source = source
        try:
            self._tree = ast.parse(source, mode="eval").body
        except SyntaxError as exc:
            raise ExpressionError(f"invalid expression {source!r}: {exc.msg}") from exc
        _validate(self._tree, source)

    def evaluate(self, row: dict[str, Any]) -> Any:
        return _eval(self._tree, row)

    @property
    def tree(self) -> ast.AST:
        """The validated AST root.

        Exposed so alternative backends can lower an expression to their own
        target language (e.g. SQL) instead of evaluating it row-wise.
        """
        return self._tree

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Expression({self.source!r})"


def compile_expr(source: str) -> Expression:
    return Expression(source)


def column_names(tree: ast.AST) -> set[str]:
    """Collect the column references in an expression AST.

    Descends into call arguments but not the called function's name, so
    ``lower(status)`` yields ``{"status"}`` (not ``"lower"``). Used by schema
    analysis to check that referenced columns exist upstream.
    """
    out: set[str] = set()

    def rec(node: ast.AST) -> None:
        if isinstance(node, ast.Call):
            for arg in node.args:
                rec(arg)
            return
        if isinstance(node, ast.Name):
            if node.id not in _CONSTANTS:
                out.add(node.id)
            return
        for child in ast.iter_child_nodes(node):
            rec(child)

    rec(tree)
    return out


def _validate(node: ast.AST, source: str) -> None:
    """Reject any node type outside the allowlist before evaluation."""
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            if not isinstance(child.func, ast.Name) or child.func.id not in _FUNCTIONS:
                name = getattr(child.func, "id", "<expr>")
                raise ExpressionError(
                    f"call to unknown or disallowed function {name!r} in {source!r}"
                )
            if child.keywords:
                raise ExpressionError(
                    f"keyword arguments are not allowed in {source!r}"
                )
        elif isinstance(child, _DISALLOWED):
            raise ExpressionError(
                f"unsupported syntax {type(child).__name__} in expression {source!r}"
            )


# Node types that must never appear. Listing them explicitly (rather than only
# allowlisting) gives clearer error messages and a second line of defence.
_DISALLOWED = (
    ast.Attribute,
    ast.Subscript,
    ast.Lambda,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
    ast.Starred,
    ast.Await,
    ast.Yield,
    ast.NamedExpr,
)


def _eval(node: ast.AST, row: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        if node.id in row:
            return row[node.id]
        raise ExpressionError(f"unknown column {node.id!r}")
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            return all(_eval(v, row) for v in node.values)
        return any(_eval(v, row) for v in node.values)
    if isinstance(node, ast.UnaryOp):
        operand = _eval(node.operand, row)
        if isinstance(node.op, ast.Not):
            return not operand
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return +operand
    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is not None:
            return op(_eval(node.left, row), _eval(node.right, row))
    if isinstance(node, ast.Compare):
        left = _eval(node.left, row)
        for op_node, right_node in zip(node.ops, node.comparators):
            op = _CMP_OPS.get(type(op_node))
            if op is None:
                raise ExpressionError(f"unsupported comparison in {ast.dump(node)}")
            right = _eval(right_node, row)
            if not op(left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.List):
        return [_eval(e, row) for e in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_eval(e, row) for e in node.elts)
    if isinstance(node, ast.Call):
        fn = _FUNCTIONS[node.func.id]  # type: ignore[union-attr]
        return fn(*(_eval(a, row) for a in node.args))
    raise ExpressionError(f"unsupported expression node {type(node).__name__}")
