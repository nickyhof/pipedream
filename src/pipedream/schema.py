"""Schema analysis: name resolution and type inference over the IR.

This is the compiler's type checker. It propagates a column schema (names +
coarse dtypes) through the pipeline DAG and rejects, at compile time, pipelines
that reference columns that don't exist, join on missing/colliding keys, or
aggregate the wrong type — errors that would otherwise only surface at run time.

Source columns come from a :class:`SchemaProvider`. The default reads CSV headers
(and sniffs dtypes) from disk and infers ``load_inline`` schemas from the JSON
itself. When a source can't be resolved (e.g. the file isn't present at compile
time) its schema is *unknown* and checks downstream of it are relaxed, so
``compile``/``explain`` stay usable without the data on hand.

Type checking is deliberately conservative: dtypes are inferred and propagated
(so they can be shown and used), but the only hard type rule is that numeric
aggregations (``sum``/``mean``/``std``/``median``) require a numeric column.
Expression-internal type mismatches are inferred-but-not-rejected to avoid
false positives that would block otherwise-valid pipelines.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from .errors import CompileError
from .expr import column_names, compile_expr
from .ir import (
    Aggregate,
    Classify,
    Derive,
    Filter,
    Join,
    Limit,
    LoadCsv,
    LoadInline,
    Pipeline,
    Rename,
    Select,
    Sort,
    Step,
)
from .templating import placeholders

# Coarse dtypes. "unknown" means "could not determine"; checks relax on it.
DType = str
INT, FLOAT, BOOL, STR, NULL, UNKNOWN = "int", "float", "bool", "str", "null", "unknown"
_NUMERIC = {INT, FLOAT}


def is_numeric(dtype: DType) -> bool:
    return dtype in _NUMERIC


@dataclass(frozen=True)
class Column:
    name: str
    dtype: DType = UNKNOWN


@dataclass(frozen=True)
class Schema:
    """An ordered set of typed columns, or ``unknown`` when undeterminable."""

    columns: tuple[Column, ...] | None

    @property
    def known(self) -> bool:
        return self.columns is not None

    def names(self) -> list[str]:
        return [c.name for c in self.columns] if self.columns else []

    def get(self, name: str) -> DType | None:
        if not self.columns:
            return None
        for col in self.columns:
            if col.name == name:
                return col.dtype
        return None

    def has(self, name: str) -> bool:
        return self.get(name) is not None

    @classmethod
    def unknown(cls) -> "Schema":
        return cls(None)

    @classmethod
    def of(cls, pairs: list[tuple[str, DType]]) -> "Schema":
        return cls(tuple(Column(n, t) for n, t in pairs))


class SchemaProvider(Protocol):
    """Resolves the schema of a *source* step (one that introduces data)."""

    def resolve(self, step: Step) -> Schema: ...


class FileSchemaProvider:
    """Default provider: reads CSV headers/dtypes from the local filesystem."""

    def resolve(self, step: Step) -> Schema:
        if isinstance(step, LoadCsv):
            return self._csv_schema(step)
        return Schema.unknown()

    @staticmethod
    def _csv_schema(step: LoadCsv) -> Schema:
        if not step.has_header:
            return Schema.unknown()
        try:
            import pyarrow.csv as pacsv

            arrow_schema = pacsv.open_csv(step.path).schema
        except Exception:  # noqa: BLE001 - missing/unreadable file -> relax
            return Schema.unknown()
        return Schema.of([(f.name, _arrow_dtype(f.type)) for f in arrow_schema])


def _arrow_dtype(dtype: Any) -> DType:
    import pyarrow as pa

    if pa.types.is_integer(dtype):
        return INT
    if pa.types.is_floating(dtype):
        return FLOAT
    if pa.types.is_boolean(dtype):
        return BOOL
    return STR


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def infer_schemas(
    pipeline: Pipeline, provider: SchemaProvider | None = None
) -> dict[str, Schema]:
    """Compute the output schema of every (reachable) step, raising on errors."""
    from .passes import execution_order  # local import avoids an import cycle

    provider = provider or FileSchemaProvider()
    schemas: dict[str, Schema] = {}
    for step in execution_order(pipeline):
        schemas[step.id] = _infer_step(step, schemas, provider)
    return schemas


def check_schema(pipeline: Pipeline, provider: SchemaProvider | None = None) -> None:
    """Run inference for its side effect of raising :class:`CompileError`."""
    infer_schemas(pipeline, provider)


def _infer_step(
    step: Step, schemas: dict[str, Schema], provider: SchemaProvider
) -> Schema:
    if isinstance(step, LoadInline):
        return _inline_schema(step)
    if isinstance(step, LoadCsv):
        return provider.resolve(step)
    if isinstance(step, Join):
        return _join_schema(step, schemas[step.left], schemas[step.right])
    if isinstance(step, Select):
        return _select_schema(step, schemas[step.input])
    if isinstance(step, Filter):
        return _filter_schema(step, schemas[step.input])
    if isinstance(step, Derive):
        return _derive_schema(step, schemas[step.input])
    if isinstance(step, Aggregate):
        return _aggregate_schema(step, schemas[step.input])
    if isinstance(step, Sort):
        _require_columns(step, schemas[step.input], step.by)
        return schemas[step.input]
    if isinstance(step, Limit):
        return schemas[step.input]
    if isinstance(step, Rename):
        return _rename_schema(step, schemas[step.input])
    if isinstance(step, Classify):
        return _classify_schema(step, schemas[step.input])
    return Schema.unknown()


def _inline_schema(step: LoadInline) -> Schema:
    try:
        records = json.loads(step.data_json)
    except Exception as exc:  # noqa: BLE001
        raise CompileError(f"step {step.id!r}: invalid inline JSON: {exc}") from exc
    if not isinstance(records, list):
        raise CompileError(f"step {step.id!r}: inline data must be a JSON array")
    order: list[str] = []
    values: dict[str, list[Any]] = {}
    for rec in records:
        if not isinstance(rec, dict):
            raise CompileError(f"step {step.id!r}: inline rows must be JSON objects")
        for key, val in rec.items():
            if key not in values:
                values[key] = []
                order.append(key)
            values[key].append(val)
    return Schema.of([(name, _infer_value_dtype(values[name])) for name in order])


def _infer_value_dtype(values: list[Any]) -> DType:
    kinds: set[str] = set()
    for v in values:
        if v is None:
            continue
        if isinstance(v, bool):
            kinds.add(BOOL)
        elif isinstance(v, int):
            kinds.add(INT)
        elif isinstance(v, float):
            kinds.add(FLOAT)
        elif isinstance(v, str):
            kinds.add(STR)
        else:
            kinds.add(UNKNOWN)
    if not kinds:
        return NULL
    if kinds == {INT}:
        return INT
    if kinds <= {INT, FLOAT}:
        return FLOAT
    if len(kinds) == 1:
        return next(iter(kinds))
    return UNKNOWN


def _require_columns(step: Step, schema: Schema, cols: list[str]) -> None:
    if not schema.known:
        return
    for col in cols:
        if not schema.has(col):
            raise CompileError(
                f"step {step.id!r} ({step.op}) references unknown column {col!r}; "
                f"available: {', '.join(schema.names()) or '(none)'}"
            )


def _select_schema(step: Select, src: Schema) -> Schema:
    _require_columns(step, src, step.columns)
    if not src.known:
        return Schema.of([(c, UNKNOWN) for c in step.columns])
    return Schema.of([(c, src.get(c) or UNKNOWN) for c in step.columns])


def _check_expr_columns(step: Step, src: Schema, source: str) -> None:
    if not src.known:
        return
    refs = column_names(compile_expr(source).tree)
    missing = [r for r in sorted(refs) if not src.has(r)]
    if missing:
        raise CompileError(
            f"step {step.id!r} ({step.op}) uses unknown column(s) "
            f"{', '.join(missing)}; available: {', '.join(src.names()) or '(none)'}"
        )


def _filter_schema(step: Filter, src: Schema) -> Schema:
    _check_expr_columns(step, src, step.predicate)
    return src


def _derive_schema(step: Derive, src: Schema) -> Schema:
    _check_expr_columns(step, src, step.expr)
    if not src.known:
        return Schema.unknown()
    dtype = _infer_expr_type(compile_expr(step.expr).tree, {c.name: c.dtype for c in src.columns})
    cols = [c for c in src.columns if c.name != step.column]
    cols.append(Column(step.column, dtype))
    # Preserve original position when overwriting an existing column.
    if src.has(step.column):
        ordered = [
            Column(step.column, dtype) if c.name == step.column else c
            for c in src.columns
        ]
        return Schema(tuple(ordered))
    return Schema(tuple(cols))


def _aggregate_schema(step: Aggregate, src: Schema) -> Schema:
    _require_columns(step, src, step.group_by)
    out: list[Column] = []
    for col in step.group_by:
        out.append(Column(col, src.get(col) or UNKNOWN if src.known else UNKNOWN))
    for agg in step.aggregations:
        if agg.func == "count" and agg.column == "":
            out.append(Column(agg.output, INT))
            continue
        if src.known and not src.has(agg.column):
            raise CompileError(
                f"step {step.id!r} (aggregate) references unknown column "
                f"{agg.column!r}; available: {', '.join(src.names()) or '(none)'}"
            )
        col_dtype = src.get(agg.column) if src.known else UNKNOWN
        if agg.func in ("sum", "mean", "std", "median") and col_dtype not in (None, UNKNOWN):
            if not is_numeric(col_dtype):
                raise CompileError(
                    f"step {step.id!r} (aggregate): {agg.func}() requires a numeric "
                    f"column but {agg.column!r} is {col_dtype}"
                )
        out.append(Column(agg.output, _agg_dtype(agg.func, col_dtype or UNKNOWN)))
    _check_duplicate_names(step, [c.name for c in out])
    return Schema(tuple(out))


def _agg_dtype(func: str, col_dtype: DType) -> DType:
    if func in ("count", "nunique"):
        return INT
    if func in ("mean", "std", "median"):
        return FLOAT
    if func == "sum":
        return col_dtype if is_numeric(col_dtype) else UNKNOWN
    return col_dtype  # min/max/first/last preserve the column dtype


def _join_schema(step: Join, left: Schema, right: Schema) -> Schema:
    if left.known:
        _require_columns(step, left, step.on)
    if right.known:
        _require_columns(step, right, step.on)
    if not (left.known and right.known):
        return Schema.unknown()
    on = set(step.on)
    left_extra = [c for c in left.columns if c.name not in on]
    right_extra = [c for c in right.columns if c.name not in on]
    collision = {c.name for c in left_extra} & {c.name for c in right_extra}
    if collision:
        raise CompileError(
            f"step {step.id!r} (join) produces ambiguous column(s) "
            f"{', '.join(sorted(collision))}; rename one side before joining"
        )
    keys = [Column(name, left.get(name) or UNKNOWN) for name in step.on]
    return Schema(tuple(keys + left_extra + right_extra))


def _rename_schema(step: Rename, src: Schema) -> Schema:
    _require_columns(step, src, [r.source for r in step.renames])
    mapping = {r.source: r.target for r in step.renames}
    if not src.known:
        return Schema.unknown()
    renamed = [Column(mapping.get(c.name, c.name), c.dtype) for c in src.columns]
    _check_duplicate_names(step, [c.name for c in renamed])
    return Schema(tuple(renamed))


def _classify_schema(step: Classify, src: Schema) -> Schema:
    _require_columns(step, src, placeholders(step.template))
    if not step.labels:
        raise CompileError(f"step {step.id!r} (classify) must define at least one label")
    if not src.known:
        return Schema.unknown()
    cols = [c for c in src.columns if c.name != step.column]
    cols.append(Column(step.column, STR))
    if src.has(step.column):
        return Schema(
            tuple(
                Column(step.column, STR) if c.name == step.column else c
                for c in src.columns
            )
        )
    return Schema(tuple(cols))


def _check_duplicate_names(step: Step, names: list[str]) -> None:
    seen: set[str] = set()
    for name in names:
        if name in seen:
            raise CompileError(
                f"step {step.id!r} ({step.op}) produces duplicate column {name!r}"
            )
        seen.add(name)


# ---------------------------------------------------------------------------
# Expression type inference (lenient: returns UNKNOWN when unsure, never raises)
# ---------------------------------------------------------------------------


def _infer_expr_type(node: Any, env: dict[str, DType]) -> DType:
    import ast

    if isinstance(node, ast.Constant):
        v = node.value
        if isinstance(v, bool):
            return BOOL
        if isinstance(v, int):
            return INT
        if isinstance(v, float):
            return FLOAT
        if isinstance(v, str):
            return STR
        return NULL
    if isinstance(node, ast.Name):
        return env.get(node.id, UNKNOWN)
    if isinstance(node, ast.BoolOp):
        return BOOL
    if isinstance(node, ast.Compare):
        return BOOL
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.Not):
            return BOOL
        return _infer_expr_type(node.operand, env)
    if isinstance(node, ast.BinOp):
        left = _infer_expr_type(node.left, env)
        right = _infer_expr_type(node.right, env)
        if isinstance(node.op, ast.Div):
            return FLOAT
        if isinstance(node.op, ast.Add) and (left == STR or right == STR):
            return STR
        if is_numeric(left) and is_numeric(right):
            return FLOAT if FLOAT in (left, right) else INT
        return UNKNOWN
    if isinstance(node, ast.Call):
        return _CALL_TYPES.get(node.func.id, UNKNOWN)  # type: ignore[union-attr]
    return UNKNOWN


_CALL_TYPES = {
    "lower": STR,
    "upper": STR,
    "len": INT,
    "round": FLOAT,
    "abs": FLOAT,
    "startswith": BOOL,
    "endswith": BOOL,
    "contains": BOOL,
    "is_null": BOOL,
    "coalesce": UNKNOWN,
}
