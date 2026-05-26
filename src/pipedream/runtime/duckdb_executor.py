"""DuckDB executor: the single execution engine.

Compiles a validated pipeline down to SQL. Each step becomes a temporary view
named by its step id, defined in dependency order; the pipeline output is then
selected from the final view. Row expressions are translated from the safe
expression AST into SQL expressions rather than evaluated.

Deterministic ops run as SQL. The one model-driven op, ``classify``, runs as a
DuckDB Python scalar UDF: a per-step function (bound to that step's labels and
the runtime model) is registered on the connection and called per row. The model
is only resolved if a pipeline actually contains a ``classify`` step.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any

from ..errors import ExecutionError
from ..expr import compile_expr
from ..ir import (
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
from ..passes import execution_order
from ..templating import placeholders, render
from .executor import Executor, register
from .model import RuntimeModel, get_model

# IR aggregation function -> SQL. Functions not listed here are emitted verbatim.
_AGG_SQL = {"mean": "avg", "std": "stddev_samp"}

_JOIN_SQL = {
    "inner": "JOIN",
    "left": "LEFT JOIN",
    "right": "RIGHT JOIN",
    "outer": "FULL OUTER JOIN",
}


def _ident(name: str) -> str:
    """Quote a SQL identifier (step id or column name)."""
    return '"' + name.replace('"', '""') + '"'


def _str_lit(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


class DuckDBExecutor(Executor):
    name = "duckdb"

    def __init__(self, model: RuntimeModel | None = None) -> None:
        self._model = model

    def _runtime_model(self) -> RuntimeModel:
        if self._model is None:
            self._model = get_model()
        return self._model

    def run(self, pipeline: Pipeline) -> Any:
        try:
            import duckdb
        except ImportError as exc:  # pragma: no cover - duckdb is a declared dep
            raise ExecutionError(
                "the 'duckdb' package is required for the duckdb executor"
            ) from exc

        con = duckdb.connect()
        try:
            for step in execution_order(pipeline):
                sql = self._step_sql(step, con)
                try:
                    con.execute(f"CREATE TEMP VIEW {_ident(step.id)} AS {sql}")
                except Exception as exc:  # noqa: BLE001 - add step context
                    raise ExecutionError(
                        f"step {step.id!r} ({step.op}) failed: {exc}"
                    ) from exc
            result = con.execute(
                f"SELECT * FROM {_ident(pipeline.output_id())}"
            ).fetchdf()
        finally:
            con.close()
        return result.reset_index(drop=True)

    def _step_sql(self, step: Step, con: Any) -> str:
        if isinstance(step, LoadCsv):
            header = "true" if step.has_header else "false"
            return (
                f"SELECT * FROM read_csv_auto({_str_lit(step.path)}, header={header})"
            )
        if isinstance(step, LoadInline):
            import pandas as pd

            records = json.loads(step.data_json)
            df = pd.DataFrame.from_records(records)
            reg = f"_inline_{step.id}"
            con.register(reg, df)
            return f"SELECT * FROM {_ident(reg)}"
        if isinstance(step, Select):
            cols = ", ".join(_ident(c) for c in step.columns)
            return f"SELECT {cols} FROM {_ident(step.input)}"
        if isinstance(step, Filter):
            return (
                f"SELECT * FROM {_ident(step.input)} "
                f"WHERE {self._sql_expr(step.predicate)}"
            )
        if isinstance(step, Derive):
            existing = _columns(con, step.input)
            keep = f" EXCLUDE ({_ident(step.column)})" if step.column in existing else ""
            return (
                f"SELECT *{keep}, ({self._sql_expr(step.expr)}) AS {_ident(step.column)} "
                f"FROM {_ident(step.input)}"
            )
        if isinstance(step, Aggregate):
            return self._aggregate_sql(step)
        if isinstance(step, Join):
            on = ", ".join(_ident(c) for c in step.on)
            join = _JOIN_SQL[step.how]
            return (
                f"SELECT * FROM {_ident(step.left)} {join} {_ident(step.right)} "
                f"USING ({on})"
            )
        if isinstance(step, Sort):
            direction = "DESC" if step.descending else "ASC"
            cols = ", ".join(f"{_ident(c)} {direction}" for c in step.by)
            return f"SELECT * FROM {_ident(step.input)} ORDER BY {cols}"
        if isinstance(step, Limit):
            return f"SELECT * FROM {_ident(step.input)} LIMIT {int(step.count)}"
        if isinstance(step, Rename):
            if not step.renames:
                return f"SELECT * FROM {_ident(step.input)}"
            pairs = ", ".join(
                f"{_ident(r.source)} AS {_ident(r.target)}" for r in step.renames
            )
            return f"SELECT * RENAME ({pairs}) FROM {_ident(step.input)}"
        if isinstance(step, Classify):
            return self._classify_sql(step, con)
        raise ExecutionError(f"no duckdb implementation for op {step.op!r}")

    def _classify_sql(self, step: Classify, con: Any) -> str:
        model = self._runtime_model()
        labels = list(step.labels)
        template = step.template
        cols = placeholders(template)

        def udf(*values: Any) -> str:
            row = {c: ("" if v is None else v) for c, v in zip(cols, values)}
            return model.classify(render(template, row), labels)

        fn = "pd_classify_" + re.sub(r"\W", "_", step.id)
        n = max(1, len(cols))
        con.create_function(fn, udf, ["VARCHAR"] * n, "VARCHAR")

        if cols:
            call = ", ".join(f"CAST({_ident(c)} AS VARCHAR)" for c in cols)
        else:
            call = "CAST('' AS VARCHAR)"  # constant template: UDF ignores the arg
        existing = _columns(con, step.input)
        keep = f" EXCLUDE ({_ident(step.column)})" if step.column in existing else ""
        return (
            f"SELECT *{keep}, {fn}({call}) AS {_ident(step.column)} "
            f"FROM {_ident(step.input)}"
        )

    def _aggregate_sql(self, step: Aggregate) -> str:
        projections: list[str] = [_ident(c) for c in step.group_by]
        for agg in step.aggregations:
            if agg.func == "count" and agg.column == "":
                expr = "count(*)"
            elif agg.func == "nunique":
                expr = f"count(DISTINCT {_ident(agg.column)})"
            else:
                fn = _AGG_SQL.get(agg.func, agg.func)
                expr = f"{fn}({_ident(agg.column)})"
            projections.append(f"{expr} AS {_ident(agg.output)}")
        select = ", ".join(projections)
        sql = f"SELECT {select} FROM {_ident(step.input)}"
        if step.group_by:
            sql += " GROUP BY " + ", ".join(_ident(c) for c in step.group_by)
        return sql

    def _sql_expr(self, source: str) -> str:
        # compile_expr validates against the safe grammar; we then translate the
        # validated AST to SQL.
        return _to_sql(compile_expr(source).tree)


def _columns(con: Any, view: str) -> set[str]:
    cursor = con.execute(f"SELECT * FROM {_ident(view)} LIMIT 0")
    return {col[0] for col in cursor.description}


# ---------------------------------------------------------------------------
# Expression AST -> SQL
# ---------------------------------------------------------------------------

_BIN_SQL = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
    ast.Mod: "%",
}

_CMP_SQL = {
    ast.Eq: "=",
    ast.NotEq: "<>",
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
}

_FUNC_SQL = {
    "lower": "lower",
    "upper": "upper",
    "len": "length",
    "abs": "abs",
    "round": "round",
    "startswith": "starts_with",
    "endswith": "ends_with",
    "contains": "contains",
    "coalesce": "coalesce",
}


def _to_sql(node: ast.AST) -> str:
    if isinstance(node, ast.Constant):
        return _const_sql(node.value)
    if isinstance(node, ast.Name):
        if node.id == "True":
            return "TRUE"
        if node.id == "False":
            return "FALSE"
        if node.id == "None":
            return "NULL"
        return _ident(node.id)
    if isinstance(node, ast.BoolOp):
        joiner = " AND " if isinstance(node.op, ast.And) else " OR "
        return "(" + joiner.join(_to_sql(v) for v in node.values) + ")"
    if isinstance(node, ast.UnaryOp):
        operand = _to_sql(node.operand)
        if isinstance(node.op, ast.Not):
            return f"(NOT {operand})"
        if isinstance(node.op, ast.USub):
            return f"(-{operand})"
        if isinstance(node.op, ast.UAdd):
            return f"(+{operand})"
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        left, right = _to_sql(node.left), _to_sql(node.right)
        if op_type is ast.Pow:
            return f"power({left}, {right})"
        if op_type is ast.FloorDiv:
            return f"floor(({left}) / ({right}))"
        sql_op = _BIN_SQL.get(op_type)
        if sql_op is not None:
            return f"({left} {sql_op} {right})"
    if isinstance(node, ast.Compare):
        return _compare_sql(node)
    if isinstance(node, (ast.List, ast.Tuple)):
        return "(" + ", ".join(_to_sql(e) for e in node.elts) + ")"
    if isinstance(node, ast.Call):
        return _call_sql(node)
    raise ExecutionError(f"cannot translate expression node {type(node).__name__} to SQL")


def _const_sql(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, str):
        return _str_lit(value)
    return repr(value)


def _compare_sql(node: ast.Compare) -> str:
    parts: list[str] = []
    left = _to_sql(node.left)
    for op_node, comparator in zip(node.ops, node.comparators):
        right = _to_sql(comparator)
        if isinstance(op_node, ast.In):
            parts.append(f"({left} IN {right})")
        elif isinstance(op_node, ast.NotIn):
            parts.append(f"({left} NOT IN {right})")
        else:
            sql_op = _CMP_SQL.get(type(op_node))
            if sql_op is None:
                raise ExecutionError("unsupported comparison operator in SQL lowering")
            parts.append(f"({left} {sql_op} {right})")
        left = right
    return "(" + " AND ".join(parts) + ")"


def _call_sql(node: ast.Call) -> str:
    name = node.func.id  # type: ignore[union-attr]
    args = [_to_sql(a) for a in node.args]
    if name == "is_null":
        return f"({args[0]} IS NULL)"
    sql_name = _FUNC_SQL.get(name)
    if sql_name is None:
        raise ExecutionError(f"cannot translate function {name!r} to SQL")
    return f"{sql_name}({', '.join(args)})"


register("duckdb", DuckDBExecutor)
