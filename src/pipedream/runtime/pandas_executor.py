"""In-memory executor backed by pandas — the default runtime backend.

It walks the pipeline's dependency graph, materializing each step's output as a
``DataFrame`` and memoizing it so shared upstream steps run once. Row
expressions (filter predicates, derived columns) are evaluated through the safe
:mod:`pipedream.expr` evaluator, never Python ``eval``.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pandas as pd

from ..errors import ExecutionError, ExpressionError
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
    step_inputs,
)
from ..templating import render
from .executor import Executor, register
from .model import RuntimeModel, get_model


class PandasExecutor(Executor):
    name = "pandas"

    def __init__(self, model: RuntimeModel | None = None) -> None:
        # The runtime model is only resolved if a pipeline actually uses an LLM
        # op, so pipelines without one never touch model config.
        self._model = model

    def _runtime_model(self) -> RuntimeModel:
        if self._model is None:
            self._model = get_model()
        return self._model

    def run(self, pipeline: Pipeline) -> pd.DataFrame:
        cache: dict[str, pd.DataFrame] = {}
        step_by_id = pipeline.step_map()
        result = self._materialize(pipeline.output_id(), step_by_id, cache, [])
        return result.reset_index(drop=True)

    def _materialize(
        self,
        sid: str,
        step_by_id: dict[str, Step],
        cache: dict[str, pd.DataFrame],
        path: list[str],
    ) -> pd.DataFrame:
        if sid in cache:
            return cache[sid]
        step = step_by_id[sid]
        inputs = [
            self._materialize(ref, step_by_id, cache, path + [sid])
            for ref in step_inputs(step)
        ]
        try:
            out = self._execute_step(step, inputs)
        except ExecutionError:
            raise
        except ExpressionError as exc:
            raise ExecutionError(f"step {step.id!r}: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - surface engine errors with context
            raise ExecutionError(
                f"step {step.id!r} ({step.op}) failed: {exc}"
            ) from exc
        cache[sid] = out
        return out

    def _execute_step(self, step: Step, inputs: list[pd.DataFrame]) -> pd.DataFrame:
        if isinstance(step, LoadCsv):
            return pd.read_csv(step.path, header=0 if step.has_header else None)
        if isinstance(step, LoadInline):
            records = json.loads(step.data_json)
            return pd.DataFrame.from_records(records)
        if isinstance(step, Select):
            return inputs[0][list(step.columns)].copy()
        if isinstance(step, Filter):
            return self._filter(inputs[0], step)
        if isinstance(step, Derive):
            return self._derive(inputs[0], step)
        if isinstance(step, Aggregate):
            return self._aggregate(inputs[0], step)
        if isinstance(step, Join):
            return pd.merge(inputs[0], inputs[1], on=list(step.on), how=step.how)
        if isinstance(step, Sort):
            return inputs[0].sort_values(
                by=list(step.by), ascending=not step.descending
            )
        if isinstance(step, Limit):
            return inputs[0].head(step.count)
        if isinstance(step, Rename):
            mapping = {r.source: r.target for r in step.renames}
            return inputs[0].rename(columns=mapping)
        if isinstance(step, Classify):
            return self._classify(inputs[0], step)
        raise ExecutionError(f"no pandas implementation for op {step.op!r}")

    def _classify(self, df: pd.DataFrame, step: Classify) -> pd.DataFrame:
        model = self._runtime_model()
        prompts = [render(step.template, row) for row in df.to_dict("records")]

        def label_for(text: str) -> str:
            return model.classify(text, step.labels)

        workers = max(1, int(os.environ.get("PIPEDREAM_LLM_CONCURRENCY", "8")))
        if workers == 1 or len(prompts) <= 1:
            labels = [label_for(p) for p in prompts]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                labels = list(pool.map(label_for, prompts))

        out = df.copy()
        out[step.column] = pd.Series(labels, index=df.index)
        return out

    @staticmethod
    def _filter(df: pd.DataFrame, step: Filter) -> pd.DataFrame:
        predicate = compile_expr(step.predicate)
        mask = [bool(predicate.evaluate(row)) for row in df.to_dict("records")]
        return df[pd.Series(mask, index=df.index)]

    @staticmethod
    def _derive(df: pd.DataFrame, step: Derive) -> pd.DataFrame:
        expr = compile_expr(step.expr)
        values = [expr.evaluate(row) for row in df.to_dict("records")]
        out = df.copy()
        out[step.column] = pd.Series(values, index=df.index)
        return out

    @staticmethod
    def _aggregate(df: pd.DataFrame, step: Aggregate) -> pd.DataFrame:
        if step.group_by:
            grouped = df.groupby(list(step.group_by), dropna=False, sort=False)
            columns: dict[str, pd.Series] = {}
            for agg in step.aggregations:
                if agg.func == "count" and agg.column == "":
                    columns[agg.output] = grouped.size()
                else:
                    columns[agg.output] = grouped[agg.column].agg(agg.func)
            return pd.DataFrame(columns).reset_index()

        row: dict[str, Any] = {}
        for agg in step.aggregations:
            if agg.func == "count" and agg.column == "":
                row[agg.output] = len(df)
            else:
                row[agg.output] = _series_agg(df[agg.column], agg.func)
        return pd.DataFrame([row])


def _series_agg(series: pd.Series, func: str) -> Any:
    if func == "first":
        return series.iloc[0] if len(series) else None
    if func == "last":
        return series.iloc[-1] if len(series) else None
    return getattr(series, func)()


register("pandas", PandasExecutor)
