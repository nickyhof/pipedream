"""Deterministic analysis and optimization passes over a compiled pipeline.

These run after the LLM frontend produces an IR and before execution. They are
pure functions of the IR (no model calls), which makes them fast, testable, and
the place where we turn a plausible-looking pipeline into a provably-runnable
one: every reference resolves, the graph is acyclic, and unreachable work is
dropped.
"""

from __future__ import annotations

from .errors import CompileError
from .expr import compile_expr
from .ir import Derive, Filter, Pipeline, Step, step_inputs


def validate(pipeline: Pipeline) -> None:
    """Check structural and semantic invariants, raising on the first problem."""
    steps = pipeline.steps
    if not steps:
        raise CompileError("pipeline has no steps")

    ids = [s.id for s in steps]
    seen: set[str] = set()
    for sid in ids:
        if not sid:
            raise CompileError("every step must have a non-empty id")
        if sid in seen:
            raise CompileError(f"duplicate step id {sid!r}")
        seen.add(sid)

    known = set(ids)
    for step in steps:
        for ref in step_inputs(step):
            if ref not in known:
                raise CompileError(
                    f"step {step.id!r} references unknown input {ref!r}"
                )

    out = pipeline.output_id()
    if out not in known:
        raise CompileError(f"output references unknown step {out!r}")

    _check_acyclic(pipeline)
    _check_expressions(pipeline)


def _check_acyclic(pipeline: Pipeline) -> None:
    """Detect cycles via DFS with a recursion stack."""
    step_by_id = pipeline.step_map()
    state: dict[str, int] = {}  # 0 = visiting, 1 = done

    def visit(sid: str, path: list[str]) -> None:
        mark = state.get(sid)
        if mark == 1:
            return
        if mark == 0:
            cycle = " -> ".join(path + [sid])
            raise CompileError(f"pipeline has a cycle: {cycle}")
        state[sid] = 0
        for ref in step_inputs(step_by_id[sid]):
            visit(ref, path + [sid])
        state[sid] = 1

    for step in pipeline.steps:
        visit(step.id, [])


def _check_expressions(pipeline: Pipeline) -> None:
    """Compile every row expression so malformed ones fail at compile time."""
    for step in pipeline.steps:
        if isinstance(step, Filter):
            try:
                compile_expr(step.predicate)
            except Exception as exc:
                raise CompileError(
                    f"step {step.id!r} has an invalid predicate: {exc}"
                ) from exc
        elif isinstance(step, Derive):
            try:
                compile_expr(step.expr)
            except Exception as exc:
                raise CompileError(
                    f"step {step.id!r} has an invalid expression: {exc}"
                ) from exc


def reachable_ids(pipeline: Pipeline) -> set[str]:
    """Return the set of step ids that contribute to the pipeline output."""
    step_by_id = pipeline.step_map()
    keep: set[str] = set()
    stack = [pipeline.output_id()]
    while stack:
        sid = stack.pop()
        if sid in keep:
            continue
        keep.add(sid)
        stack.extend(step_inputs(step_by_id[sid]))
    return keep


def execution_order(pipeline: Pipeline) -> list[Step]:
    """Return reachable steps in dependency order (inputs before dependents)."""
    step_by_id = pipeline.step_map()
    ordered: list[Step] = []
    seen: set[str] = set()

    def visit(sid: str) -> None:
        if sid in seen:
            return
        for ref in step_inputs(step_by_id[sid]):
            visit(ref)
        seen.add(sid)
        ordered.append(step_by_id[sid])

    visit(pipeline.output_id())
    return ordered


def prune(pipeline: Pipeline) -> Pipeline:
    """Drop steps that don't feed the output (dead-step elimination)."""
    keep = reachable_ids(pipeline)
    if len(keep) == len(pipeline.steps):
        return pipeline
    pruned: list[Step] = [s for s in pipeline.steps if s.id in keep]
    return Pipeline(name=pipeline.name, steps=pruned, output=pipeline.output)


def analyze(pipeline: Pipeline, schema_provider=None) -> Pipeline:
    """Run the full middle-end: validate, prune, then schema-check.

    Validation runs first on the whole pipeline so that an error in a dead step
    is still surfaced rather than silently dropped. Schema analysis runs on the
    pruned pipeline (only the steps that actually execute). ``schema_provider``
    overrides how source columns are resolved; the default reads CSV headers and
    inline JSON, relaxing checks where a source can't be resolved.
    """
    from . import schema  # local import avoids an import cycle

    validate(pipeline)
    pruned = prune(pipeline)
    schema.check_schema(pruned, schema_provider)
    return pruned
