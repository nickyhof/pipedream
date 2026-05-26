"""Command-line interface: compile, run, and explain pipelines."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import passes, schema
from .compiler import Compiler, load_pipeline, save_pipeline
from .errors import PipeDreamError
from .ir import (
    Aggregate,
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
from .llm import DEFAULT_MODEL
from .runtime import available, get_executor


def _load_or_compile(path: str, model: str, use_cache: bool) -> Pipeline:
    """Load IR from a .json file, or compile source from any other file."""
    if path.endswith(".json"):
        return passes.analyze(load_pipeline(path))
    return Compiler(model=model).compile_file(path, use_cache=use_cache)


def _describe(step: Step) -> str:
    if isinstance(step, LoadCsv):
        return f"load CSV {step.path!r}"
    if isinstance(step, LoadInline):
        return "load inline data"
    if isinstance(step, Select):
        return f"select [{', '.join(step.columns)}] from {step.input}"
    if isinstance(step, Filter):
        return f"filter {step.input} where {step.predicate}"
    if isinstance(step, Derive):
        return f"derive {step.column} = {step.expr} from {step.input}"
    if isinstance(step, Aggregate):
        by = ", ".join(step.group_by) if step.group_by else "(all rows)"
        aggs = ", ".join(f"{a.func}({a.column or '*'}) as {a.output}" for a in step.aggregations)
        return f"aggregate {step.input} by [{by}] -> {aggs}"
    if isinstance(step, Join):
        return f"join {step.left} {step.how} {step.right} on [{', '.join(step.on)}]"
    if isinstance(step, Sort):
        order = "desc" if step.descending else "asc"
        return f"sort {step.input} by [{', '.join(step.by)}] {order}"
    if isinstance(step, Limit):
        return f"limit {step.input} to {step.count} rows"
    if isinstance(step, Rename):
        pairs = ", ".join(f"{r.source}->{r.target}" for r in step.renames)
        return f"rename in {step.input}: {pairs}"
    return step.op


def render_plan(pipeline: Pipeline) -> str:
    lines = [f"pipeline: {pipeline.name}"]
    for i, step in enumerate(passes.execution_order(pipeline), start=1):
        marker = " *" if step.id == pipeline.output_id() else "  "
        lines.append(f"{marker}{i:>2}. {step.id}: {_describe(step)}")
    lines.append(f"output: {pipeline.output_id()}")
    out_schema = _safe_output_schema(pipeline)
    if out_schema is not None and out_schema.known:
        cols = ", ".join(f"{c.name}:{c.dtype}" for c in out_schema.columns)
        lines.append(f"schema: {cols}")
    return "\n".join(lines)


def _safe_output_schema(pipeline: Pipeline):
    """Inferred output schema, or None if it can't be determined."""
    try:
        return schema.infer_schemas(pipeline).get(pipeline.output_id())
    except Exception:  # noqa: BLE001 - explain should never hard-fail on schema
        return None


def cmd_compile(args: argparse.Namespace) -> int:
    pipeline = Compiler(model=args.model).compile_file(
        args.source, use_cache=not args.no_cache
    )
    if args.out:
        save_pipeline(pipeline, args.out)
        print(f"compiled {args.source} -> {args.out}")
    else:
        print(pipeline.model_dump_json(indent=2))
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    pipeline = _load_or_compile(args.input, args.model, not args.no_cache)
    print(render_plan(pipeline))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    pipeline = _load_or_compile(args.input, args.model, not args.no_cache)
    executor = get_executor(args.executor)
    result = executor.run(pipeline)
    if args.limit is not None:
        result = result.head(args.limit)
    if args.out:
        result.to_csv(args.out, index=False)
        print(f"wrote {len(result)} rows -> {args.out}")
    else:
        print(result.to_string(index=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pipedream",
        description="Compile and run natural-language data pipelines.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Claude model id.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_compile = sub.add_parser("compile", help="Compile pipeline source to IR.")
    p_compile.add_argument("source", help="Path to a .pipe (natural-language) file.")
    p_compile.add_argument("-o", "--out", help="Write IR JSON to this path.")
    p_compile.add_argument("--no-cache", action="store_true", help="Bypass the compile cache.")
    p_compile.set_defaults(func=cmd_compile)

    p_explain = sub.add_parser("explain", help="Show the execution plan for a pipeline.")
    p_explain.add_argument("input", help="A .pipe source file or a .json IR file.")
    p_explain.add_argument("--no-cache", action="store_true")
    p_explain.set_defaults(func=cmd_explain)

    p_run = sub.add_parser("run", help="Execute a pipeline and print the result.")
    p_run.add_argument("input", help="A .pipe source file or a .json IR file.")
    p_run.add_argument(
        "--executor",
        default="pandas",
        help=f"Runtime backend (available: {', '.join(available())}).",
    )
    p_run.add_argument("--limit", type=int, help="Only print the first N rows.")
    p_run.add_argument("-o", "--out", help="Write the result to this CSV path.")
    p_run.add_argument("--no-cache", action="store_true")
    p_run.set_defaults(func=cmd_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except PipeDreamError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"error: file not found: {exc.filename}", file=sys.stderr)
        return 1
    except KeyError as exc:
        # e.g. an unknown executor name from the registry.
        print(f"error: {exc.args[0] if exc.args else exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
