"""Result-based evaluation harness for the compiler.

Each case under ``evals/cases/<name>/`` is a self-contained directory:

    case.json        {description, ordered, executor}
    prompt.pipe      natural-language source
    golden.ir.json   a committed IR (used in offline/replay mode)
    data/            input files the pipeline reads (relative paths)
    expected.csv     the golden output table

Cases are judged by **behavior, not syntax**: a pipeline is run and its output
table compared to ``expected.csv``. There are two modes:

* **offline** (default): run the committed ``golden.ir.json``. No model, no API
  key — this is a regression check on the runtime + golden IRs, and is what the
  pytest suite drives.
* **live**: compile ``prompt.pipe`` through the model and score the result. This
  measures NL->IR accuracy and needs ``ANTHROPIC_API_KEY``; it is skipped when
  the key is absent.

Run with ``python -m pipedream.eval`` (add ``--live`` to score the model).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import pyarrow as pa

from . import passes, table
from .compiler import Compiler, load_pipeline
from .errors import PipeDreamError
from .runtime import get_executor

DEFAULT_CASES_DIR = Path("evals/cases")


@dataclass
class Case:
    name: str
    path: Path
    description: str
    ordered: bool
    executor: str

    @property
    def prompt(self) -> str:
        return (self.path / "prompt.pipe").read_text(encoding="utf-8")

    @property
    def golden_ir(self) -> Path:
        return self.path / "golden.ir.json"

    @property
    def expected(self) -> Path:
        return self.path / "expected.csv"


@dataclass
class Result:
    name: str
    passed: bool
    detail: str


def load_cases(root: str | Path = DEFAULT_CASES_DIR) -> list[Case]:
    root = Path(root)
    cases: list[Case] = []
    for case_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        manifest = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
        cases.append(
            Case(
                name=case_dir.name,
                path=case_dir,
                description=manifest.get("description", ""),
                ordered=bool(manifest.get("ordered", False)),
                executor=manifest.get("executor", "duckdb"),
            )
        )
    return cases


@contextmanager
def _in_dir(path: Path) -> Iterator[None]:
    """Run with cwd at the case dir so its relative data paths resolve."""
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def run_case(case: Case, live: bool = False, compiler: Compiler | None = None) -> Result:
    try:
        with _in_dir(case.path):
            if live:
                comp = compiler or Compiler()
                pipeline = comp.compile(case.prompt)
            else:
                pipeline = passes.analyze(load_pipeline("golden.ir.json"))
            actual = get_executor(case.executor).run(pipeline)
            expected = table.read_csv("expected.csv")
    except PipeDreamError as exc:
        return Result(case.name, False, f"error: {exc}")
    except Exception as exc:  # noqa: BLE001 - surface anything else as a failure
        return Result(case.name, False, f"error: {type(exc).__name__}: {exc}")

    ok, detail = compare(expected, actual, ordered=case.ordered)
    return Result(case.name, ok, detail)


def compare(expected: pa.Table, actual: pa.Table, ordered: bool) -> tuple[bool, str]:
    exp_cols, act_cols = expected.column_names, actual.column_names
    if set(exp_cols) != set(act_cols):
        return (
            False,
            f"columns differ: expected {sorted(exp_cols)}, got {sorted(act_cols)}",
        )
    exp_rows = _rows(expected, exp_cols)
    act_rows = _rows(actual, exp_cols)
    if not ordered:
        key = lambda r: tuple(str(x) for x in r)  # noqa: E731
        exp_rows, act_rows = sorted(exp_rows, key=key), sorted(act_rows, key=key)
    if exp_rows == act_rows:
        return True, f"{len(exp_rows)} rows match"
    return False, f"expected {exp_rows} but got {act_rows}"


def _rows(t: pa.Table, columns: list[str]) -> list[tuple]:
    return [tuple(_canon(rec[c]) for c in columns) for rec in t.to_pylist()]


def _canon(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return None if math.isnan(value) else round(float(value), 6)
    if isinstance(value, int):
        return int(value)
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pipedream.eval", description="Run the PipeDream eval suite."
    )
    parser.add_argument("--cases", default=str(DEFAULT_CASES_DIR), help="Cases directory.")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Compile prompts via the model (needs ANTHROPIC_API_KEY).",
    )
    args = parser.parse_args(argv)

    if args.live and not os.environ.get("ANTHROPIC_API_KEY"):
        print("skipped: --live requires ANTHROPIC_API_KEY")
        return 0

    cases = load_cases(args.cases)
    compiler = Compiler() if args.live else None
    results = [run_case(c, live=args.live, compiler=compiler) for c in cases]

    mode = "live" if args.live else "offline"
    print(f"PipeDream eval ({mode}) — {args.cases}\n")
    for r in results:
        mark = "PASS" if r.passed else "FAIL"
        print(f"  [{mark}] {r.name}: {r.detail}")
    passed = sum(r.passed for r in results)
    print(f"\n{passed}/{len(results)} cases passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
