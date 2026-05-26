"""Compile-and-repair loop: when generated IR fails validation, the compiler
feeds the error back to the frontend for a bounded number of fix attempts."""

import pytest

from pipedream.compiler import Compiler
from pipedream.errors import CompileError
from pipedream.ir import LoadInline, Pipeline, Select


def _bad() -> Pipeline:
    # References a step id that does not exist -> fails middle-end validation.
    return Pipeline(
        name="t", steps=[Select(id="x", input="ghost", columns=["a"])], output="x"
    )


def _good() -> Pipeline:
    return Pipeline(
        name="t",
        steps=[
            LoadInline(id="s", data_json='[{"a": 1}]'),
            Select(id="x", input="s", columns=["a"]),
        ],
        output="x",
    )


class ScriptedFrontend:
    """Returns a queued pipeline per call: compile uses the first, each repair
    the next."""

    def __init__(self, results):
        self._results = list(results)
        self.compile_calls = 0
        self.repair_calls = 0

    def compile_source(self, source):
        self.compile_calls += 1
        return self._results[0]

    def repair(self, source, previous, error):
        self.repair_calls += 1
        self.last_error = error
        self.last_previous = previous
        return self._results[self.repair_calls]


class NoRepairFrontend:
    def __init__(self, result):
        self._result = result
        self.compile_calls = 0

    def compile_source(self, source):
        self.compile_calls += 1
        return self._result


def test_repair_fixes_invalid_pipeline(tmp_path):
    fe = ScriptedFrontend([_bad(), _good()])
    out = Compiler(frontend=fe, cache_dir=tmp_path, max_repairs=1).compile("req")
    assert {s.id for s in out.steps} == {"s", "x"}
    assert fe.compile_calls == 1 and fe.repair_calls == 1
    # The validation error and the failed IR are handed to the model.
    assert "ghost" in fe.last_error
    assert "ghost" in fe.last_previous


def test_repair_exhausts_and_raises(tmp_path):
    fe = ScriptedFrontend([_bad(), _bad(), _bad()])
    with pytest.raises(CompileError, match="ghost"):
        Compiler(frontend=fe, cache_dir=tmp_path, max_repairs=2).compile("req")
    assert fe.repair_calls == 2  # bounded


def test_no_repair_when_disabled(tmp_path):
    fe = ScriptedFrontend([_bad(), _good()])
    with pytest.raises(CompileError):
        Compiler(frontend=fe, cache_dir=tmp_path, max_repairs=0).compile("req")
    assert fe.repair_calls == 0


def test_frontend_without_repair_degrades(tmp_path):
    fe = NoRepairFrontend(_bad())
    with pytest.raises(CompileError):
        Compiler(frontend=fe, cache_dir=tmp_path, max_repairs=2).compile("req")
    assert fe.compile_calls == 1


def test_repaired_result_is_cached(tmp_path):
    fe = ScriptedFrontend([_bad(), _good()])
    compiler = Compiler(frontend=fe, cache_dir=tmp_path, max_repairs=1)
    compiler.compile("req")
    compiler.compile("req")  # second call should hit the cache
    assert fe.compile_calls == 1 and fe.repair_calls == 1
