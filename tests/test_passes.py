import pytest

from pipedream import passes
from pipedream.errors import CompileError
from pipedream.ir import Filter, Limit, LoadInline, Pipeline, Select

DATA = '[{"a": 1, "b": 2}, {"a": 3, "b": 4}]'


def test_validate_accepts_well_formed_pipeline():
    p = Pipeline(
        name="ok",
        steps=[
            LoadInline(id="src", data_json=DATA),
            Select(id="proj", input="src", columns=["a"]),
        ],
        output="proj",
    )
    passes.validate(p)  # should not raise


def test_unknown_reference_rejected():
    p = Pipeline(
        name="bad",
        steps=[Select(id="proj", input="missing", columns=["a"])],
        output="proj",
    )
    with pytest.raises(CompileError, match="unknown input"):
        passes.validate(p)


def test_duplicate_ids_rejected():
    p = Pipeline(
        name="dup",
        steps=[
            LoadInline(id="src", data_json=DATA),
            LoadInline(id="src", data_json=DATA),
        ],
    )
    with pytest.raises(CompileError, match="duplicate step id"):
        passes.validate(p)


def test_cycle_detected():
    p = Pipeline(
        name="cyclic",
        steps=[
            Select(id="x", input="y", columns=["a"]),
            Select(id="y", input="x", columns=["a"]),
        ],
        output="x",
    )
    with pytest.raises(CompileError, match="cycle"):
        passes.validate(p)


def test_invalid_predicate_rejected():
    p = Pipeline(
        name="badpred",
        steps=[
            LoadInline(id="src", data_json=DATA),
            Filter(id="f", input="src", predicate="a > "),
        ],
        output="f",
    )
    with pytest.raises(CompileError, match="invalid predicate"):
        passes.validate(p)


def test_prune_drops_dead_steps():
    p = Pipeline(
        name="deadcode",
        steps=[
            LoadInline(id="src", data_json=DATA),
            Select(id="used", input="src", columns=["a"]),
            Limit(id="dead", input="src", count=1),
        ],
        output="used",
    )
    pruned = passes.prune(p)
    assert {s.id for s in pruned.steps} == {"src", "used"}


def test_analyze_validates_before_pruning():
    # A dead step with a bad reference must still surface as an error.
    p = Pipeline(
        name="deadbad",
        steps=[
            LoadInline(id="src", data_json=DATA),
            Select(id="used", input="src", columns=["a"]),
            Select(id="dead", input="ghost", columns=["a"]),
        ],
        output="used",
    )
    with pytest.raises(CompileError):
        passes.analyze(p)
