import json

import pytest

from pipedream import passes
from pipedream.errors import CompileError
from pipedream.ir import (
    Aggregate,
    Aggregation,
    Derive,
    Filter,
    Join,
    LoadCsv,
    LoadInline,
    Pipeline,
    Rename,
    RenameItem,
    Select,
)
from pipedream.schema import (
    FLOAT,
    INT,
    STR,
    Schema,
    infer_schemas,
)

TYPED = json.dumps([{"a": 1, "b": 1.5, "c": "x", "d": True, "e": None}])


def schema_of(pipeline) -> dict:
    return infer_schemas(pipeline)


def out_schema(pipeline) -> Schema:
    return infer_schemas(pipeline)[pipeline.output_id()]


def test_inline_type_inference():
    p = Pipeline(name="t", steps=[LoadInline(id="src", data_json=TYPED)], output="src")
    s = out_schema(p)
    assert s.get("a") == INT
    assert s.get("b") == FLOAT
    assert s.get("c") == STR
    assert s.get("d") == "bool"
    assert s.get("e") == "null"


def test_inline_mixed_int_float_is_float():
    p = Pipeline(
        name="t",
        steps=[LoadInline(id="src", data_json=json.dumps([{"x": 1}, {"x": 2.5}]))],
        output="src",
    )
    assert out_schema(p).get("x") == FLOAT


def test_csv_schema_from_disk():
    p = Pipeline(
        name="t",
        steps=[LoadCsv(id="orders", path="examples/data/orders.csv")],
        output="orders",
    )
    s = out_schema(p)
    assert s.get("order_id") == INT
    assert s.get("customer") == STR
    assert s.get("amount") == FLOAT


def test_unknown_column_in_select_raises():
    p = Pipeline(
        name="t",
        steps=[
            LoadInline(id="src", data_json=TYPED),
            Select(id="proj", input="src", columns=["a", "missing"]),
        ],
        output="proj",
    )
    with pytest.raises(CompileError, match="unknown column 'missing'"):
        schema_of(p)


def test_unknown_column_in_filter_expr_raises():
    p = Pipeline(
        name="t",
        steps=[
            LoadInline(id="src", data_json=TYPED),
            Filter(id="f", input="src", predicate="ghost > 1"),
        ],
        output="f",
    )
    with pytest.raises(CompileError, match="ghost"):
        schema_of(p)


def test_numeric_aggregation_on_string_raises():
    p = Pipeline(
        name="t",
        steps=[
            LoadInline(id="src", data_json=TYPED),
            Aggregate(
                id="agg",
                input="src",
                group_by=[],
                aggregations=[Aggregation(column="c", func="sum", output="bad")],
            ),
        ],
        output="agg",
    )
    with pytest.raises(CompileError, match="requires a numeric column"):
        schema_of(p)


def test_aggregation_output_types():
    p = Pipeline(
        name="t",
        steps=[
            LoadInline(id="src", data_json=TYPED),
            Aggregate(
                id="agg",
                input="src",
                group_by=["c"],
                aggregations=[
                    Aggregation(column="b", func="mean", output="avg_b"),
                    Aggregation(column="", func="count", output="n"),
                ],
            ),
        ],
        output="agg",
    )
    s = out_schema(p)
    assert s.get("c") == STR  # group key keeps its type
    assert s.get("avg_b") == FLOAT
    assert s.get("n") == INT


def test_join_missing_key_raises():
    left = json.dumps([{"id": 1}])
    right = json.dumps([{"other": 1}])
    p = Pipeline(
        name="t",
        steps=[
            LoadInline(id="l", data_json=left),
            LoadInline(id="r", data_json=right),
            Join(id="j", left="l", right="r", on=["id"], how="inner"),
        ],
        output="j",
    )
    with pytest.raises(CompileError, match="unknown column 'id'"):
        schema_of(p)


def test_join_collision_raises():
    left = json.dumps([{"id": 1, "name": "a"}])
    right = json.dumps([{"id": 1, "name": "b"}])
    p = Pipeline(
        name="t",
        steps=[
            LoadInline(id="l", data_json=left),
            LoadInline(id="r", data_json=right),
            Join(id="j", left="l", right="r", on=["id"], how="inner"),
        ],
        output="j",
    )
    with pytest.raises(CompileError, match="ambiguous column"):
        schema_of(p)


def test_rename_duplicate_target_raises():
    p = Pipeline(
        name="t",
        steps=[
            LoadInline(id="src", data_json=json.dumps([{"a": 1, "b": 2}])),
            Rename(id="ren", input="src", renames=[RenameItem(source="a", target="b")]),
        ],
        output="ren",
    )
    with pytest.raises(CompileError, match="duplicate column 'b'"):
        schema_of(p)


def test_derive_adds_typed_column():
    p = Pipeline(
        name="t",
        steps=[
            LoadInline(id="src", data_json=TYPED),
            Derive(id="d", input="src", column="b2", expr="b * 2"),
        ],
        output="d",
    )
    s = out_schema(p)
    assert s.get("b2") == FLOAT
    assert s.names() == ["a", "b", "c", "d", "e", "b2"]


def test_missing_source_relaxes_checks():
    # CSV file does not exist -> source schema unknown -> downstream not checked.
    p = Pipeline(
        name="t",
        steps=[
            LoadCsv(id="src", path="nope_does_not_exist.csv"),
            Select(id="proj", input="src", columns=["anything", "goes"]),
        ],
        output="proj",
    )
    s = out_schema(p)  # must not raise
    assert s.names() == ["anything", "goes"]


def test_analyze_raises_on_unknown_csv_column():
    # Integration: schema checking is wired into passes.analyze.
    p = Pipeline(
        name="t",
        steps=[
            LoadCsv(id="orders", path="examples/data/orders.csv"),
            Select(id="proj", input="orders", columns=["customer", "nonexistent"]),
        ],
        output="proj",
    )
    with pytest.raises(CompileError, match="nonexistent"):
        passes.analyze(p)
