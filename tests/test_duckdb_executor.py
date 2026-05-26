import json

import pytest

from pipedream import get_executor, load_pipeline, passes
from pipedream.ir import (
    Aggregate,
    Aggregation,
    Derive,
    Filter,
    Join,
    Limit,
    LoadInline,
    Pipeline,
    Select,
    Sort,
)

ORDERS = json.dumps(
    [
        {"customer": "alice", "status": "completed", "month": "2026-01", "amount": 120.5},
        {"customer": "bob", "status": "completed", "month": "2026-01", "amount": 80.0},
        {"customer": "alice", "status": "cancelled", "month": "2026-01", "amount": 200.0},
        {"customer": "carol", "status": "completed", "month": "2026-02", "amount": 300.0},
    ]
)


def duck(pipeline):
    return get_executor("duckdb").run(passes.analyze(pipeline))


def test_filter_select_sort_limit():
    p = Pipeline(
        name="t",
        steps=[
            LoadInline(id="src", data_json=ORDERS),
            Filter(id="done", input="src", predicate="status == 'completed'"),
            Select(id="cols", input="done", columns=["customer", "amount"]),
            Sort(id="ranked", input="cols", by=["amount"], descending=True),
            Limit(id="top", input="ranked", count=2),
        ],
        output="top",
    )
    df = duck(p)
    assert list(df["customer"]) == ["carol", "alice"]
    assert list(df["amount"]) == [300.0, 120.5]


def test_derive():
    p = Pipeline(
        name="t",
        steps=[
            LoadInline(id="src", data_json=ORDERS),
            Derive(id="net", input="src", column="net", expr="round(amount * 0.9, 2)"),
        ],
        output="net",
    )
    df = duck(p)
    by_customer = df[(df["customer"] == "alice") & (df["month"] == "2026-01")]
    assert by_customer["net"].iloc[0] == 108.45


def test_aggregate_grouped():
    p = Pipeline(
        name="t",
        steps=[
            LoadInline(id="src", data_json=ORDERS),
            Filter(id="done", input="src", predicate="status == 'completed'"),
            Aggregate(
                id="bymonth",
                input="done",
                group_by=["month"],
                aggregations=[
                    Aggregation(column="amount", func="sum", output="revenue"),
                    Aggregation(column="", func="count", output="n"),
                ],
            ),
        ],
        output="bymonth",
    )
    df = duck(p).sort_values("month").reset_index(drop=True)
    assert list(df["month"]) == ["2026-01", "2026-02"]
    assert list(df["revenue"]) == [200.5, 300.0]
    assert list(df["n"]) == [2, 1]


def test_aggregate_whole_table():
    p = Pipeline(
        name="t",
        steps=[
            LoadInline(id="src", data_json=ORDERS),
            Aggregate(
                id="total",
                input="src",
                group_by=[],
                aggregations=[
                    Aggregation(column="amount", func="sum", output="total_amount"),
                    Aggregation(column="customer", func="nunique", output="customers"),
                ],
            ),
        ],
        output="total",
    )
    df = duck(p)
    assert len(df) == 1
    assert df.loc[0, "total_amount"] == 700.5
    assert df.loc[0, "customers"] == 3


def test_join():
    left = json.dumps([{"id": 1, "name": "a"}, {"id": 2, "name": "b"}])
    right = json.dumps([{"id": 1, "score": 10}, {"id": 2, "score": 20}])
    p = Pipeline(
        name="t",
        steps=[
            LoadInline(id="l", data_json=left),
            LoadInline(id="r", data_json=right),
            Join(id="j", left="l", right="r", on=["id"], how="inner"),
        ],
        output="j",
    )
    df = duck(p).sort_values("id").reset_index(drop=True)
    assert set(df.columns) == {"id", "name", "score"}
    assert list(df["score"]) == [10, 20]


@pytest.mark.parametrize(
    "predicate,expected",
    [
        ("amount > 100", {"alice", "carol"}),  # alice 120.5 + alice 200.0, carol 300
        ("status in ['completed'] and amount < 100", {"bob"}),
        ("not is_null(customer) and contains(customer, 'a')", {"alice", "carol"}),
    ],
)
def test_expression_translation(predicate, expected):
    p = Pipeline(
        name="t",
        steps=[
            LoadInline(id="src", data_json=ORDERS),
            Filter(id="f", input="src", predicate=predicate),
        ],
        output="f",
    )
    df = duck(p)
    assert set(df["customer"]) == expected


def test_example_ir_end_to_end():
    pipeline = load_pipeline("examples/orders.ir.json")
    df = duck(pipeline)
    assert list(df["month"]) == ["2026-02", "2026-03", "2026-01"]
    assert list(df["order_count"]) == [3, 3, 2]
    assert df.loc[0, "revenue"] == 495.25
