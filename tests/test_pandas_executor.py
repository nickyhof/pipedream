import json

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
    Rename,
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


def run(pipeline):
    return get_executor("pandas").run(passes.analyze(pipeline))


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
    df = run(p)
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
    df = run(p)
    assert df.loc[0, "net"] == 108.45


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
    df = run(p).sort_values("month").reset_index(drop=True)
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
    df = run(p)
    assert len(df) == 1
    assert df.loc[0, "total_amount"] == 700.5
    assert df.loc[0, "customers"] == 3


def test_join_and_rename():
    left = json.dumps([{"id": 1, "name": "a"}, {"id": 2, "name": "b"}])
    right = json.dumps([{"id": 1, "score": 10}, {"id": 2, "score": 20}])
    p = Pipeline(
        name="t",
        steps=[
            LoadInline(id="l", data_json=left),
            LoadInline(id="r", data_json=right),
            Join(id="j", left="l", right="r", on=["id"], how="inner"),
            Rename(id="ren", input="j", renames=[]),
        ],
        output="j",
    )
    # rename with no-op renames list is allowed; output is the join.
    df = run(p).sort_values("id").reset_index(drop=True)
    assert list(df.columns) == ["id", "name", "score"]
    assert list(df["score"]) == [10, 20]


def test_shared_upstream_runs_once():
    # Two branches consume 'src'; result still correct (memoization in executor).
    p = Pipeline(
        name="t",
        steps=[
            LoadInline(id="src", data_json=ORDERS),
            Filter(id="a", input="src", predicate="amount > 100"),
            Select(id="out", input="a", columns=["customer"]),
            Filter(id="dead", input="src", predicate="amount < 100"),
        ],
        output="out",
    )
    df = run(p)
    assert sorted(df["customer"]) == ["alice", "alice", "carol"]


def test_example_ir_end_to_end():
    pipeline = load_pipeline("examples/orders.ir.json")
    df = get_executor("pandas").run(passes.analyze(pipeline))
    # Completed revenue per month, ranked descending.
    assert list(df["month"]) == ["2026-02", "2026-03", "2026-01"]
    assert list(df["order_count"]) == [3, 3, 2]
    assert df.loc[0, "revenue"] == 495.25
