import json

import pytest

from pipedream import passes
from pipedream.errors import CompileError
from pipedream.ir import Classify, Filter, LoadInline, Pipeline
from pipedream.runtime.duckdb_executor import DuckDBExecutor
from pipedream.runtime.model import build_classify_messages, match_label
from pipedream.schema import STR, infer_schemas

REVIEWS = json.dumps(
    [
        {"id": 1, "text": "loved it, fantastic"},
        {"id": 2, "text": "terrible, hated it"},
    ]
)


class StubModel:
    """Deterministic runtime model so classify tests need no network."""

    def __init__(self):
        self.calls = []

    def classify(self, text, labels):
        self.calls.append(text)
        return "positive" if "loved" in text else "negative"


def _pipeline():
    return Pipeline(
        name="sentiment",
        steps=[
            LoadInline(id="src", data_json=REVIEWS),
            Classify(
                id="c",
                input="src",
                template="Review: {text}",
                labels=["positive", "negative"],
                column="sentiment",
            ),
        ],
        output="c",
    )


def test_classify_adds_label_column():
    stub = StubModel()
    df = (
        DuckDBExecutor(model=stub)
        .run(passes.analyze(_pipeline()))
        .sort_values("id")
        .reset_index(drop=True)
    )
    assert list(df["sentiment"]) == ["positive", "negative"]
    # Template is rendered per row before the model is called.
    assert "Review: loved it, fantastic" in stub.calls


def test_classify_schema_adds_str_column():
    s = infer_schemas(passes.analyze(_pipeline()))["c"]
    assert s.get("sentiment") == STR
    assert s.names() == ["id", "text", "sentiment"]


def test_classify_unknown_template_column_raises():
    p = Pipeline(
        name="t",
        steps=[
            LoadInline(id="src", data_json=REVIEWS),
            Classify(
                id="c",
                input="src",
                template="Review: {missing}",
                labels=["a", "b"],
                column="out",
            ),
        ],
        output="c",
    )
    with pytest.raises(CompileError, match="missing"):
        passes.analyze(p)


def test_classify_requires_labels():
    p = Pipeline(
        name="t",
        steps=[
            LoadInline(id="src", data_json=REVIEWS),
            Classify(id="c", input="src", template="{text}", labels=[], column="out"),
        ],
        output="c",
    )
    with pytest.raises(CompileError, match="at least one label"):
        passes.analyze(p)


def test_classify_composes_with_sql():
    # The classify UDF produces a view that downstream SQL steps consume.
    p = Pipeline(
        name="t",
        steps=[
            LoadInline(id="src", data_json=REVIEWS),
            Classify(
                id="c",
                input="src",
                template="Review: {text}",
                labels=["positive", "negative"],
                column="sentiment",
            ),
            Filter(id="pos", input="c", predicate="sentiment == 'positive'"),
        ],
        output="pos",
    )
    df = DuckDBExecutor(model=StubModel()).run(passes.analyze(p))
    assert list(df["id"]) == [1]


def test_match_label():
    labels = ["positive", "negative", "neutral"]
    assert match_label("Positive", labels) == "positive"
    assert match_label("I'd say negative.", labels) == "negative"
    assert match_label("", labels) == "positive"  # falls back to first label


def test_build_classify_messages_includes_labels():
    msgs = build_classify_messages("hello", ["a", "b"])
    assert msgs[0]["role"] == "system" and "a, b" in msgs[0]["content"]
    assert msgs[1] == {"role": "user", "content": "hello"}
