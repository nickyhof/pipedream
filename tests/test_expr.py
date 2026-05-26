import pytest

from pipedream.errors import ExpressionError
from pipedream.expr import compile_expr


def ev(src, row):
    return compile_expr(src).evaluate(row)


def test_comparisons_and_boolean_logic():
    row = {"status": "completed", "amount": 120}
    assert ev("status == 'completed' and amount > 100", row) is True
    assert ev("status == 'cancelled' or amount > 100", row) is True
    assert ev("not (amount > 100)", row) is False


def test_arithmetic_and_chained_comparison():
    row = {"a": 10, "b": 3}
    assert ev("a + b * 2", row) == 16
    assert ev("a // b", row) == 3
    assert ev("0 < b < a", row) is True


def test_membership():
    row = {"status": "pending"}
    assert ev("status in ['pending', 'completed']", row) is True
    assert ev("status not in ['cancelled']", row) is True


def test_functions():
    row = {"name": "Alice", "note": None, "tag": "abc"}
    assert ev("lower(name) == 'alice'", row) is True
    assert ev("startswith(tag, 'ab')", row) is True
    assert ev("contains(tag, 'b')", row) is True
    assert ev("is_null(note)", row) is True
    assert ev("coalesce(note, 'default')", row) == "default"


def test_unknown_column_raises():
    with pytest.raises(ExpressionError):
        ev("missing > 1", {"present": 1})


@pytest.mark.parametrize(
    "src",
    [
        "__import__('os').system('echo hi')",
        "obj.attr",
        "data[0]",
        "(lambda: 1)()",
        "[x for x in range(3)]",
        "evil()",
    ],
)
def test_disallowed_constructs_raise(src):
    with pytest.raises(ExpressionError):
        compile_expr(src)


def test_keyword_arguments_rejected():
    with pytest.raises(ExpressionError):
        compile_expr("round(amount, ndigits=2)")
