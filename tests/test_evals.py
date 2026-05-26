"""Drive the eval suite offline as part of CI.

Each case replays its committed golden IR through the runtime and compares the
result to the golden expected.csv. This doubles as regression coverage for the
executors and the example IRs.
"""

import pytest

from pipedream.eval import load_cases, run_case

CASES = load_cases()


def test_cases_exist():
    assert CASES, "no eval cases found under evals/cases"


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_case_offline(case):
    result = run_case(case, live=False)
    assert result.passed, f"{case.name}: {result.detail}"
