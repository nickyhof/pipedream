"""The compiler frontend: natural language -> IR, via Claude.

This is the only place that talks to a model. It constrains the response to the
:class:`~pipedream.ir.Pipeline` schema using structured outputs, so the model
cannot return anything that isn't a syntactically valid pipeline — the
middle-end passes then check that it's *semantically* valid.

The frontend is defined behind a small :class:`Frontend` protocol so the rest of
the compiler doesn't depend on Anthropic specifically, and so tests can inject a
deterministic stub instead of making network calls.
"""

from __future__ import annotations

from typing import Protocol

from .errors import CompileError
from .ir import Pipeline

DEFAULT_MODEL = "claude-opus-4-7"

# Stable instruction prefix. Kept free of per-request/volatile content (no
# timestamps, ids, or interpolated source) so it forms a cacheable prompt prefix
# — see the prompt-caching guidance: stable content first, volatile last.
SYSTEM_PROMPT = """\
You are the frontend of PipeDream, a compiler that turns a plain-English \
description of a data pipeline into a typed intermediate representation (IR).

You translate intent into a directed acyclic graph of steps. You never invent \
data, columns, or files that the description does not imply. If the description \
references a CSV file, emit a `load_csv` step for it. Give every step a short, \
descriptive, unique `id` (snake_case), and reference upstream steps by that id.

Available operations:
- load_csv(path, has_header): read a CSV file as a source.
- load_inline(data_json): a source from a literal JSON array of row objects.
- select(input, columns): keep only the named columns, in order.
- filter(input, predicate): keep rows where a boolean row expression holds.
- derive(input, column, expr): add or overwrite a column from a row expression.
- aggregate(input, group_by, aggregations): group and aggregate. Each \
aggregation is {column, func, output}; func is one of sum, mean, min, max, \
count, median, std, first, last, nunique. For a plain row count use \
func="count" with column="".
- join(left, right, on, how): join two steps on shared columns; how is one of \
inner, left, right, outer.
- sort(input, by, descending): order rows.
- limit(input, count): keep the first N rows.
- rename(input, renames): rename columns; each rename is {source, target}.
- classify(input, template, labels, column): label each row with a runtime LLM. \
The template is a per-row prompt with {column} placeholders (e.g. "Review: \
{text}"), labels is the list of allowed outputs, and column names the new label \
column. Use this only when the task needs a model to judge/categorize free text \
(e.g. sentiment, topic) that no deterministic rule could express.

Row expressions (used by filter and derive) reference columns by bare name and \
support comparisons (== != < <= > >=), membership (in / not in), boolean \
and/or/not, arithmetic (+ - * / // % **), and these functions only: lower, \
upper, len, abs, round, startswith, endswith, contains, is_null, coalesce. \
Do not use any other functions, attribute access, or indexing.

Set the pipeline `output` to the id of the step that produces the final result \
(usually the last transformation). Choose a concise `name` for the pipeline.
"""


class Frontend(Protocol):
    """Anything that can turn pipeline source text into an IR."""

    def compile_source(self, source: str) -> Pipeline: ...


class AnthropicFrontend:
    """Frontend backed by the Anthropic Messages API and structured outputs."""

    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None) -> None:
        self.model = model
        self._api_key = api_key

    def compile_source(self, source: str) -> Pipeline:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise CompileError(
                "the 'anthropic' package is required to compile from natural language"
            ) from exc

        try:
            client = (
                anthropic.Anthropic(api_key=self._api_key)
                if self._api_key
                else anthropic.Anthropic()
            )
        except Exception as exc:  # noqa: BLE001 - typically a missing API key
            raise CompileError(
                "could not initialize the Anthropic client; set ANTHROPIC_API_KEY "
                f"or pass api_key. ({exc})"
            ) from exc

        try:
            response = client.messages.parse(
                model=self.model,
                max_tokens=8000,
                thinking={"type": "adaptive"},
                system=[
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": source}],
                output_format=Pipeline,
            )
        except Exception as exc:  # noqa: BLE001 - transport/validation failures
            raise CompileError(f"the model failed to compile the pipeline: {exc}") from exc

        pipeline = response.parsed_output
        if pipeline is None:
            raise CompileError(
                "the model did not return a valid pipeline "
                f"(stop_reason={response.stop_reason!r})"
            )
        return pipeline
