"""The PipeDream intermediate representation.

A compiled pipeline is a directed acyclic graph of typed steps. Each step has a
unique ``id`` and an ``op`` discriminator; steps reference the output of earlier
steps by id. The IR is deliberately executor-agnostic: it describes *what*
transformation to apply, never *how* a particular engine should run it. That
separation is what lets the same IR run on DuckDB today and on a warehouse/Spark
executor later.

The IR is also the compiler's contract with the model. The JSON schema derived
from these models is what the LLM frontend is constrained to emit, so the field
shapes here are kept friendly to structured-output generation (discriminated
union via ``op``, no free-form ``dict`` payloads, expressions as strings).
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Leaf value types
# ---------------------------------------------------------------------------

AggFunc = Literal[
    "sum", "mean", "min", "max", "count", "first", "last", "median", "std", "nunique"
]
JoinHow = Literal["inner", "left", "right", "outer"]


class Aggregation(BaseModel):
    """A single aggregation produced by an ``aggregate`` step.

    ``column`` is the input column to aggregate. For ``count`` it may be left as
    an empty string to count rows.
    """

    column: str = Field(description="Input column to aggregate; '' for a row count.")
    func: AggFunc = Field(description="Aggregation function to apply.")
    output: str = Field(description="Name of the resulting column.")


class RenameItem(BaseModel):
    source: str = Field(description="Existing column name.")
    target: str = Field(description="New column name.")


# ---------------------------------------------------------------------------
# Steps
#
# Every step carries a literal ``op`` so the union below can be discriminated,
# and ``id`` so later steps can reference its output. Source steps (those that
# introduce data) have no input field; transform steps reference one or two
# upstream step ids.
# ---------------------------------------------------------------------------


class LoadCsv(BaseModel):
    op: Literal["load_csv"] = "load_csv"
    id: str = Field(description="Unique name for this step's output.")
    path: str = Field(description="Filesystem path to the CSV file.")
    has_header: bool = Field(default=True, description="Whether row one is a header.")


class LoadInline(BaseModel):
    """A source backed by literal data, supplied as a JSON array of objects.

    Primarily useful for examples and tests where the data is small and known
    up front, so a pipeline can run with no external files.
    """

    op: Literal["load_inline"] = "load_inline"
    id: str = Field(description="Unique name for this step's output.")
    data_json: str = Field(description="JSON array of row objects, e.g. '[{\"a\": 1}]'.")


class Select(BaseModel):
    op: Literal["select"] = "select"
    id: str
    input: str = Field(description="Step id whose output to project.")
    columns: list[str] = Field(description="Columns to keep, in order.")


class Filter(BaseModel):
    op: Literal["filter"] = "filter"
    id: str
    input: str
    predicate: str = Field(
        description=(
            "Boolean row expression. References columns by name and may use "
            "comparisons, and/or/not, arithmetic, and a small set of functions "
            "(lower, upper, len, abs, round, startswith, endswith, contains, "
            "is_null, coalesce). Example: status == 'completed' and amount > 100"
        )
    )


class Derive(BaseModel):
    op: Literal["derive"] = "derive"
    id: str
    input: str
    column: str = Field(description="Name of the new (or overwritten) column.")
    expr: str = Field(description="Row expression computing the column value.")


class Aggregate(BaseModel):
    op: Literal["aggregate"] = "aggregate"
    id: str
    input: str
    group_by: list[str] = Field(
        description="Columns to group by; empty list aggregates the whole table."
    )
    aggregations: list[Aggregation]


class Join(BaseModel):
    op: Literal["join"] = "join"
    id: str
    left: str = Field(description="Step id for the left side.")
    right: str = Field(description="Step id for the right side.")
    on: list[str] = Field(description="Column names to join on (present on both sides).")
    how: JoinHow = "inner"


class Sort(BaseModel):
    op: Literal["sort"] = "sort"
    id: str
    input: str
    by: list[str] = Field(description="Columns to sort by, primary first.")
    descending: bool = False


class Limit(BaseModel):
    op: Literal["limit"] = "limit"
    id: str
    input: str
    count: int = Field(description="Maximum number of rows to keep.")


class Rename(BaseModel):
    op: Literal["rename"] = "rename"
    id: str
    input: str
    renames: list[RenameItem]


class Classify(BaseModel):
    """Label each row using a runtime LLM.

    The ``template`` is filled per row (``{column}`` placeholders) and sent to a
    runtime model, which must answer with one of ``labels``. This is the one op
    that calls a model at execution time; the compiler frontend (which turns
    natural language into this IR) is unaffected.
    """

    op: Literal["classify"] = "classify"
    id: str
    input: str
    template: str = Field(
        description="Per-row prompt with {column} placeholders, e.g. 'Review: {text}'."
    )
    labels: list[str] = Field(description="The allowed output labels.")
    column: str = Field(description="Name of the resulting label column.")


Step = Annotated[
    Union[
        LoadCsv,
        LoadInline,
        Select,
        Filter,
        Derive,
        Aggregate,
        Join,
        Sort,
        Limit,
        Rename,
        Classify,
    ],
    Field(discriminator="op"),
]

# Steps that introduce data rather than transform an existing input.
SOURCE_OPS = frozenset({"load_csv", "load_inline"})


class Pipeline(BaseModel):
    """A complete compiled pipeline.

    ``steps`` is topologically meaningful only through id references, not list
    order; the runtime resolves execution order from the dependency graph. The
    ``output`` field names the step whose result is the pipeline's result; when
    omitted, the last step is used.
    """

    name: str = Field(description="Short human-readable pipeline name.")
    steps: list[Step]
    output: str | None = Field(
        default=None,
        description="Step id to return as the result; defaults to the last step.",
    )

    def step_map(self) -> dict[str, Step]:
        return {s.id: s for s in self.steps}

    def output_id(self) -> str:
        if self.output is not None:
            return self.output
        if not self.steps:
            raise ValueError("pipeline has no steps")
        return self.steps[-1].id


def step_inputs(step: Step) -> list[str]:
    """Return the upstream step ids a step depends on."""
    if isinstance(step, Join):
        return [step.left, step.right]
    if step.op in SOURCE_OPS:
        return []
    return [step.input]  # type: ignore[attr-defined]
