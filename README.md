# PipeDream

An LLM-based **compiler and runtime** for data pipelines authored in plain English.

You describe what you want in natural language; PipeDream compiles it — using
Claude — into a typed, executor-agnostic intermediate representation (IR),
validates and optimizes that IR with deterministic passes, and runs it on a
pluggable runtime (in-memory pandas by default).

```
  orders.pipe                  Pipeline (IR)                result
 ┌────────────┐  frontend   ┌───────────────┐  passes   ┌──────────┐  executor
 │ plain      │ ──(Claude)─▶│ typed DAG of  │ ─────────▶│ validated│ ─────────▶ table
 │ English    │             │ steps         │  analyze  │ + pruned │ (pandas/duckdb)
 └────────────┘             └───────────────┘           └──────────┘
                                   │                                   ▲
                                   └────── on-disk cache ──────────────┘
                                  (run works offline after one compile)
```

## Why it's structured like a compiler

The "LLM" part is deliberately confined to a single phase — the **frontend** —
which is the only nondeterministic, network-bound step. Everything downstream
is ordinary, testable code:

- The model never executes anything. It only emits an IR, constrained to a JSON
  schema via structured outputs, so it cannot return something that isn't a
  syntactically valid pipeline.
- Deterministic **passes** then prove the IR is *semantically* runnable (every
  reference resolves, the graph is acyclic) and optimize it (dead-step
  elimination).
- The **runtime** is decoupled from both the model and the IR shape, so new
  backends can be added without touching the compiler.

This separation is also what makes the compile cache possible: once a pipeline
is compiled, the IR is saved to disk and `run` never needs the model (or an API
key) again.

## Setup

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .            # add ".[dev]" for tests
                            # ".[serve]" for the local model server (FastAPI)
                            # ".[local-model]" to load a Hugging Face model (torch)
```

Compiling natural language requires an Anthropic API key:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

Running an already-compiled IR (a `.json` file) needs no key.

## Quickstart

```bash
# 1. Compile plain English into an IR artifact (uses the model)
pipedream compile examples/orders.pipe -o orders.ir.json

# 2. Inspect the execution plan (no model needed for a .json IR)
pipedream explain orders.ir.json

# 3. Run it (offline)
pipedream run orders.ir.json
```

`run` and `explain` accept either a `.json` IR file or a `.pipe` source file. If
given source, they compile it first (cached after the first call):

```bash
pipedream run examples/orders.pipe --limit 20
```

Example output:

```
  month  revenue  order_count
2026-02   495.25            3
2026-03   384.99            3
2026-01   200.50            2
```

### CLI reference

| Command | Purpose |
| --- | --- |
| `pipedream compile <src.pipe> [-o out.json] [--no-cache]` | Compile source to IR; prints JSON or writes to `-o`. |
| `pipedream run <src.pipe\|ir.json> [--executor NAME] [--limit N] [-o out.csv]` | Execute and print (or write CSV). |
| `pipedream explain <src.pipe\|ir.json>` | Print the execution plan in dependency order. |

Global `--model` overrides the Claude model (default `claude-opus-4-7`).

### The `.pipe` source format

Free-form natural language. The compiler maps intent onto IR steps and never
invents data, columns, or files the description doesn't imply:

```text
Load the orders from examples/data/orders.csv.
Keep only the completed orders.
Compute total revenue per month, counting how many orders contributed.
Sort the months from highest revenue to lowest.
```

## Design

### Phases

| Phase | Module | Role |
| --- | --- | --- |
| Frontend | `llm.py` | NL → IR via Claude. Structured outputs constrain the response to the `Pipeline` schema; adaptive thinking is on; the stable schema/instruction prompt is prompt-cached. Hidden behind a `Frontend` protocol so tests inject a stub. |
| IR | `ir.py` | A Pydantic model of a pipeline: a DAG of typed `Step`s (discriminated union on `op`) referencing each other by `id`. Executor-agnostic — describes *what*, not *how*. |
| Middle-end | `passes.py` | Pure functions over the IR: reference resolution, duplicate-id and cycle detection, expression compilation, and dead-step elimination. |
| Schema analysis | `schema.py` | The type checker. Propagates a column schema (names + coarse dtypes) through the DAG and rejects unknown columns, bad/colliding join keys, and non-numeric aggregations at compile time. Sources resolve via a `SchemaProvider` (default reads CSV headers and inline JSON); unresolved sources relax downstream checks. |
| Expressions | `expr.py` | A safe, AST-walking evaluator for filter predicates and derived columns. Allowlists node types and a fixed function set — never `eval`. |
| Runtime | `runtime/` | An `Executor` ABC + registry with two built-in backends. `PandasExecutor` (default) walks the DAG in memory and evaluates row expressions through `expr.py`. `DuckDBExecutor` lowers the whole pipeline to SQL. |
| Driver | `compiler.py` | Orchestrates the phases and caches compiled IR on disk, keyed by a hash of `(schema version, model, instruction prompt, source)`. |
| Runtime models | `runtime/model.py`, `serve.py` | The *local* model a pipeline calls during execution (for the `classify` op) — separate from the Anthropic compiler frontend. A pluggable `RuntimeModel` registry with an OpenAI-compatible client, plus a shipped local server. |

### The IR

A `Pipeline` has a `name`, a list of `steps`, and an `output` step id (defaults
to the last step). Execution order is derived from id references, not list
order. Operations:

| Op | Shape |
| --- | --- |
| `load_csv` | `path`, `has_header` — source |
| `load_inline` | `data_json` (JSON array of row objects) — source, handy for tests |
| `select` | `input`, `columns` |
| `filter` | `input`, `predicate` (row expression) |
| `derive` | `input`, `column`, `expr` (row expression) |
| `aggregate` | `input`, `group_by`, `aggregations` (`{column, func, output}`) |
| `join` | `left`, `right`, `on`, `how` (inner/left/right/outer) |
| `sort` | `input`, `by`, `descending` |
| `limit` | `input`, `count` |
| `rename` | `input`, `renames` (`{source, target}`) |
| `classify` | `input`, `template` (`{column}` placeholders), `labels`, `column` — labels each row with a runtime LLM |

Aggregation functions: `sum`, `mean`, `min`, `max`, `count`, `median`, `std`,
`first`, `last`, `nunique`. For a plain row count use `func="count"` with
`column=""`.

### Row expressions

Used by `filter` and `derive`. Columns are referenced by bare name. Supported:

- comparisons `== != < <= > >=`, membership `in` / `not in`
- boolean `and` / `or` / `not`, arithmetic `+ - * / // % **`
- functions only: `lower`, `upper`, `len`, `abs`, `round`, `startswith`,
  `endswith`, `contains`, `is_null`, `coalesce`

Anything else (attribute access, indexing, lambdas, comprehensions, unknown
calls) is rejected at compile time. This means a compiled pipeline — even one
produced by the model — cannot run arbitrary code.

```text
status == 'completed' and amount > 100
coalesce(discount, 0)
```

### Schema analysis

After the structural passes, `schema.py` propagates a column **schema** through
the DAG and type-checks it. A schema is an ordered set of typed columns; a
`SchemaProvider` resolves *source* columns (the default reads CSV headers and
sniffs dtypes from disk, and infers `load_inline` schemas from the JSON). Each
op has a rule that both validates and produces the next schema — `select` must
project columns that exist, `filter`/`derive` expressions may only reference
existing columns, `join` keys must be present on both sides (and non-key
collisions are rejected), and numeric aggregations require a numeric column.

Errors become precise `CompileError`s, e.g.:

```
step 'proj' (select) references unknown column 'custmer'; available: order_id, customer, status, month, amount
```

If a source can't be resolved (its file isn't present at compile time) its
schema is *unknown* and checks downstream of it relax, so `compile` and
`explain` stay usable without the data on hand. `explain` prints the inferred
output schema when it's known:

```
$ pipedream explain examples/orders.ir.json
...
output: ranked
schema: month:str, revenue:float, order_count:int
```

Type checking is intentionally conservative — dtypes are inferred and carried
(so they can be shown and reused), but the only hard type rule is the numeric
aggregation one; expression-internal mismatches are inferred without being
rejected, to avoid false positives that would block valid pipelines. Names +
types here lay the groundwork for a future compile-and-repair loop (feed these
errors back to the model to fix).

### Pluggable executors

Two backends ship today, selected with `--executor` (or `get_executor(name)`):

| Backend | How it runs the IR |
| --- | --- |
| `pandas` (default) | Walks the DAG in memory, memoizing shared upstream steps, and evaluates row expressions row-by-row via the safe evaluator. Good for a fast, dependency-light run. |
| `duckdb` | **Lowers the IR to SQL**: each step becomes a temporary view defined in dependency order, and row expressions are translated from the same validated AST into SQL. Pushes work into DuckDB's engine. |

Both produce identical results — the parity tests run the same pipeline through
each and compare. That equivalence is the point of an executor-agnostic IR: the
`duckdb` backend reuses the exact expression grammar the `pandas` backend does
(`expr.py` exposes its parsed AST), translating it instead of evaluating it.

```bash
pipedream run examples/orders.ir.json --executor duckdb
```

Backends register themselves under a name and implement a single `run` method:

```python
from pipedream.runtime import register, Executor

class MyExecutor(Executor):
    name = "my_backend"
    def run(self, pipeline):
        ...  # pipeline is already validated

register("my_backend", MyExecutor)
```

Then `pipedream run ... --executor my_backend`, or `get_executor("my_backend")`
in code. By the time `run` is called the pipeline has passed semantic analysis,
so an executor may assume references resolve and the graph is acyclic. The IR is
intentionally engine-agnostic so the same expression grammar can be evaluated
row-wise (pandas) or lowered to SQL (a future warehouse backend).

### Compile cache

Compiled IR is written to `.pipedream/cache/<hash>.json`. The hash covers the
source text, model, instruction prompt, and a schema version, so any change
invalidates the entry. Each cache entry records provenance (model + source)
alongside the IR. Use `--no-cache` to force recompilation.

## Library use

```python
from pipedream import Compiler, get_executor

pipeline = Compiler().compile_file("examples/orders.pipe")  # needs ANTHROPIC_API_KEY
df = get_executor("pandas").run(pipeline)

# Or load a pre-compiled IR — no key required:
from pipedream import load_pipeline
df = get_executor("pandas").run(load_pipeline("examples/orders.ir.json"))
```

### Runtime LLM ops and the local model server

Two model uses, kept separate:

- **Compile time** — the frontend turns natural language into IR using **Anthropic** (cloud). Unchanged.
- **Run time** — the `classify` op calls a **local** model per row to label free text the IR couldn't classify deterministically (sentiment, topic, …). The pipeline's intent — the rendered template and the allowed `labels` — is sent in the prompt, so a stock instruct model behaves as a pipeline-aware classifier ("IR-awareness" comes from *what we send*, not a model trained on the IR).

Runtime models are pluggable like executors (`runtime/model.py`). The default `OpenAIEndpointModel` speaks the OpenAI chat API, configured by env:

```
PIPEDREAM_LLM_BASE_URL   default http://localhost:8000/v1
PIPEDREAM_LLM_MODEL      default "local"
PIPEDREAM_LLM_API_KEY    optional
PIPEDREAM_LLM_CONCURRENCY default 8   # per-row calls run concurrently
```

It works against any OpenAI-compatible server. PipeDream also **ships its own** (`pipedream.serve`): a small FastAPI app that loads a local Hugging Face model and exposes `/v1/chat/completions`. The heavy ML stack is an optional extra, imported lazily:

```bash
pip install -e ".[serve,local-model]"
pipedream-serve --model Qwen/Qwen2.5-0.5B-Instruct --port 8000   # terminal 1

export PIPEDREAM_LLM_BASE_URL=http://localhost:8000/v1
export PIPEDREAM_LLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct
pipedream run examples/reviews.ir.json --executor pandas          # terminal 2
```

Only the `pandas` executor runs `classify` (it calls a Python model per row); the `duckdb` backend reports it as unsupported. The `classify` example (`examples/reviews.*`) needs a running model server; `pipedream explain examples/reviews.ir.json` works offline.

## Evaluation

The compiler is judged by **behavior, not syntax**: cases live under
`evals/cases/<name>/` as self-contained dirs (`prompt.pipe`, `golden.ir.json`,
`data/`, `expected.csv`, and a small `case.json`), and a pipeline's output table
is compared to the golden `expected.csv`. Two modes:

```bash
python -m pipedream.eval          # offline: replay golden IRs, no API key
python -m pipedream.eval --live   # compile prompts via the model, score NL->IR
```

- **offline** (default) replays the committed `golden.ir.json` for each case —
  regression coverage for the runtime and golden IRs, with no model. The pytest
  suite drives this, so the cases double as integration tests.
- **live** compiles `prompt.pipe` through the model and scores the result. It
  measures NL->IR accuracy and is skipped when `ANTHROPIC_API_KEY` is unset.

Comparison is order-insensitive unless a case sets `"ordered": true` (i.e. the
pipeline itself sorts), and tolerant of float rounding.

## Testing

```bash
pip install -e ".[dev]"
pytest
```

The suite covers the expression evaluator (including rejection of unsafe
constructs), the analysis and schema passes, every executor op (pandas and
duckdb, with parity checks), compiler caching (via a stub frontend, so no
network), the CLI, and the eval cases — all offline.

## Project layout

```
src/pipedream/
  ir.py                 # typed IR (Pydantic)
  expr.py               # safe row-expression evaluator
  passes.py             # validation + optimization
  schema.py             # name resolution + type inference (the type checker)
  llm.py                # Anthropic frontend (NL -> IR)
  templating.py         # {column} templating for LLM ops
  compiler.py           # phase driver + on-disk cache
  cli.py                # compile / run / explain
  eval.py               # result-based eval harness (python -m pipedream.eval)
  serve.py              # local OpenAI-compatible model server (pipedream-serve)
  runtime/
    executor.py         # Executor ABC + registry
    pandas_executor.py  # default in-memory backend
    duckdb_executor.py  # SQL-lowering backend
    model.py            # RuntimeModel registry + OpenAI-compatible client
examples/
  orders.pipe           # natural-language source
  orders.ir.json        # pre-compiled IR (runs offline)
  reviews.ir.json       # classify example (needs a local model server)
  data/
evals/cases/            # eval fixtures (prompt, golden IR, data, expected)
tests/
```
