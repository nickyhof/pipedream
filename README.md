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
 │ English    │             │ steps         │  analyze  │ + pruned │  (pandas)
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
pip install -e .            # add ".[dev]" for the test dependencies
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
| Expressions | `expr.py` | A safe, AST-walking evaluator for filter predicates and derived columns. Allowlists node types and a fixed function set — never `eval`. |
| Runtime | `runtime/` | An `Executor` ABC + registry. `PandasExecutor` is the default; it walks the DAG, memoizes shared steps, and evaluates row expressions through `expr.py`. |
| Driver | `compiler.py` | Orchestrates the phases and caches compiled IR on disk, keyed by a hash of `(schema version, model, instruction prompt, source)`. |

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

### Pluggable executors

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

## Testing

```bash
pip install -e ".[dev]"
pytest
```

The suite covers the expression evaluator (including rejection of unsafe
constructs), the analysis passes, every executor op, compiler caching (via a
stub frontend, so no network), and the CLI — all offline.

## Project layout

```
src/pipedream/
  ir.py                 # typed IR (Pydantic)
  expr.py               # safe row-expression evaluator
  passes.py             # validation + optimization
  llm.py                # Anthropic frontend (NL -> IR)
  compiler.py           # phase driver + on-disk cache
  cli.py                # compile / run / explain
  runtime/
    executor.py         # Executor ABC + registry
    pandas_executor.py  # default in-memory backend
examples/
  orders.pipe           # natural-language source
  orders.ir.json        # pre-compiled IR (runs offline)
  data/orders.csv
tests/
```
