import json

from pipedream import parse_pipeline, save_pipeline
from pipedream.compiler import Compiler, cache_key, load_pipeline
from pipedream.ir import Filter, Limit, LoadInline, Pipeline, Select

DATA = '[{"a": 1}, {"a": 2}]'


class StubFrontend:
    """A deterministic frontend that records how many times it was invoked."""

    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline
        self.calls = 0

    def compile_source(self, source: str) -> Pipeline:
        self.calls += 1
        return self.pipeline


def make_pipeline():
    return Pipeline(
        name="stub",
        steps=[
            LoadInline(id="src", data_json=DATA),
            Filter(id="f", input="src", predicate="a > 1"),
            Limit(id="dead", input="src", count=1),
        ],
        output="f",
    )


def test_compile_runs_passes_and_prunes(tmp_path):
    frontend = StubFrontend(make_pipeline())
    compiler = Compiler(frontend=frontend, cache_dir=tmp_path / "cache")
    pipeline = compiler.compile("anything")
    assert {s.id for s in pipeline.steps} == {"src", "f"}  # dead step pruned
    assert frontend.calls == 1


def test_cache_hit_skips_frontend(tmp_path):
    frontend = StubFrontend(make_pipeline())
    compiler = Compiler(frontend=frontend, cache_dir=tmp_path / "cache")
    compiler.compile("same source")
    compiler.compile("same source")
    assert frontend.calls == 1  # second call served from cache


def test_no_cache_always_calls_frontend(tmp_path):
    frontend = StubFrontend(make_pipeline())
    compiler = Compiler(frontend=frontend, cache_dir=tmp_path / "cache")
    compiler.compile("s", use_cache=False)
    compiler.compile("s", use_cache=False)
    assert frontend.calls == 2


def test_cache_key_changes_with_source_and_model():
    assert cache_key("a", "m1") != cache_key("b", "m1")
    assert cache_key("a", "m1") != cache_key("a", "m2")
    assert cache_key("a", "m1") == cache_key("a", "m1")


def test_cache_envelope_records_provenance(tmp_path):
    frontend = StubFrontend(make_pipeline())
    cache_dir = tmp_path / "cache"
    compiler = Compiler(frontend=frontend, cache_dir=cache_dir, model="test-model")
    compiler.compile("hello source")
    entries = list(cache_dir.glob("*.json"))
    assert len(entries) == 1
    envelope = json.loads(entries[0].read_text())
    assert envelope["model"] == "test-model"
    assert envelope["source"] == "hello source"
    assert envelope["pipeline"]["name"] == "stub"


def test_pipeline_roundtrip(tmp_path):
    pipeline = Pipeline(
        name="rt",
        steps=[LoadInline(id="src", data_json=DATA), Select(id="p", input="src", columns=["a"])],
        output="p",
    )
    path = tmp_path / "ir.json"
    save_pipeline(pipeline, path)
    reloaded = load_pipeline(path)
    assert reloaded.model_dump() == pipeline.model_dump()
    assert parse_pipeline(path.read_text()).output_id() == "p"
