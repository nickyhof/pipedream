"""The compiler driver.

Ties the phases together: source text -> frontend (LLM) -> middle-end passes ->
a validated, optimized :class:`~pipedream.ir.Pipeline`. Compiled IR is cached on
disk keyed by a hash of the source, model, and instruction prompt, so a pipeline
is only sent to the model once. After that first compile, ``run`` works entirely
offline from the cached artifact — no API key required.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import passes
from .errors import CompileError
from .ir import Pipeline
from .llm import DEFAULT_MODEL, SYSTEM_PROMPT, AnthropicFrontend, Frontend

CACHE_DIR = Path(".pipedream/cache")
_SCHEMA_VERSION = "1"


def cache_key(source: str, model: str) -> str:
    """A stable digest of everything that affects compilation output."""
    h = hashlib.sha256()
    for part in (_SCHEMA_VERSION, model, SYSTEM_PROMPT, source):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def load_pipeline(path: str | Path) -> Pipeline:
    """Load and validate a pipeline from a JSON IR file."""
    text = Path(path).read_text(encoding="utf-8")
    return parse_pipeline(text)


def parse_pipeline(text: str) -> Pipeline:
    try:
        return Pipeline.model_validate_json(text)
    except Exception as exc:  # noqa: BLE001 - pydantic ValidationError et al.
        raise CompileError(f"invalid pipeline IR: {exc}") from exc


def save_pipeline(pipeline: Pipeline, path: str | Path) -> None:
    Path(path).write_text(pipeline.model_dump_json(indent=2), encoding="utf-8")


class Compiler:
    """Compiles natural-language pipeline source into a runnable IR."""

    def __init__(
        self,
        frontend: Frontend | None = None,
        model: str = DEFAULT_MODEL,
        cache_dir: str | Path = CACHE_DIR,
        max_repairs: int = 2,
    ) -> None:
        self.model = model
        self.frontend = frontend if frontend is not None else AnthropicFrontend(model)
        self.cache_dir = Path(cache_dir)
        self.max_repairs = max_repairs

    def compile(self, source: str, use_cache: bool = True) -> Pipeline:
        """Compile source text into a validated, optimized pipeline."""
        key = cache_key(source, self.model)
        cache_path = self.cache_dir / f"{key}.json"

        if use_cache and cache_path.exists():
            pipeline = self._read_cache(cache_path)
            if pipeline is not None:
                return passes.analyze(pipeline)

        pipeline = self.frontend.compile_source(source)
        analyzed = self._analyze_with_repair(source, pipeline)

        if use_cache:
            self._write_cache(cache_path, source, analyzed)
        return analyzed

    def _analyze_with_repair(self, source: str, pipeline: Pipeline) -> Pipeline:
        """Validate, and on failure ask the frontend to fix its own output.

        The middle-end's :class:`CompileError` is fed back to the model so it can
        correct semantic mistakes (unknown columns, bad join keys, ...) that
        structured outputs can't prevent. Bounded by ``max_repairs``.
        """
        can_repair = hasattr(self.frontend, "repair")
        error: CompileError | None = None
        for attempt in range(self.max_repairs + 1):
            try:
                return passes.analyze(pipeline)
            except CompileError as exc:
                error = exc
                if attempt >= self.max_repairs or not can_repair:
                    break
                pipeline = self.frontend.repair(
                    source, pipeline.model_dump_json(indent=2), str(exc)
                )
        assert error is not None  # the loop only breaks after catching one
        raise error

    def compile_file(self, path: str | Path, use_cache: bool = True) -> Pipeline:
        source = Path(path).read_text(encoding="utf-8")
        return self.compile(source, use_cache=use_cache)

    def _read_cache(self, path: Path) -> Pipeline | None:
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
            return Pipeline.model_validate(envelope["pipeline"])
        except Exception:  # noqa: BLE001 - a corrupt cache entry is a miss
            return None

    def _write_cache(self, path: Path, source: str, pipeline: Pipeline) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        envelope = {
            "schema_version": _SCHEMA_VERSION,
            "model": self.model,
            "source": source,
            "pipeline": pipeline.model_dump(),
        }
        path.write_text(json.dumps(envelope, indent=2), encoding="utf-8")
