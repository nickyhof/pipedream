"""A local, OpenAI-compatible model server for the runtime.

This is the model PipeDream pipelines call at execution time (e.g. for the
``classify`` op) — distinct from the Anthropic frontend that compiles natural
language into IR. It exposes ``POST /v1/chat/completions`` so the runtime's
:class:`~pipedream.runtime.model.OpenAIEndpointModel` (and any OpenAI client)
can talk to it, and loads a local Hugging Face model behind a small
:class:`Generator` seam.

The heavy ML stack (``transformers``/``torch``) is an optional extra and is
imported lazily, so importing this module — and unit-testing the HTTP surface
with a fake generator — needs neither.

Run it:

    pip install -e ".[serve,local-model]"
    pipedream-serve --model Qwen/Qwen2.5-0.5B-Instruct --port 8000

then point the runtime at it:

    export PIPEDREAM_LLM_BASE_URL=http://localhost:8000/v1
    export PIPEDREAM_LLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct
"""

from __future__ import annotations

import argparse
import time
import uuid
from typing import Protocol

from pydantic import BaseModel, ConfigDict


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    # protected_namespaces=() avoids pydantic's warning about the "model" field.
    model_config = ConfigDict(protected_namespaces=())
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int = 64
    temperature: float = 0.0


class Generator(Protocol):
    """Turns a chat message list into a completion string."""

    name: str

    def generate(self, messages: list[dict], max_tokens: int) -> str: ...


class TransformersGenerator:
    """Local Hugging Face causal-LM backend (optional ``transformers``/``torch``)."""

    def __init__(self, model: str) -> None:
        self.name = model
        try:
            from transformers import pipeline
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "the local model backend needs the 'local-model' extra: "
                'pip install "pipedream[local-model]"'
            ) from exc
        self._pipe = pipeline("text-generation", model=model)

    def generate(self, messages: list[dict], max_tokens: int) -> str:
        tokenizer = self._pipe.tokenizer
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        out = self._pipe(
            prompt, max_new_tokens=max_tokens, do_sample=False, return_full_text=False
        )
        return out[0]["generated_text"].strip()


def create_app(generator: Generator):
    """Build the OpenAI-compatible FastAPI app around a generator."""
    from fastapi import FastAPI

    app = FastAPI(title="PipeDream local model server")

    @app.get("/v1/models")
    def list_models() -> dict:
        return {"object": "list", "data": [{"id": generator.name, "object": "model"}]}

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatRequest) -> dict:
        messages = [{"role": m.role, "content": m.content} for m in req.messages]
        content = generator.generate(messages, req.max_tokens)
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model or generator.name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
        }

    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pipedream-serve",
        description="Serve a local OpenAI-compatible model for the runtime.",
    )
    parser.add_argument("--model", required=True, help="Hugging Face model id.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    import uvicorn

    app = create_app(TransformersGenerator(args.model))
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
