"""Runtime LLM models — the local model the *pipeline* calls during execution.

This is separate from the compiler frontend (which turns natural language into
IR using Anthropic). These models are invoked by LLM ops like ``classify`` while
a pipeline runs, and are meant to be small/local. Like executors, they're
pluggable via a registry.

The default :class:`OpenAIEndpointModel` speaks the OpenAI chat-completions API,
so it works against anything that exposes that surface — the server shipped in
``pipedream.serve``, Ollama, LM Studio, vLLM, etc. It is configured by
environment variables so a pipeline doesn't hard-code an endpoint:

    PIPEDREAM_LLM_BASE_URL   default http://localhost:8000/v1
    PIPEDREAM_LLM_MODEL      default "local"
    PIPEDREAM_LLM_API_KEY    optional bearer token

"IR-awareness" is achieved by *what we send*: the classify prompt carries the
step's intent (the rendered template) and the allowed labels, so a stock instruct
model behaves as a pipeline-aware classifier without being trained on the IR.
"""

from __future__ import annotations

import os
from typing import Callable, Protocol

from ..errors import ExecutionError

DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_MODEL = "local"


class RuntimeModel(Protocol):
    """A model the runtime can call to classify a row's text."""

    def classify(self, text: str, labels: list[str]) -> str: ...


def build_classify_messages(text: str, labels: list[str]) -> list[dict[str, str]]:
    """Build the chat messages for a classify call.

    Shared so the prompt shape is consistent wherever classification happens.
    """
    label_list = ", ".join(labels)
    system = (
        "You are executing one step of a data pipeline: classifying an input "
        f"into exactly one of these labels: {label_list}. "
        "Reply with only the single best label and nothing else."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": text},
    ]


def match_label(response: str, labels: list[str]) -> str:
    """Map a model's free-text reply to one of ``labels`` (best effort)."""
    cleaned = response.strip()
    lowered = cleaned.lower()
    for label in labels:
        if lowered == label.lower():
            return label
    for label in labels:  # label appears somewhere in the reply
        if label.lower() in lowered:
            return label
    return cleaned.splitlines()[0].strip() if cleaned else (labels[0] if labels else "")


class OpenAIEndpointModel:
    """Calls an OpenAI-compatible ``/chat/completions`` endpoint."""

    name = "openai"

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
        client: "object | None" = None,
    ) -> None:
        self.base_url = (base_url or os.environ.get("PIPEDREAM_LLM_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        self.model = model or os.environ.get("PIPEDREAM_LLM_MODEL", DEFAULT_MODEL)
        self.api_key = api_key or os.environ.get("PIPEDREAM_LLM_API_KEY")
        self.timeout = timeout
        self._client = client  # optional preconfigured httpx.Client (used in tests)

    def classify(self, text: str, labels: list[str]) -> str:
        content = self._chat(build_classify_messages(text, labels), max_tokens=32)
        return match_label(content, labels)

    def _chat(self, messages: list[dict[str, str]], max_tokens: int) -> str:
        import httpx

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        client = self._client or httpx.Client(timeout=self.timeout)
        try:
            resp = client.post(
                f"{self.base_url}/chat/completions", json=payload, headers=headers
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"] or ""
        except Exception as exc:  # noqa: BLE001 - network/shape errors -> runtime error
            raise ExecutionError(
                f"runtime model request to {self.base_url} failed: {exc}"
            ) from exc
        finally:
            if self._client is None:
                client.close()


_REGISTRY: dict[str, Callable[[], RuntimeModel]] = {}


def register(name: str, factory: Callable[[], RuntimeModel]) -> None:
    _REGISTRY[name] = factory


def available() -> list[str]:
    return sorted(_REGISTRY)


def get_model(name: str = "openai") -> RuntimeModel:
    try:
        factory = _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown runtime model {name!r}; available: {', '.join(available()) or 'none'}"
        ) from None
    return factory()


register("openai", OpenAIEndpointModel)
