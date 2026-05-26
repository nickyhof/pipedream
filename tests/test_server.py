import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from pipedream.runtime.model import OpenAIEndpointModel
from pipedream.serve import create_app


class FakeGenerator:
    """Stands in for the Transformers backend — no torch, deterministic."""

    name = "fake-model"

    def generate(self, messages, max_tokens):
        user = messages[-1]["content"].lower()
        return "positive" if "loved" in user else "negative"


def _client() -> TestClient:
    return TestClient(create_app(FakeGenerator()))


def test_chat_completions_openai_shape():
    resp = _client().post(
        "/v1/chat/completions",
        json={
            "model": "x",
            "messages": [{"role": "user", "content": "loved it"}],
            "max_tokens": 8,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["content"] == "positive"
    assert data["choices"][0]["finish_reason"] == "stop"


def test_models_endpoint():
    data = _client().get("/v1/models").json()
    assert data["data"][0]["id"] == "fake-model"


def test_endpoint_model_round_trip():
    # The runtime's OpenAI client talks to our shipped server end-to-end.
    model = OpenAIEndpointModel(
        base_url="http://testserver/v1", model="fake", client=_client()
    )
    labels = ["positive", "negative"]
    assert model.classify("Review: loved it", labels) == "positive"
    assert model.classify("Review: hated it", labels) == "negative"
