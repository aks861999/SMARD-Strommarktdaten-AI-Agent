import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
import api
from energy_agent.agent import MAX_HISTORY_MESSAGES

client = TestClient(api.app)


@pytest.fixture(autouse=True)
def _no_rate_limits_unless_a_test_asks(monkeypatch):
    """These tests post several times from one client; limits would make them depend on timing and order."""
    for name in ("RATE_LIMIT_PER_MINUTE", "RATE_LIMIT_PER_HOUR", "RATE_LIMIT_GLOBAL_PER_HOUR", "RATE_LIMIT_MAX_ACTIVE_PER_IP"):
        monkeypatch.setenv(name, "0")
    api._reset_rate_limits()
    api._JOBS.clear()
    yield
    monkeypatch.undo()
    api._reset_rate_limits()


def test_health_route():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_custom_catalog_is_versioned_and_lists_safe_interactions():
    catalog = client.get("/api/catalog").json()
    assert catalog["catalog_id"] == "urn:energy-agent:chat-genui:catalog:v1"
    assert "Chart" in catalog["components"]
    assert catalog["components"]["Calculator"]["operations"] == ["energy_cost", "multiply", "add", "subtract", "divide", "percentage"]
    assert catalog["components"]["AskAgentButton"]["actions"] == ["ask_agent"]


def test_provider_is_backend_configuration_not_a_client_choice(monkeypatch):
    # The client cannot select a provider; an unknown field is simply ignored.
    assert not hasattr(api.ChatRequest(message="hello", provider="llama-cpp"), "provider")
    monkeypatch.setenv("LLM_PROVIDER", "arbitrary-provider")
    with pytest.raises(ValueError):
        api.make_agent()


def test_chat_request_accepts_ten_exchanges_and_rejects_more():
    ten_exchanges = [{"role": "user" if i % 2 == 0 else "assistant", "content": "x"}
                     for i in range(MAX_HISTORY_MESSAGES)]
    assert MAX_HISTORY_MESSAGES == 20
    assert len(api.ChatRequest(message="hello", history=ten_exchanges).history) == 20
    with pytest.raises(ValidationError):
        api.ChatRequest(message="hello", history=ten_exchanges + [{"role": "user", "content": "x"}])


def test_sse_chat_stream_forwards_text_status_and_a2ui_messages(monkeypatch):
    ui = [{"version": "v0.9", "createSurface": {"surfaceId": "chat-ui", "catalogId": "urn:test"}}]
    seen = {}

    class FakeAgent:
        last_turn_state = {"date": "2026-11-16", "duration_hours": 3}
        def run(self, query, conversation_context=None, on_event=None, cancel_event=None, prior_state=None):
            seen["prior_state"] = prior_state
            on_event({"type": "status", "status": "Generating answer"})
            on_event({"type": "text_delta", "delta": "A streamed answer."})
            on_event({"type": "a2ui_message", "message": ui[0]})
            return "A streamed answer.", ui

    monkeypatch.setattr(api, "make_agent", lambda: FakeAgent())
    response = client.post("/api/chat", json={
        "message": "Explain energy prices", "history": [],
        "state": {"date": "2026-11-15"},
    })
    assert response.status_code == 200
    assert "text_delta" in response.text
    assert "A streamed answer." in response.text
    assert "a2ui_message" in response.text
    assert "createSurface" in response.text
    assert "done" in response.text
    assert '"state"' in response.text
    assert seen["prior_state"] == {"date": "2026-11-15"}


def test_reusing_a_request_id_for_a_different_payload_is_rejected(monkeypatch):
    class QuickAgent:
        last_turn_state = {}
        def run(self, query, conversation_context=None, on_event=None, cancel_event=None, prior_state=None):
            return "ok", []

    monkeypatch.setattr(api, "make_agent", lambda: QuickAgent())
    request_id = "collision-test-001"
    first = client.post("/api/chat", json={"message": "first", "request_id": request_id})
    assert first.status_code == 200
    second = client.post("/api/chat", json={"message": "different", "request_id": request_id})
    assert second.status_code == 409
    # The identical payload is still a valid resume, not a collision.
    resume = client.post("/api/chat", json={"message": "first", "request_id": request_id})
    assert resume.status_code == 200


def test_second_question_within_the_minute_gets_429_with_retry_after(monkeypatch):
    class QuickAgent:
        last_turn_state = {}
        def run(self, query, conversation_context=None, on_event=None, cancel_event=None, prior_state=None):
            return "ok", []

    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "1")
    api._reset_rate_limits()
    monkeypatch.setattr(api, "make_agent", lambda: QuickAgent())
    first = client.post("/api/chat", json={"message": "one", "request_id": "rl-test-1"})
    assert first.status_code == 200
    second = client.post("/api/chat", json={"message": "two", "request_id": "rl-test-2"})
    assert second.status_code == 429
    assert int(second.headers["retry-after"]) >= 1
    assert "wait" in second.json()["detail"].lower()
    # the same request_id with the same payload is a resume of the first question, not a new one
    resume = client.post("/api/chat", json={"message": "one", "request_id": "rl-test-1"})
    assert resume.status_code == 200
