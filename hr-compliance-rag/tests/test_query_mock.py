import os
import pytest


def test_query_endpoint_with_mock(monkeypatch):
    # Ensure MOCK_LLM is set before importing the app
    monkeypatch.setenv("MOCK_LLM", "1")
    from fastapi.testclient import TestClient
    from rag_pipeline.api_service import app

    client = TestClient(app)

    resp = client.post("/query", json={"query": "What is the maternity leave policy?"})
    assert resp.status_code == 200
    j = resp.json()
    assert "answer" in j
    assert j["answer"].startswith("MOCK_RESPONSE")
    assert isinstance(j.get("sources"), list)
