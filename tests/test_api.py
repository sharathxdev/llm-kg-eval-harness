"""Tests for the FastAPI GraphRAG endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.app import create_app
from src.pipeline import LLMGenerationError


class MockRetriever:
    """Predictable retriever for API tests."""

    def retrieve(self, query, top_k=3, hops=2):
        if query == "trigger retrieval error":
            raise ValueError("Simulated retrieval error.")

        return {
            "query": query,
            "retrieved_chunks": [
                {
                    "id": "DOC-001",
                    "title": "Payment Architecture",
                    "category": "Architecture",
                    "content": "PaymentGateway uses PostgresCluster.",
                    "similarity_score": 0.9,
                }
            ],
            "graph_context": [
                "PaymentGateway --[DEPENDS_ON]--> PostgresCluster"
            ],
            "graph_entities": [
                {
                    "id": "PaymentGateway",
                    "name": "PaymentGateway",
                    "entity_type": "SERVICE",
                }
            ],
            "context_text": "Payment architecture test context.",
        }


class MockPipeline:
    """Predictable GraphRAG pipeline for API tests."""

    def run(self, query, top_k=3, hops=2):
        if query == "trigger llm error":
            raise LLMGenerationError("Simulated LLM failure.")

        if query == "trigger validation error":
            raise ValueError("Simulated validation error.")

        return {
            "query": query,
            "answer": "PaymentGateway depends on PostgresCluster.",
            "retrieved_chunks": [
                {
                    "id": "DOC-001",
                    "title": "Payment Architecture",
                    "category": "Architecture",
                    "content": "PaymentGateway uses PostgresCluster.",
                    "similarity_score": 0.9,
                }
            ],
            "graph_entities_used": ["PaymentGateway"],
            "graph_context": [
                "PaymentGateway --[DEPENDS_ON]--> PostgresCluster"
            ],
            "latency_ms": 12.5,
        }


@pytest.fixture
def client():
    """Create an API client with injected mock components."""
    application = create_app(
        retriever=MockRetriever(),
        pipeline=MockPipeline(),
    )
    return TestClient(application)


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["service"] == (
        "GraphRAG Evaluation Harness API"
    )


def test_retrieve_endpoint_returns_evidence(client):
    response = client.post(
        "/retrieve",
        json={"query": "PaymentGateway"},
    )

    assert response.status_code == 200

    result = response.json()
    assert result["query"] == "PaymentGateway"
    assert len(result["retrieved_chunks"]) == 1
    assert "PaymentGateway" in result["graph_entities"][0]["name"]
    assert result["graph_context"]


def test_retrieve_endpoint_accepts_custom_parameters(client):
    response = client.post(
        "/retrieve",
        json={
            "query": "PaymentGateway",
            "top_k": 5,
            "hops": 3,
        },
    )

    assert response.status_code == 200


def test_query_endpoint_returns_generated_answer(client):
    response = client.post(
        "/query",
        json={"query": "Why did PaymentGateway fail?"},
    )

    assert response.status_code == 200

    result = response.json()
    assert result["query"] == "Why did PaymentGateway fail?"
    assert result["answer"] == (
        "PaymentGateway depends on PostgresCluster."
    )
    assert "retrieved_chunks" in result
    assert "graph_entities_used" in result
    assert "graph_context" in result
    assert "latency_ms" in result


def test_query_endpoint_accepts_custom_parameters(client):
    response = client.post(
        "/query",
        json={
            "query": "PaymentGateway",
            "top_k": 5,
            "hops": 3,
        },
    )

    assert response.status_code == 200


@pytest.mark.parametrize(
    "endpoint",
    ["/retrieve", "/query"],
)
def test_empty_query_is_rejected(client, endpoint):
    response = client.post(
        endpoint,
        json={"query": "   "},
    )

    assert response.status_code == 422


@pytest.mark.parametrize(
    "endpoint",
    ["/retrieve", "/query"],
)
def test_missing_query_is_rejected(client, endpoint):
    response = client.post(endpoint, json={})

    assert response.status_code == 422


@pytest.mark.parametrize(
    "endpoint",
    ["/retrieve", "/query"],
)
def test_top_k_below_minimum_is_rejected(client, endpoint):
    response = client.post(
        endpoint,
        json={
            "query": "PaymentGateway",
            "top_k": 0,
        },
    )

    assert response.status_code == 422


@pytest.mark.parametrize(
    "endpoint",
    ["/retrieve", "/query"],
)
def test_top_k_above_maximum_is_rejected(client, endpoint):
    response = client.post(
        endpoint,
        json={
            "query": "PaymentGateway",
            "top_k": 21,
        },
    )

    assert response.status_code == 422


@pytest.mark.parametrize(
    "endpoint",
    ["/retrieve", "/query"],
)
def test_negative_hops_are_rejected(client, endpoint):
    response = client.post(
        endpoint,
        json={
            "query": "PaymentGateway",
            "hops": -1,
        },
    )

    assert response.status_code == 422


def test_query_endpoint_maps_llm_failure_to_502(client):
    response = client.post(
        "/query",
        json={"query": "trigger llm error"},
    )

    assert response.status_code == 502
    assert response.json()["detail"] == (
        "The language model could not generate an answer."
    )


def test_query_endpoint_maps_validation_error_to_422(client):
    response = client.post(
        "/query",
        json={"query": "trigger validation error"},
    )

    assert response.status_code == 422


def test_retrieve_endpoint_maps_retrieval_error_to_422(client):
    response = client.post(
        "/retrieve",
        json={"query": "trigger retrieval error"},
    )

    assert response.status_code == 422


def test_query_too_long_is_rejected(client):
    response = client.post(
        "/query",
        json={"query": "a" * 2001},
    )

    assert response.status_code == 422


def test_retrieve_too_long_is_rejected(client):
    response = client.post(
        "/retrieve",
        json={"query": "a" * 2001},
    )

    assert response.status_code == 422
