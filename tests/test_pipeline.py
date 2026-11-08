
"""Tests for the GraphRAG generation pipeline."""

import pytest

from src.pipeline import GraphRAGPipeline, LLMGenerationError


class MockMessage:
    content = "PaymentGateway depends on PostgresCluster."


class MockChoice:
    message = MockMessage()


class MockResponse:
    choices = [MockChoice()]


class MockCompletions:
    def __init__(self):
        self.last_request = None

    def create(self, **kwargs):
        self.last_request = kwargs
        return MockResponse()


class MockChat:
    def __init__(self):
        self.completions = MockCompletions()


class MockClient:
    def __init__(self):
        self.chat = MockChat()


class MockRetriever:
    def __init__(self, result=None):
        self.result = result or {
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
            "context_text": "Test context",
        }

    def retrieve(self, query, top_k=3, hops=2):
        return self.result


@pytest.fixture
def client():
    return MockClient()


@pytest.fixture
def pipeline(client):
    return GraphRAGPipeline(
        retriever=MockRetriever(),
        client=client,
    )


def test_successful_generation(pipeline):
    result = pipeline.run("Why did PaymentGateway fail?")

    assert result["answer"] == (
        "PaymentGateway depends on PostgresCluster."
    )
    assert result["query"] == "Why did PaymentGateway fail?"
    assert result["latency_ms"] >= 0


def test_response_structure(pipeline):
    result = pipeline.run("PaymentGateway")

    assert "answer" in result
    assert "retrieved_chunks" in result
    assert "graph_entities_used" in result
    assert "graph_context" in result
    assert "latency_ms" in result


def test_graph_entities_are_returned(pipeline):
    result = pipeline.run("PaymentGateway")

    assert "PaymentGateway" in result["graph_entities_used"]


def test_prompt_contains_document_and_graph_evidence(
    pipeline,
    client,
):
    pipeline.run("PaymentGateway")

    request = client.chat.completions.last_request
    prompt = request["messages"][1]["content"]

    assert "[Retrieved Documents]" in prompt
    assert "[Knowledge Graph Relations]" in prompt


def test_empty_graph_context_uses_documents(client):
    retriever = MockRetriever({
        "retrieved_chunks": [
            {
                "id": "DOC-001",
                "title": "Payment",
                "category": "Architecture",
                "content": "Payment information.",
                "similarity_score": 0.8,
            }
        ],
        "graph_context": [],
        "graph_entities": [],
        "context_text": "Document context",
    })

    pipeline = GraphRAGPipeline(retriever, client)
    result = pipeline.run("Payment")

    assert result["answer"]
    assert result["graph_context"] == []


def test_empty_retrieval_does_not_call_llm(client):
    retriever = MockRetriever({
        "retrieved_chunks": [],
        "graph_context": [],
        "graph_entities": [],
        "context_text": "",
    })

    pipeline = GraphRAGPipeline(retriever, client)
    result = pipeline.run("Unknown component")

    assert "Insufficient evidence" in result["answer"]
    assert client.chat.completions.last_request is None


def test_invalid_query_raises_error(pipeline):
    with pytest.raises(ValueError):
        pipeline.run("   ")


def test_invalid_hops_raises_error(pipeline):
    with pytest.raises(ValueError):
        pipeline.run("PaymentGateway", hops=-1)


def test_invalid_top_k_raises_error(pipeline):
    with pytest.raises(ValueError):
        pipeline.run("PaymentGateway", top_k=0)


def test_llm_failure_raises_error():
    class FailingCompletions:
        def create(self, **kwargs):
            raise RuntimeError("Simulated API failure")

    class FailingChat:
        completions = FailingCompletions()

    class FailingClient:
        chat = FailingChat()

    pipeline = GraphRAGPipeline(
        MockRetriever(),
        FailingClient(),
    )

    with pytest.raises(LLMGenerationError):
        pipeline.run("PaymentGateway")


def test_missing_api_key_raises_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    pipeline = GraphRAGPipeline(
        MockRetriever(),
        client=None,
    )

    with pytest.raises(LLMGenerationError):
        pipeline.run("PaymentGateway")
