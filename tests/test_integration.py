
"""End-to-end integration tests for the GraphRAG system."""

import json
from typing import ClassVar

from src.pipeline import GraphRAGPipeline
from src.retriever import (
    HybridGraphRetriever,
    TfidfEmbeddingProvider,
)


class FakeMessage:
    content = (
        "PaymentGateway depends on AuthService "
        "for identity verification and authorization."
    )


class FakeChoice:
    message = FakeMessage()


class FakeResponse:
    choices: ClassVar[list[FakeChoice]] = [FakeChoice()]


class FakeCompletions:
    def __init__(self):
        self.last_request = None

    def create(self, **kwargs):
        self.last_request = kwargs
        return FakeResponse()


class FakeChat:
    def __init__(self):
        self.completions = FakeCompletions()


class FakeLLMClient:
    def __init__(self):
        self.chat = FakeChat()


def test_complete_graphrag_workflow(tmp_path):
    """Test documents through retrieval and LLM answer generation."""

    # 1. Create a small, isolated document corpus.
    documents = [
        {
            "id": "DOC-001",
            "title": "Payment Authentication Architecture",
            "category": "Architecture",
            "content": (
                "PaymentGateway depends on AuthService "
                "for identity verification and authorization."
            ),
        },
        {
            "id": "DOC-002",
            "title": "Inventory Architecture",
            "category": "Architecture",
            "content": (
                "InventoryService manages inventory records "
                "and stock availability."
            ),
        },
    ]

    corpus_path = tmp_path / "documents.json"
    corpus_path.write_text(
        json.dumps(documents),
        encoding="utf-8",
    )

    # 2. Initialize the real hybrid retriever.
    retriever = HybridGraphRetriever(
        documents_path=corpus_path,
        embedding_provider=TfidfEmbeddingProvider(),
    )

    # 3. Verify document loading and graph construction.
    assert len(retriever.documents) == 2
    assert retriever.graph_engine.graph.number_of_nodes() >= 2
    assert retriever.graph_engine.graph.number_of_edges() >= 1

    # 4. Perform actual hybrid retrieval.
    retrieval = retriever.retrieve(
        query="Why does PaymentGateway depend on AuthService?",
        top_k=2,
        hops=1,
    )

    assert retrieval["retrieved_chunks"]
    assert retrieval["graph_entities"]

    assert any(
        chunk["id"] == "DOC-001"
        for chunk in retrieval["retrieved_chunks"]
    )

    expected_relation = (
        "PaymentGateway --[DEPENDS_ON]--> AuthService"
    )

    assert expected_relation in retrieval["graph_context"]

    # 5. Initialize the pipeline with a fake LLM client.
    fake_client = FakeLLMClient()

    pipeline = GraphRAGPipeline(
        retriever=retriever,
        client=fake_client,
    )

    # 6. Generate an answer using the real retrieval output.
    result = pipeline.run(
        query="Why does PaymentGateway depend on AuthService?",
        top_k=2,
        hops=1,
    )

    # 7. Verify the final pipeline response.
    assert result["answer"]
    assert "PaymentGateway" in result["answer"]
    assert result["retrieved_chunks"]
    assert result["graph_context"]
    assert "PaymentGateway" in result["graph_entities_used"]
    assert result["latency_ms"] >= 0

    # 8. Verify that both evidence sources reached the LLM.
    request = fake_client.chat.completions.last_request

    assert request is not None

    prompt = request["messages"][1]["content"]

    assert "[Retrieved Documents]" in prompt
    assert "[Knowledge Graph Relations]" in prompt
    assert "PaymentGateway" in prompt
    assert "AuthService" in prompt
