"""Tests for hybrid document and graph retrieval."""

import networkx as nx
import numpy as np
import pytest

from src.graph_builder import KnowledgeGraphBuilder
from src.graph_engine import GraphQueryEngine
from src.retriever import HybridGraphRetriever


class MockEmbeddingProvider:
    """Deterministic embeddings for testing without external models."""

    def fit(self, texts):
        self.texts = texts

    def encode(self, texts):
        vectors = []

        for text in texts:
            normalized = text.casefold()
            vectors.append(
                [
                    float("payment" in normalized),
                    float("database" in normalized or "postgres" in normalized),
                    float("incident" in normalized),
                    1.0,
                ]
            )

        return np.asarray(vectors, dtype=float)


@pytest.fixture
def retriever(tmp_path):
    documents = [
        {
            "id": "DOC-001",
            "title": "Payment Architecture",
            "category": "Architecture",
            "content": (
                "PaymentGateway depends on PostgresCluster "
                "for payment transactions."
            ),
        },
        {
            "id": "DOC-002",
            "title": "Incident Report",
            "category": "Incident",
            "content": (
                "Incident-304 affected PaymentGateway "
                "during a database outage."
            ),
        },
        {
            "id": "DOC-003",
            "title": "Authentication",
            "category": "Architecture",
            "content": "AuthService validates incoming requests.",
        },
    ]

    corpus_path = tmp_path / "documents.json"
    import json

    corpus_path.write_text(json.dumps(documents), encoding="utf-8")

    builder = KnowledgeGraphBuilder()
    graph = builder.build_from_corpus(str(corpus_path))

    engine = GraphQueryEngine(graph)

    return HybridGraphRetriever(
        documents_path=corpus_path,
        graph_engine=engine,
        embedding_provider=MockEmbeddingProvider(),
    )


def test_retriever_initializes(retriever):
    assert len(retriever.documents) == 3
    assert retriever.document_embeddings is not None


def test_retrieves_relevant_documents(retriever):
    result = retriever.retrieve("PaymentGateway payment", top_k=2)

    assert len(result["retrieved_chunks"]) == 2
    assert result["retrieved_chunks"][0]["id"] == "DOC-001"


def test_detects_graph_entities(retriever):
    result = retriever.retrieve("Why did PaymentGateway fail?")

    entity_names = [
        entity["name"] for entity in result["graph_entities"]
    ]

    assert "PaymentGateway" in entity_names


def test_returns_graph_relationships(retriever):
    result = retriever.retrieve(
        "PaymentGateway database failure",
        hops=2,
    )

    assert len(result["graph_context"]) > 0
    assert any(
        "PaymentGateway" in relation
        for relation in result["graph_context"]
    )


def test_returns_combined_context(retriever):
    result = retriever.retrieve("PaymentGateway payment")

    assert "[Retrieved Documents]" in result["context_text"]
    assert "[Knowledge Graph Relations]" in result["context_text"]


def test_empty_query_raises_error(retriever):
    with pytest.raises(ValueError):
        retriever.retrieve("   ")


def test_invalid_top_k_raises_error(retriever):
    with pytest.raises(ValueError):
        retriever.retrieve("PaymentGateway", top_k=0)


def test_invalid_hops_raises_error(retriever):
    with pytest.raises(ValueError):
        retriever.retrieve("PaymentGateway", hops=-1)


def test_unknown_entity_returns_no_graph_entities(retriever):
    result = retriever.retrieve("Tell me about an unknown component")

    assert result["graph_entities"] == []


def test_empty_graph_returns_empty_graph_context():
    graph = nx.MultiDiGraph()
    engine = GraphQueryEngine(graph)

    retriever = HybridGraphRetriever(
        graph_engine=engine,
        embedding_provider=MockEmbeddingProvider(),
        documents=[],
    )

    result = retriever.retrieve("Unknown service")

    assert result["graph_entities"] == []
    assert result["graph_context"] == []


def test_retrieval_with_empty_documents():
    builder = KnowledgeGraphBuilder()
    engine = GraphQueryEngine(builder.graph)

    retriever = HybridGraphRetriever(
        graph_engine=engine,
        embedding_provider=MockEmbeddingProvider(),
        documents=[],
    )

    result = retriever.retrieve("PaymentGateway")

    assert result["retrieved_chunks"] == []


def test_missing_entity_relationships_do_not_crash(retriever):
    result = retriever.retrieve(
        "PaymentGateway",
        hops=0,
    )

    assert isinstance(result["graph_context"], list)


def test_tfidf_is_the_default_retrieval_method(retriever):
    assert retriever.retrieval_method == "tfidf"


def test_bm25_retrieval_method_can_be_selected(retriever):
    bm25_retriever = HybridGraphRetriever(
        documents_path=retriever.documents_path,
        graph_engine=retriever.graph_engine,
        embedding_provider=MockEmbeddingProvider(),
        documents=retriever.documents,
        retrieval_method="bm25",
    )

    assert bm25_retriever.retrieval_method == "bm25"


def test_invalid_retrieval_method_raises_error():
    with pytest.raises(
        ValueError,
        match="retrieval_method must be either 'tfidf' or 'bm25'",
    ):
        HybridGraphRetriever(
            documents=[],
            graph_engine=GraphQueryEngine(nx.MultiDiGraph()),
            embedding_provider=MockEmbeddingProvider(),
            retrieval_method="invalid",
        )


def test_bm25_returns_expected_document_fields(retriever):
    bm25_retriever = HybridGraphRetriever(
        documents_path=retriever.documents_path,
        graph_engine=retriever.graph_engine,
        embedding_provider=MockEmbeddingProvider(),
        documents=retriever.documents,
        retrieval_method="bm25",
    )

    result = bm25_retriever.retrieve(
        "PaymentGateway payment transactions",
        top_k=2,
    )

    chunks = result["retrieved_chunks"]

    assert len(chunks) == 2
    assert chunks[0]["id"] == "DOC-001"
    assert chunks[0]["retrieval_method"] == "bm25"
    assert "retrieval_score" in chunks[0]
    assert "similarity_score" not in chunks[0]


def test_bm25_preserves_graph_retrieval(retriever):
    bm25_retriever = HybridGraphRetriever(
        documents_path=retriever.documents_path,
        graph_engine=retriever.graph_engine,
        embedding_provider=MockEmbeddingProvider(),
        documents=retriever.documents,
        retrieval_method="bm25",
    )

    result = bm25_retriever.retrieve(
        "PaymentGateway database failure",
        hops=2,
    )

    entity_names = [
        entity["name"] for entity in result["graph_entities"]
    ]

    assert "PaymentGateway" in entity_names
    assert len(result["graph_context"]) > 0
