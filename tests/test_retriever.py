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


def test_bm25_does_not_initialize_embedding_provider(retriever):
    class UnusedEmbeddingProvider:
        def fit(self, texts):
            raise AssertionError("BM25 must not fit embeddings.")

        def encode(self, texts):
            raise AssertionError("BM25 must not encode embeddings.")

    bm25_retriever = HybridGraphRetriever(
        documents_path=retriever.documents_path,
        graph_engine=retriever.graph_engine,
        embedding_provider=UnusedEmbeddingProvider(),
        documents=retriever.documents,
        retrieval_method="bm25",
    )

    assert bm25_retriever.embedding_provider is None
    assert bm25_retriever.document_embeddings is None

    result = bm25_retriever.retrieve(
        "PaymentGateway payment transactions",
        top_k=2,
    )

    assert result["retrieved_chunks"]
    assert result["retrieved_chunks"][0]["id"] == "DOC-001"


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


def test_retriever_loads_graph_from_json(
    tmp_path,
    valid_graph_data=None,
):
    import json

    graph_data = {
        "directed": True,
        "graph_type": "MultiDiGraph",
        "nodes": [
            {
                "id": "AuthService",
                "name": "AuthService",
                "entity_type": "SERVICE",
                "description": "Authentication service",
                "metadata": {},
            },
            {
                "id": "PaymentGateway",
                "name": "PaymentGateway",
                "entity_type": "SERVICE",
                "description": "Payment service",
                "metadata": {},
            },
        ],
        "edges": [
            {
                "source_id": "PaymentGateway",
                "target_id": "AuthService",
                "relation_type": "DEPENDS_ON",
                "weight": 1.0,
            }
        ],
    }

    graph_path = tmp_path / "llm_graph.json"
    graph_path.write_text(
        json.dumps(graph_data),
        encoding="utf-8",
    )

    documents = [
        {
            "id": "DOC-001",
            "title": "Payment Architecture",
            "category": "Architecture",
            "content": "PaymentGateway depends on AuthService.",
        }
    ]

    retriever = HybridGraphRetriever(
        graph_path=graph_path,
        documents=documents,
        embedding_provider=MockEmbeddingProvider(),
    )

    result = retriever.retrieve("PaymentGateway")

    assert "PaymentGateway" in [
        entity["name"] for entity in result["graph_entities"]
    ]

    assert (
        "PaymentGateway --[DEPENDS_ON]--> AuthService"
        in result["graph_context"]
    )


def test_retriever_rejects_graph_engine_and_graph_path_together(
    tmp_path,
):
    graph_engine = GraphQueryEngine(nx.MultiDiGraph())

    with pytest.raises(
        ValueError,
        match="either graph_engine or graph_path",
    ):
        HybridGraphRetriever(
            graph_engine=graph_engine,
            graph_path=tmp_path / "graph.json",
            documents=[],
            embedding_provider=MockEmbeddingProvider(),
        )


def test_retriever_default_graph_behavior_is_preserved(
    tmp_path,
):
    import json

    documents = [
        {
            "id": "DOC-001",
            "title": "Payment Architecture",
            "category": "Architecture",
            "content": (
                "PaymentGateway depends on PostgresCluster "
                "for payment transactions."
            ),
        }
    ]

    documents_path = tmp_path / "documents.json"
    documents_path.write_text(
        json.dumps(documents),
        encoding="utf-8",
    )

    retriever = HybridGraphRetriever(
        documents_path=documents_path,
        embedding_provider=MockEmbeddingProvider(),
    )

    assert retriever.graph_engine.graph.has_node("PaymentGateway")
    assert retriever.graph_engine.graph.has_node("PostgresCluster")


def test_rrf_retrieval_method_can_be_selected(retriever):
    rrf_retriever = HybridGraphRetriever(
        documents_path=retriever.documents_path,
        graph_engine=retriever.graph_engine,
        embedding_provider=MockEmbeddingProvider(),
        documents=retriever.documents,
        retrieval_method="rrf",
    )

    assert rrf_retriever.retrieval_method == "rrf"
    assert rrf_retriever.embedding_provider is not None
    assert rrf_retriever.document_embeddings is not None


def test_rrf_returns_expected_document_fields(retriever):
    rrf_retriever = HybridGraphRetriever(
        documents_path=retriever.documents_path,
        graph_engine=retriever.graph_engine,
        embedding_provider=MockEmbeddingProvider(),
        documents=retriever.documents,
        retrieval_method="rrf",
    )

    result = rrf_retriever.retrieve(
        "PaymentGateway payment transactions",
        top_k=3,
    )

    chunks = result["retrieved_chunks"]

    assert len(chunks) > 0
    assert chunks[0]["id"] == "DOC-001"
    assert chunks[0]["retrieval_method"] == "rrf"
    assert "retrieval_score" in chunks[0]
    assert "embedding_rank" in chunks[0]
    assert "bm25_rank" in chunks[0]
    assert "embedding_score" in chunks[0]
    assert "bm25_score" in chunks[0]
    assert chunks[0]["rrf_constant"] == 60


def test_rrf_score_is_calculated_from_ranks(retriever):
    rrf_retriever = HybridGraphRetriever(
        documents_path=retriever.documents_path,
        graph_engine=retriever.graph_engine,
        embedding_provider=MockEmbeddingProvider(),
        documents=retriever.documents,
        retrieval_method="rrf",
        rrf_constant=60,
    )

    results = rrf_retriever._retrieve_documents_rrf(
        "PaymentGateway payment transactions",
        top_k=3,
    )

    doc_001 = next(
        document
        for document in results
        if document["id"] == "DOC-001"
    )

    expected_score = sum(
        1.0 / (60 + rank)
        for rank in (
            doc_001["embedding_rank"],
            doc_001["bm25_rank"],
        )
        if rank is not None
    )

    assert doc_001["retrieval_score"] == pytest.approx(
        round(expected_score, 10)
    )


def test_rrf_combines_embedding_and_bm25_rankings(retriever):
    rrf_retriever = HybridGraphRetriever(
        documents_path=retriever.documents_path,
        graph_engine=retriever.graph_engine,
        embedding_provider=MockEmbeddingProvider(),
        documents=retriever.documents,
        retrieval_method="rrf",
    )

    results = rrf_retriever._retrieve_documents_rrf(
        "PaymentGateway payment transactions",
        top_k=3,
    )

    assert results

    assert any(
        document["embedding_rank"] is not None
        for document in results
    )

    assert any(
        document["bm25_rank"] is not None
        for document in results
    )


def test_rrf_returns_empty_results_when_no_documents_are_available(retriever):
    rrf_retriever = HybridGraphRetriever(
        documents_path=retriever.documents_path,
        graph_engine=retriever.graph_engine,
        embedding_provider=MockEmbeddingProvider(),
        documents=[],
        retrieval_method="rrf",
    )

    results = rrf_retriever._retrieve_documents_rrf(
        "payment transactions",
        top_k=3,
    )

    assert results == []


@pytest.mark.parametrize("invalid_constant", [0, -1, 1.5, True, "60"])
def test_rrf_rejects_invalid_constant(retriever, invalid_constant):
    with pytest.raises(
        ValueError,
        match="rrf_constant must be a positive integer",
    ):
        HybridGraphRetriever(
            documents_path=retriever.documents_path,
            graph_engine=retriever.graph_engine,
            embedding_provider=MockEmbeddingProvider(),
            documents=retriever.documents,
            retrieval_method="rrf",
            rrf_constant=invalid_constant,
        )


def test_rrf_keeps_negative_bm25_scores_and_original_rank(
    retriever,
    monkeypatch,
):
    rrf_retriever = HybridGraphRetriever(
        documents_path=retriever.documents_path,
        graph_engine=retriever.graph_engine,
        embedding_provider=MockEmbeddingProvider(),
        documents=retriever.documents,
        retrieval_method="rrf",
    )

    monkeypatch.setattr(
        rrf_retriever,
        "_retrieve_documents_tfidf",
        lambda query, top_k: [],
    )

    monkeypatch.setattr(
        rrf_retriever,
        "_retrieve_documents_bm25",
        lambda query, top_k: [
            {
                **retriever.documents[2],
                "retrieval_score": 0.0,
                "retrieval_method": "bm25",
            },
            {
                **retriever.documents[0],
                "retrieval_score": -0.5,
                "retrieval_method": "bm25",
            },
        ],
    )

    results = rrf_retriever._retrieve_documents_rrf(
        "payment transactions",
        top_k=3,
    )

    assert len(results) == 1
    assert results[0]["id"] == "DOC-001"
    assert results[0]["bm25_rank"] == 2
    assert results[0]["bm25_score"] == -0.5
    assert results[0]["retrieval_score"] == pytest.approx(1 / 62)



def test_graph_context_excludes_untraversed_relationships():
    graph = nx.MultiDiGraph()

    for entity in (
        "Root",
        "Neighbor",
        "SecondHopA",
        "SecondHopB",
    ):
        graph.add_node(entity, name=entity)

    graph.add_edge(
        "Root", "Neighbor",
        key="DEPENDS_ON",
        relation_type="DEPENDS_ON",
        weight=1.0,
    )
    graph.add_edge(
        "Neighbor", "SecondHopA",
        key="DEPENDS_ON",
        relation_type="DEPENDS_ON",
        weight=1.0,
    )
    graph.add_edge(
        "Neighbor", "SecondHopB",
        key="ROUTES_TO",
        relation_type="ROUTES_TO",
        weight=1.0,
    )

    # This connects two discovered nodes but is not part of the
    # breadth-first discovery paths from Root.
    graph.add_edge(
        "SecondHopA", "SecondHopB",
        key="WORKS_WITH",
        relation_type="WORKS_WITH",
        weight=1.0,
    )

    retriever = HybridGraphRetriever(
        graph_engine=GraphQueryEngine(graph),
        embedding_provider=MockEmbeddingProvider(),
        documents=[],
    )

    context = retriever._get_graph_context("Root", hops=2)

    assert "Root --[DEPENDS_ON]--> Neighbor" in context
    assert "Neighbor --[DEPENDS_ON]--> SecondHopA" in context
    assert "Neighbor --[ROUTES_TO]--> SecondHopB" in context

    assert (
        "SecondHopA --[WORKS_WITH]--> SecondHopB"
        not in context
    )



def test_graph_context_supports_multiple_seed_entities():
    graph = nx.MultiDiGraph()

    for entity in ("RootA", "RootB", "Shared", "Leaf"):
        graph.add_node(entity, name=entity)

    graph.add_edge(
        "RootA", "Shared",
        key="DEPENDS_ON",
        relation_type="DEPENDS_ON",
        weight=1.0,
    )
    graph.add_edge(
        "RootB", "Shared",
        key="ROUTES_TO",
        relation_type="ROUTES_TO",
        weight=1.0,
    )
    graph.add_edge(
        "RootB", "Leaf",
        key="PUBLISHES_TO",
        relation_type="PUBLISHES_TO",
        weight=1.0,
    )

    retriever = HybridGraphRetriever(
        graph_engine=GraphQueryEngine(graph),
        embedding_provider=MockEmbeddingProvider(),
        documents=[],
    )

    context = retriever._get_graph_context(
        ["RootA", "RootB"],
        hops=1,
    )

    assert "RootA --[DEPENDS_ON]--> Shared" in context
    assert "RootB --[ROUTES_TO]--> Shared" in context
    assert "RootB --[PUBLISHES_TO]--> Leaf" in context
