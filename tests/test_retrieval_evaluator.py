"""Tests for document and graph retrieval evaluation."""

import math

import pytest

from src.retrieval_evaluator import RetrievalEvaluationEngine


@pytest.fixture
def evaluator():
    return RetrievalEvaluationEngine(k=3)


def test_perfect_document_retrieval(evaluator):
    result = evaluator.evaluate_document_retrieval(
        expected_doc_ids=["DOC-001", "DOC-002"],
        retrieved_doc_ids=["DOC-001", "DOC-002", "DOC-003"],
    )

    assert result["precision_at_3"] == pytest.approx(2 / 3, abs=0.0001)
    assert result["recall_at_3"] == 1.0
    assert result["hit_rate_at_3"] == 1.0
    assert result["mrr_at_3"] == 1.0
    assert result["ndcg_at_3"] == 1.0


def test_relevant_document_at_second_position(evaluator):
    result = evaluator.evaluate_document_retrieval(
        expected_doc_ids=["DOC-001"],
        retrieved_doc_ids=["DOC-003", "DOC-001", "DOC-002"],
    )

    assert result["precision_at_3"] == pytest.approx(1 / 3, abs=0.0001)
    assert result["recall_at_3"] == 1.0
    assert result["mrr_at_3"] == 0.5
    assert result["ndcg_at_3"] == pytest.approx(
        1 / math.log2(3),
        abs=0.0001,
    )


def test_no_relevant_documents_retrieved(evaluator):
    result = evaluator.evaluate_document_retrieval(
        expected_doc_ids=["DOC-001"],
        retrieved_doc_ids=["DOC-002", "DOC-003"],
    )

    assert result["precision_at_3"] == 0.0
    assert result["recall_at_3"] == 0.0
    assert result["hit_rate_at_3"] == 0.0
    assert result["mrr_at_3"] == 0.0
    assert result["ndcg_at_3"] == 0.0


def test_empty_retrieval_results(evaluator):
    result = evaluator.evaluate_document_retrieval(
        expected_doc_ids=["DOC-001"],
        retrieved_doc_ids=[],
    )

    assert result["precision_at_3"] == 0.0
    assert result["recall_at_3"] == 0.0
    assert result["mrr_at_3"] == 0.0


def test_duplicate_predictions_are_removed(evaluator):
    result = evaluator.evaluate_document_retrieval(
        expected_doc_ids=["DOC-001"],
        retrieved_doc_ids=[
            "DOC-001",
            "DOC-001",
            "DOC-002",
        ],
    )

    assert result["retrieved_count"] == 2
    assert result["relevant_retrieved"] == 1
    assert result["mrr_at_3"] == 1.0


def test_normalization_is_case_insensitive(evaluator):
    result = evaluator.evaluate_document_retrieval(
        expected_doc_ids=["DOC-001"],
        retrieved_doc_ids=["doc-001"],
    )

    assert result["recall_at_3"] == 1.0
    assert result["mrr_at_3"] == 1.0


def test_empty_expected_documents_raise_error(evaluator):
    with pytest.raises(ValueError, match="expected relevant document"):
        evaluator.evaluate_document_retrieval(
            expected_doc_ids=[],
            retrieved_doc_ids=["DOC-001"],
        )


@pytest.mark.parametrize("invalid_k", [0, -1, 1.5, True])
def test_invalid_k_raises_error(invalid_k):
    with pytest.raises(ValueError, match="positive integer"):
        RetrievalEvaluationEngine(k=invalid_k)


def test_graph_entity_metrics(evaluator):
    result = evaluator.evaluate_graph_entities(
        expected_entities=["AuthService", "RedisCache"],
        retrieved_entities=["AuthService", "PostgresCluster"],
    )

    assert result["true_positives"] == 1
    assert result["false_positives"] == 1
    assert result["false_negatives"] == 1
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["f1_score"] == 0.5


def test_query_evaluation(evaluator):
    result = evaluator.evaluate_query({
        "id": "RET-001",
        "expected_doc_ids": ["DOC-001"],
        "retrieved_doc_ids": ["DOC-001", "DOC-002"],
        "expected_graph_entities": ["AuthService"],
        "retrieved_graph_entities": ["AuthService"],
    })

    assert result["query_id"] == "RET-001"
    assert result["document_retrieval"]["recall_at_3"] == 1.0
    assert result["graph_entities"]["f1_score"] == 1.0


def test_missing_query_field_raises_error(evaluator):
    with pytest.raises(ValueError, match="missing fields"):
        evaluator.evaluate_query({
            "id": "RET-001",
            "expected_doc_ids": ["DOC-001"],
        })


def test_dataset_aggregation(evaluator):
    samples = [
        {
            "id": "RET-001",
            "expected_doc_ids": ["DOC-001"],
            "retrieved_doc_ids": ["DOC-001"],
            "expected_graph_entities": ["AuthService"],
            "retrieved_graph_entities": ["AuthService"],
        },
        {
            "id": "RET-002",
            "expected_doc_ids": ["DOC-002"],
            "retrieved_doc_ids": ["DOC-003"],
            "expected_graph_entities": ["PaymentGateway"],
            "retrieved_graph_entities": ["PaymentGateway"],
        },
    ]

    report = evaluator.evaluate_dataset(samples)

    assert report["configuration"]["query_count"] == 2
    assert report["summary"]["document_retrieval"]["recall_at_3"] == 0.5
    assert report["summary"]["document_retrieval"]["mrr_at_3"] == 0.5
    assert report["summary"]["graph_entities"]["f1_score"] == 1.0
    assert len(report["per_query_results"]) == 2


def test_empty_dataset_raises_error(evaluator):
    with pytest.raises(ValueError, match="cannot be empty"):
        evaluator.evaluate_dataset([])


def test_invalid_dataset_type_raises_error(evaluator):
    with pytest.raises(TypeError, match="list"):
        evaluator.evaluate_dataset("invalid")
