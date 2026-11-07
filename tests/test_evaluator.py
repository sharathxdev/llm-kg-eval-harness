
"""Unit tests for the entity and relationship evaluation engine."""

import pytest

from src.evaluator import EvaluationEngine


@pytest.fixture
def engine():
    """Provide a fresh evaluation engine."""
    return EvaluationEngine()


def test_entity_perfect_match(engine):
    result = engine.evaluate_entities(
        ["AuthService", "PaymentGateway"],
        ["AuthService", "PaymentGateway"],
    )

    assert result["true_positives"] == 2
    assert result["false_positives"] == 0
    assert result["false_negatives"] == 0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1_score"] == 1.0


def test_entity_partial_match(engine):
    result = engine.evaluate_entities(
        ["AuthService", "PaymentGateway", "PostgresCluster"],
        ["AuthService", "PaymentGateway", "RedisCache"],
    )

    assert result["true_positives"] == 2
    assert result["false_positives"] == 1
    assert result["false_negatives"] == 1
    assert result["precision"] == 0.6667
    assert result["recall"] == 0.6667
    assert result["f1_score"] == 0.6667


def test_entity_no_match(engine):
    result = engine.evaluate_entities(
        ["AuthService"],
        ["RedisCache"],
    )

    assert result["true_positives"] == 0
    assert result["false_positives"] == 1
    assert result["false_negatives"] == 1
    assert result["f1_score"] == 0.0


def test_entity_empty_expected(engine):
    result = engine.evaluate_entities(
        [],
        ["AuthService"],
    )

    assert result["true_positives"] == 0
    assert result["false_positives"] == 1
    assert result["false_negatives"] == 0
    assert result["precision"] == 0.0
    assert result["recall"] == 1.0


def test_entity_empty_predictions(engine):
    result = engine.evaluate_entities(
        ["AuthService"],
        [],
    )

    assert result["true_positives"] == 0
    assert result["false_positives"] == 0
    assert result["false_negatives"] == 1
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f1_score"] == 0.0


def test_entity_both_inputs_empty(engine):
    result = engine.evaluate_entities([], [])

    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1_score"] == 1.0


def test_entity_duplicate_handling(engine):
    result = engine.evaluate_entities(
        ["AuthService", "AuthService", "PaymentGateway"],
        ["AuthService", "AuthService", "PaymentGateway"],
    )

    assert result["true_positives"] == 2
    assert result["false_positives"] == 0
    assert result["false_negatives"] == 0
    assert result["expected_count"] == 2
    assert result["predicted_count"] == 2


def test_entity_case_and_whitespace_normalization(engine):
    result = engine.evaluate_entities(
        ["AuthService", "PaymentGateway"],
        [" authservice ", " PAYMENTGATEWAY "],
    )

    assert result["true_positives"] == 2
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1_score"] == 1.0


def test_relation_perfect_match(engine):
    result = engine.evaluate_relations(
        ["AuthService|DEPENDS_ON|PostgresCluster"],
        ["AuthService|DEPENDS_ON|PostgresCluster"],
    )

    assert result["true_positives"] == 1
    assert result["false_positives"] == 0
    assert result["false_negatives"] == 0
    assert result["f1_score"] == 1.0


def test_relation_partial_match(engine):
    result = engine.evaluate_relations(
        [
            "AuthService|DEPENDS_ON|PostgresCluster",
            "PaymentGateway|DEPENDS_ON|RedisCache",
        ],
        [
            "AuthService|DEPENDS_ON|PostgresCluster",
            "PaymentGateway|AFFECTS|RedisCache",
        ],
    )

    assert result["true_positives"] == 1
    assert result["false_positives"] == 1
    assert result["false_negatives"] == 1
    assert result["f1_score"] == 0.5


def test_relation_empty_inputs(engine):
    result = engine.evaluate_relations([], [])

    assert result["true_positives"] == 0
    assert result["false_positives"] == 0
    assert result["false_negatives"] == 0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1_score"] == 1.0


def test_combined_evaluation(engine):
    result = engine.evaluate_sample(
        expected_entities=["AuthService", "PostgresCluster"],
        predicted_entities=["AuthService", "RedisCache"],
        expected_relations=[
            "AuthService|DEPENDS_ON|PostgresCluster"
        ],
        predicted_relations=[
            "AuthService|DEPENDS_ON|PostgresCluster"
        ],
    )

    assert result["entities"]["true_positives"] == 1
    assert result["entities"]["false_positives"] == 1
    assert result["entities"]["false_negatives"] == 1

    assert result["relations"]["true_positives"] == 1

    assert result["overall"]["true_positives"] == 2
    assert result["overall"]["false_positives"] == 1
    assert result["overall"]["false_negatives"] == 1


def test_dataset_aggregation(engine):
    samples = [
        {
            "id": "EVAL-001",
            "expected_entities": ["AuthService", "PaymentGateway"],
            "predicted_entities": ["AuthService", "RedisCache"],
            "expected_relations": [
                "AuthService|DEPENDS_ON|PostgresCluster"
            ],
            "predicted_relations": [
                "AuthService|DEPENDS_ON|PostgresCluster"
            ],
        },
        {
            "id": "EVAL-002",
            "expected_entities": [],
            "predicted_entities": ["ExtraService"],
            "expected_relations": [],
            "predicted_relations": [],
        },
    ]

    result = engine.evaluate_dataset(samples)

    assert result["sample_count"] == 2
    assert len(result["per_sample_results"]) == 2

    assert result["entities"]["true_positives"] == 1
    assert result["entities"]["false_positives"] == 2
    assert result["entities"]["false_negatives"] == 1

    assert result["entities"]["precision"] == 0.3333
    assert result["entities"]["recall"] == 0.5
    assert result["entities"]["f1_score"] == 0.4

    assert result["overall"]["true_positives"] == 2
    assert result["overall"]["false_positives"] == 2
    assert result["overall"]["false_negatives"] == 1


def test_f1_zero_division(engine):
    result = engine.evaluate_entities(
        ["AuthService"],
        [],
    )

    assert result["f1_score"] == 0.0


def test_invalid_input_type(engine):
    with pytest.raises(TypeError):
        engine.evaluate_entities(
            ["AuthService"],
            "AuthService",
        )


def test_invalid_dataset_sample(engine):
    with pytest.raises(ValueError):
        engine.evaluate_dataset([
            {
                "id": "EVAL-001",
                "expected_entities": ["AuthService"],
            }
        ])
