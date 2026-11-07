
"""
Evaluation engine for entity and relationship extraction.

Provides deterministic precision, recall, and F1-score calculations
for individual samples and complete evaluation datasets.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class EvaluationEngine:
    """
    Evaluate entity and relationship extraction quality.

    Duplicate predictions and ground-truth items are removed after
    normalization.

    Empty-input convention:
    - Both sets empty: precision, recall, and F1 are 1.0.
    - No predictions but expected items exist: precision is 0.0.
    - No expected items but predictions exist: recall is 1.0,
      because there are no expected positives to miss.
    - F1 is 0.0 whenever precision + recall equals zero.
    """

    @staticmethod
    def _normalize_items(items: list[str]) -> set[str]:
        """Normalize strings and remove duplicates."""
        if not isinstance(items, list):
            raise TypeError("Evaluation inputs must be lists of strings.")

        normalized = set()

        for item in items:
            if not isinstance(item, str):
                raise TypeError("Every evaluation item must be a string.")

            value = item.strip().casefold()

            if value:
                normalized.add(value)

        return normalized

    @staticmethod
    def _calculate_metrics(
        true_positives: int,
        false_positives: int,
        false_negatives: int,
    ) -> dict[str, Any]:
        """Calculate precision, recall, and F1 from confusion counts."""

        total_predictions = true_positives + false_positives
        total_expected = true_positives + false_negatives

        if total_predictions == 0:
            precision = 1.0 if total_expected == 0 else 0.0
        else:
            precision = true_positives / total_predictions

        if total_expected == 0:
            recall = 1.0
        else:
            recall = true_positives / total_expected

        if precision + recall == 0:
            f1_score = 0.0
        else:
            f1_score = (
                2 * precision * recall
                / (precision + recall)
            )

        return {
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "expected_count": total_expected,
            "predicted_count": total_predictions,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1_score, 4),
        }

    def _evaluate(
        self,
        expected: list[str],
        predicted: list[str],
        evaluation_type: str,
    ) -> dict[str, Any]:
        """Shared evaluation implementation."""

        expected_set = self._normalize_items(expected)
        predicted_set = self._normalize_items(predicted)

        true_positives = len(expected_set & predicted_set)
        false_positives = len(predicted_set - expected_set)
        false_negatives = len(expected_set - predicted_set)

        metrics = self._calculate_metrics(
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives,
        )

        logger.info(
            "%s evaluation completed: precision=%.4f, recall=%.4f, f1=%.4f",
            evaluation_type,
            metrics["precision"],
            metrics["recall"],
            metrics["f1_score"],
        )

        return metrics

    def evaluate_entities(
        self,
        expected_entities: list[str],
        predicted_entities: list[str],
    ) -> dict[str, Any]:
        """Evaluate extracted entities."""
        return self._evaluate(
            expected_entities,
            predicted_entities,
            "Entity",
        )

    def evaluate_relations(
        self,
        expected_relations: list[str],
        predicted_relations: list[str],
    ) -> dict[str, Any]:
        """Evaluate extracted relationships."""
        return self._evaluate(
            expected_relations,
            predicted_relations,
            "Relationship",
        )

    def evaluate_sample(
        self,
        expected_entities: list[str],
        predicted_entities: list[str],
        expected_relations: list[str],
        predicted_relations: list[str],
    ) -> dict[str, Any]:
        """Evaluate entity and relationship extraction for one sample."""

        entity_metrics = self.evaluate_entities(
            expected_entities,
            predicted_entities,
        )

        relation_metrics = self.evaluate_relations(
            expected_relations,
            predicted_relations,
        )

        overall_metrics = self._calculate_metrics(
            true_positives=(
                entity_metrics["true_positives"]
                + relation_metrics["true_positives"]
            ),
            false_positives=(
                entity_metrics["false_positives"]
                + relation_metrics["false_positives"]
            ),
            false_negatives=(
                entity_metrics["false_negatives"]
                + relation_metrics["false_negatives"]
            ),
        )

        return {
            "entities": entity_metrics,
            "relations": relation_metrics,
            "overall": overall_metrics,
        }

    def evaluate_dataset(
        self,
        samples: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Evaluate multiple samples using aggregate micro-averaged metrics.

        Each sample must contain:
        expected_entities
        predicted_entities
        expected_relations
        predicted_relations
        """

        if not isinstance(samples, list):
            raise TypeError("Samples must be provided as a list.")

        entity_counts = {
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
        }

        relation_counts = {
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
        }

        per_sample_results = []

        for index, sample in enumerate(samples):
            if not isinstance(sample, dict):
                raise TypeError(
                    f"Sample at index {index} must be a dictionary."
                )

            required_fields = [
                "expected_entities",
                "predicted_entities",
                "expected_relations",
                "predicted_relations",
            ]

            missing_fields = [
                field
                for field in required_fields
                if field not in sample
            ]

            if missing_fields:
                raise ValueError(
                    f"Sample at index {index} is missing fields: "
                    f"{missing_fields}"
                )

            result = self.evaluate_sample(
                expected_entities=sample["expected_entities"],
                predicted_entities=sample["predicted_entities"],
                expected_relations=sample["expected_relations"],
                predicted_relations=sample["predicted_relations"],
            )

            per_sample_results.append({
                "sample_id": sample.get("id", f"SAMPLE-{index + 1:03d}"),
                **result,
            })

            for metric in entity_counts:
                entity_counts[metric] += result["entities"][metric]
                relation_counts[metric] += result["relations"][metric]

        entity_aggregate = self._calculate_metrics(**entity_counts)
        relation_aggregate = self._calculate_metrics(**relation_counts)

        overall_counts = {
            metric: entity_counts[metric] + relation_counts[metric]
            for metric in entity_counts
        }

        overall_aggregate = self._calculate_metrics(**overall_counts)

        logger.info(
            "Dataset evaluation completed for %d samples.",
            len(samples),
        )

        return {
            "sample_count": len(samples),
            "entities": entity_aggregate,
            "relations": relation_aggregate,
            "overall": overall_aggregate,
            "per_sample_results": per_sample_results,
        }
