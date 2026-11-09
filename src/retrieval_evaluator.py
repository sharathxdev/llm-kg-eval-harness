"""Deterministic evaluation metrics for document and graph retrieval."""

from __future__ import annotations

import logging
import math
from typing import Any

from src.evaluator import EvaluationEngine

logger = logging.getLogger(__name__)


class RetrievalEvaluationEngine:
    """Evaluate ranked document retrieval and graph entity retrieval."""

    def __init__(self, k: int = 3) -> None:
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise ValueError("K must be a positive integer.")

        self.k = k
        self.entity_evaluator = EvaluationEngine()

    @staticmethod
    def _normalize_ranked_items(items: list[str]) -> list[str]:
        """Normalize ranked identifiers while preserving their order."""

        if not isinstance(items, list):
            raise TypeError("Ranked evaluation inputs must be lists.")

        normalized = []
        seen = set()

        for item in items:
            if not isinstance(item, str):
                raise TypeError("Every ranked evaluation item must be a string.")

            value = item.strip().casefold()

            if value and value not in seen:
                normalized.append(value)
                seen.add(value)

        return normalized

    @staticmethod
    def _mean(values: list[float]) -> float:
        """Calculate a safe arithmetic mean."""

        if not values:
            return 0.0

        return round(sum(values) / len(values), 4)

    def evaluate_document_retrieval(
        self,
        expected_doc_ids: list[str],
        retrieved_doc_ids: list[str],
    ) -> dict[str, Any]:
        """Calculate ranking metrics for one query."""

        expected = self.entity_evaluator._normalize_items(expected_doc_ids)
        ranked = self._normalize_ranked_items(retrieved_doc_ids)

        if not expected:
            raise ValueError(
                "At least one expected relevant document is required."
            )

        top_k = ranked[:self.k]
        relevant_count = sum(item in expected for item in top_k)

        precision_at_k = relevant_count / self.k
        recall_at_k = relevant_count / len(expected)
        hit_rate_at_k = float(relevant_count > 0)

        reciprocal_rank = 0.0

        for rank, document_id in enumerate(top_k, start=1):
            if document_id in expected:
                reciprocal_rank = 1.0 / rank
                break

        dcg = sum(
            1.0 / math.log2(rank + 1)
            for rank, document_id in enumerate(top_k, start=1)
            if document_id in expected
        )

        ideal_relevant_count = min(len(expected), self.k)

        ideal_dcg = sum(
            1.0 / math.log2(rank + 1)
            for rank in range(1, ideal_relevant_count + 1)
        )

        ndcg_at_k = dcg / ideal_dcg if ideal_dcg else 0.0

        return {
            f"precision_at_{self.k}": round(precision_at_k, 4),
            f"recall_at_{self.k}": round(recall_at_k, 4),
            f"hit_rate_at_{self.k}": round(hit_rate_at_k, 4),
            f"mrr_at_{self.k}": round(reciprocal_rank, 4),
            f"ndcg_at_{self.k}": round(ndcg_at_k, 4),
            "relevant_retrieved": relevant_count,
            "expected_relevant": len(expected),
            "retrieved_count": len(ranked),
        }

    def evaluate_graph_entities(
        self,
        expected_entities: list[str],
        retrieved_entities: list[str],
    ) -> dict[str, Any]:
        """Calculate graph entity precision, recall, and F1."""

        return self.entity_evaluator.evaluate_entities(
            expected_entities,
            retrieved_entities,
        )

    def evaluate_query(
        self,
        sample: dict[str, Any],
    ) -> dict[str, Any]:
        """Evaluate document and graph entity retrieval for one query."""

        if not isinstance(sample, dict):
            raise TypeError("Each evaluation sample must be a dictionary.")

        required_fields = [
            "expected_doc_ids",
            "retrieved_doc_ids",
            "expected_graph_entities",
            "retrieved_graph_entities",
        ]

        missing_fields = [
            field for field in required_fields if field not in sample
        ]

        if missing_fields:
            raise ValueError(
                f"Evaluation sample is missing fields: {missing_fields}"
            )

        document_metrics = self.evaluate_document_retrieval(
            sample["expected_doc_ids"],
            sample["retrieved_doc_ids"],
        )

        graph_entity_metrics = self.evaluate_graph_entities(
            sample["expected_graph_entities"],
            sample["retrieved_graph_entities"],
        )

        return {
            "query_id": sample.get("id", "UNKNOWN"),
            "document_retrieval": document_metrics,
            "graph_entities": graph_entity_metrics,
        }

    def evaluate_dataset(
        self,
        samples: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Evaluate a dataset using macro ranking and micro entity metrics."""

        if not isinstance(samples, list):
            raise TypeError("Evaluation samples must be provided as a list.")

        if not samples:
            raise ValueError("Evaluation dataset cannot be empty.")

        per_query_results = []

        document_metric_names = [
            f"precision_at_{self.k}",
            f"recall_at_{self.k}",
            f"hit_rate_at_{self.k}",
            f"mrr_at_{self.k}",
            f"ndcg_at_{self.k}",
        ]

        entity_counts = {
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
        }

        for index, sample in enumerate(samples):
            try:
                result = self.evaluate_query(sample)
            except (TypeError, ValueError) as exc:
                raise type(exc)(
                    f"Invalid sample at index {index}: {exc}"
                ) from exc

            per_query_results.append(result)

            for metric in entity_counts:
                entity_counts[metric] += (
                    result["graph_entities"][metric]
                )

        ranking_metrics = {
            metric: self._mean([
                result["document_retrieval"][metric]
                for result in per_query_results
            ])
            for metric in document_metric_names
        }

        graph_entity_metrics = self.entity_evaluator._calculate_metrics(
            **entity_counts
        )

        report = {
            "configuration": {
                "k": self.k,
                "query_count": len(samples),
                "ranking_aggregation": "macro",
                "graph_entity_aggregation": "micro",
            },
            "summary": {
                "document_retrieval": ranking_metrics,
                "graph_entities": graph_entity_metrics,
            },
            "per_query_results": per_query_results,
        }

        logger.info(
            "Retrieval evaluation completed for %d queries at K=%d.",
            len(samples),
            self.k,
        )

        return report
