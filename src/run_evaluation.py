
"""
Command-line runner for the LLM Knowledge Graph evaluation framework.

Loads golden evaluation samples and synthetic predictions,
calculates extraction metrics, and exports a JSON report.
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.evaluator import EvaluationEngine

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

GOLDEN_DATASET_PATH = (
    PROJECT_ROOT / "evals" / "golden_dataset.json"
)

PREDICTIONS_PATH = (
    PROJECT_ROOT / "evals" / "sample_predictions.json"
)

REPORT_DIRECTORY = PROJECT_ROOT / "evals" / "reports"

REPORT_PATH = REPORT_DIRECTORY / "evaluation_report.json"


# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def load_json(path: Path) -> Any:
    """Load and parse a JSON file."""

    if not path.exists():
        raise FileNotFoundError(
            f"Required file does not exist: {path}"
        )

    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON in {path}: {error}"
        ) from error


def validate_dataset(
    dataset: Any,
    dataset_name: str,
    required_fields: set[str],
) -> dict[str, dict[str, Any]]:
    """Validate dataset structure and index records by ID."""

    if not isinstance(dataset, list):
        raise TypeError(
            f"{dataset_name} must contain a JSON array."
        )

    indexed_records = {}

    for index, record in enumerate(dataset):
        if not isinstance(record, dict):
            raise TypeError(
                f"{dataset_name} record {index} must be an object."
            )

        missing_fields = required_fields - record.keys()

        if missing_fields:
            raise ValueError(
                f"{dataset_name} record {index} is missing: "
                f"{sorted(missing_fields)}"
            )

        record_id = record["id"]

        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError(
                f"{dataset_name} record {index} has an invalid ID."
            )

        if record_id in indexed_records:
            raise ValueError(
                f"Duplicate ID '{record_id}' in {dataset_name}."
            )

        for field in required_fields - {"id", "text"}:
            if not isinstance(record[field], list):
                raise TypeError(
                    f"Field '{field}' in {record_id} must be a list."
                )

        indexed_records[record_id] = record

    return indexed_records


def build_evaluation_samples(
    golden_records: dict[str, dict[str, Any]],
    prediction_records: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Match predictions to ground truth using sample IDs."""

    golden_ids = set(golden_records)
    prediction_ids = set(prediction_records)

    missing_predictions = golden_ids - prediction_ids
    unexpected_predictions = prediction_ids - golden_ids

    if missing_predictions or unexpected_predictions:
        raise ValueError(
            "Dataset ID mismatch. "
            f"Missing predictions: {sorted(missing_predictions)}. "
            f"Unexpected predictions: {sorted(unexpected_predictions)}."
        )

    samples = []

    for sample_id, golden in golden_records.items():
        prediction = prediction_records[sample_id]

        samples.append({
            "id": sample_id,
            "expected_entities": golden["expected_entities"],
            "predicted_entities": prediction["predicted_entities"],
            "expected_relations": golden["expected_relations"],
            "predicted_relations": prediction["predicted_relations"],
        })

    return samples


def create_report(
    results: dict[str, Any],
) -> dict[str, Any]:
    """Create a timestamped evaluation report."""

    return {
        "project": "llm-kg-eval-harness",
        "evaluation_type": "synthetic_demonstration",
        "prediction_source": "evals/sample_predictions.json",
        "evaluation_timestamp": datetime.now(
            UTC
        ).isoformat(),
        "sample_count": results["sample_count"],
        "metrics": {
            "entities": results["entities"],
            "relations": results["relations"],
            "overall": results["overall"],
        },
        "per_sample_results": results["per_sample_results"],
        "notes": (
            "Predictions are synthetic demonstration data. "
            "Metrics do not represent actual LLM performance."
        ),
    }


def print_summary(report: dict[str, Any]) -> None:
    """Print a readable evaluation summary."""

    metrics = report["metrics"]

    print("\n" + "=" * 60)
    print("       LLM KNOWLEDGE GRAPH EVALUATION REPORT")
    print("=" * 60)

    print(f"\nSamples evaluated: {report['sample_count']}")

    for category in ("entities", "relations", "overall"):
        result = metrics[category]

        print(f"\n{category.upper()} METRICS")
        print("-" * 40)

        print(f"True Positives : {result['true_positives']}")
        print(f"False Positives: {result['false_positives']}")
        print(f"False Negatives: {result['false_negatives']}")
        print(f"Precision      : {result['precision']:.4f}")
        print(f"Recall         : {result['recall']:.4f}")
        print(f"F1 Score       : {result['f1_score']:.4f}")

    print("\n" + "=" * 60)


def run_evaluation() -> dict[str, Any]:
    """Execute the complete evaluation workflow."""

    logger.info("Starting evaluation pipeline.")

    golden_data = load_json(GOLDEN_DATASET_PATH)
    prediction_data = load_json(PREDICTIONS_PATH)

    golden_records = validate_dataset(
        golden_data,
        "Golden dataset",
        {"id", "text", "expected_entities", "expected_relations"},
    )

    prediction_records = validate_dataset(
        prediction_data,
        "Prediction dataset",
        {"id", "predicted_entities", "predicted_relations"},
    )

    samples = build_evaluation_samples(
        golden_records,
        prediction_records,
    )

    engine = EvaluationEngine()

    results = engine.evaluate_dataset(samples)

    report = create_report(results)

    REPORT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print_summary(report)

    logger.info("Evaluation report saved to %s", REPORT_PATH)
    logger.info("Evaluation completed successfully.")

    return report


def main() -> None:
    """CLI entry point."""

    try:
        run_evaluation()

    except (FileNotFoundError, ValueError, TypeError) as error:
        logger.error("Evaluation failed: %s", error)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
