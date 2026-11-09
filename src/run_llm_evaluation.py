
"""Run real LLM extraction evaluation against the golden dataset."""

from __future__ import annotations

import argparse
import json
import logging
import time

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.evaluator import EvaluationEngine
from src.extractor import LLMEntityExtractor


PROJECT_ROOT = Path(__file__).resolve().parent.parent

GOLDEN_DATASET_PATH = (
    PROJECT_ROOT / "evals" / "golden_dataset.json"
)

REPORT_DIRECTORY = PROJECT_ROOT / "evals" / "reports"

REPORT_PATH = (
    REPORT_DIRECTORY / "llm_evaluation_report.json"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def load_golden_dataset() -> list[dict[str, Any]]:
    """Load and validate the golden evaluation dataset."""

    if not GOLDEN_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Golden dataset not found: {GOLDEN_DATASET_PATH}"
        )

    try:
        with GOLDEN_DATASET_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:
            dataset = json.load(file)

    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid golden dataset JSON: {exc}"
        ) from exc

    if not isinstance(dataset, list):
        raise TypeError(
            "Golden dataset must contain a JSON array."
        )

    seen_ids = set()

    required_fields = {
        "id",
        "text",
        "expected_entities",
        "expected_relations",
    }

    for index, sample in enumerate(dataset):
        if not isinstance(sample, dict):
            raise TypeError(
                f"Sample at index {index} must be an object."
            )

        missing = required_fields - sample.keys()

        if missing:
            raise ValueError(
                f"Sample at index {index} is missing: "
                f"{sorted(missing)}"
            )

        sample_id = sample["id"]

        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError(
                f"Sample at index {index} has an invalid ID."
            )

        if sample_id in seen_ids:
            raise ValueError(
                f"Duplicate sample ID: {sample_id}"
            )

        seen_ids.add(sample_id)

        if not isinstance(sample["text"], str) or not sample["text"].strip():
            raise ValueError(
                f"Sample {sample_id} has empty text."
            )

        for field in ("expected_entities", "expected_relations"):
            values = sample[field]

            if not isinstance(values, list):
                raise TypeError(
                    f"{field} in {sample_id} must be a list."
                )

            if not all(
                isinstance(value, str) for value in values
            ):
                raise TypeError(
                    f"Every item in {field} for {sample_id} "
                    "must be a string."
                )

    if not dataset:
        raise ValueError("Golden dataset is empty.")

    return dataset


def convert_extraction_to_predictions(
    extraction: Any,
) -> tuple[list[str], list[str]]:
    """Convert the Pydantic extraction result into evaluator inputs."""

    predicted_entities = [
        entity.name
        for entity in extraction.entities
    ]

    predicted_relations = [
        (
            f"{relation.source_id}|"
            f"{relation.relation_type}|"
            f"{relation.target_id}"
        )
        for relation in extraction.relations
    ]

    return predicted_entities, predicted_relations


def evaluate_real_predictions(
    dataset: list[dict[str, Any]],
    extractor: LLMEntityExtractor,
) -> dict[str, Any]:
    """Extract real predictions and evaluate them against ground truth."""

    engine = EvaluationEngine()

    evaluation_samples = []
    sample_details = []

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    total_latency_seconds = 0.0

    successful_samples = 0
    failed_samples = 0

    for index, golden in enumerate(dataset, start=1):
        sample_id = golden["id"]

        logger.info(
            "Processing sample %d/%d: %s",
            index,
            len(dataset),
            sample_id,
        )

        document = {
            "id": sample_id,
            "content": golden["text"],
        }

        try:
            start_time = time.perf_counter()

            extraction = extractor.extract(document)

            wall_latency = time.perf_counter() - start_time

            usage = extractor.last_usage

            predicted_entities, predicted_relations = (
                convert_extraction_to_predictions(extraction)
            )

            evaluation_samples.append({
                "id": sample_id,
                "expected_entities": golden["expected_entities"],
                "predicted_entities": predicted_entities,
                "expected_relations": golden["expected_relations"],
                "predicted_relations": predicted_relations,
            })

            prompt_tokens = usage["prompt_tokens"]
            completion_tokens = usage["completion_tokens"]
            sample_tokens = usage["total_tokens"]

            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
            total_tokens += sample_tokens

            # Measure the complete extractor call, including validation.
            total_latency_seconds += wall_latency

            successful_samples += 1

            sample_details.append({
                "sample_id": sample_id,
                "status": "success",
                "predicted_entities": predicted_entities,
                "predicted_relations": predicted_relations,
                "token_usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": sample_tokens,
                },
                "latency_seconds": round(wall_latency, 4),
            })

            logger.info(
                "%s: %d entities, %d relations, %d tokens, %.2f seconds",
                sample_id,
                len(predicted_entities),
                len(predicted_relations),
                sample_tokens,
                wall_latency,
            )

        except Exception as exc:
            failed_samples += 1

            sample_details.append({
                "sample_id": sample_id,
                "status": "failed",
                "error": str(exc),
            })

            logger.exception(
                "Evaluation extraction failed for %s.",
                sample_id,
            )

    if not evaluation_samples:
        raise RuntimeError(
            "No samples were successfully extracted. "
            "Cannot calculate evaluation metrics."
        )

    metrics = engine.evaluate_dataset(evaluation_samples)

    average_latency = (
        total_latency_seconds / successful_samples
    )

    return {
        "sample_count": len(dataset),
        "successful_samples": successful_samples,
        "failed_samples": failed_samples,
        "metrics": {
            "entities": metrics["entities"],
            "relations": metrics["relations"],
            "overall": metrics["overall"],
        },
        "performance": {
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
            "total_latency_seconds": round(
                total_latency_seconds, 4
            ),
            "average_latency_seconds": round(
                average_latency, 4
            ),
        },
        "per_sample_results": metrics["per_sample_results"],
        "sample_execution_details": sample_details,
    }


def create_report(
    results: dict[str, Any],
    model_name: str,
) -> dict[str, Any]:
    """Create the real LLM evaluation report."""

    return {
        "project": "llm-kg-eval-harness",
        "evaluation_type": "real_llm_extraction",
        "prediction_source": "OpenAI API via LLMEntityExtractor",
        "model": model_name,
        "evaluation_timestamp": datetime.now(UTC).isoformat(),
        **results,
        "notes": (
            "Metrics are calculated using actual LLM extraction "
            "predictions against the golden dataset. "
            "Failed samples are excluded from metric calculations "
            "and reported separately."
        ),
    }


def print_summary(report: dict[str, Any]) -> None:
    """Print a readable evaluation summary."""

    metrics = report["metrics"]
    performance = report["performance"]

    print("\n" + "=" * 60)
    print("          REAL LLM EVALUATION REPORT")
    print("=" * 60)

    print(f"\nSamples attempted : {report['sample_count']}")
    print(f"Successful samples: {report['successful_samples']}")
    print(f"Failed samples    : {report['failed_samples']}")

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

    print("\nPERFORMANCE")
    print("-" * 40)

    print(
        f"Prompt tokens     : "
        f"{performance['total_prompt_tokens']}"
    )
    print(
        f"Completion tokens : "
        f"{performance['total_completion_tokens']}"
    )
    print(
        f"Total tokens      : "
        f"{performance['total_tokens']}"
    )
    print(
        f"Total latency     : "
        f"{performance['total_latency_seconds']:.4f}s"
    )
    print(
        f"Average latency   : "
        f"{performance['average_latency_seconds']:.4f}s"
    )

    print("\n" + "=" * 60)


def run_evaluation(limit: int | None = None) -> dict[str, Any]:
    """Run the real LLM evaluation workflow."""

    logger.info("Starting real LLM evaluation.")

    dataset = load_golden_dataset()

    if limit is not None:
        if limit < 1:
            raise ValueError("Limit must be at least 1.")

        dataset = dataset[:limit]

    logger.info(
        "Loaded %d evaluation samples.",
        len(dataset),
    )

    extractor = LLMEntityExtractor()

    results = evaluate_real_predictions(
        dataset=dataset,
        extractor=extractor,
    )

    report = create_report(
        results,
        model_name=extractor.model,
    )

    REPORT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print_summary(report)

    logger.info(
        "Real LLM evaluation report saved to %s",
        REPORT_PATH,
    )

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run real LLM extraction evaluation."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of golden samples to evaluate.",
    )

    args = parser.parse_args()

    try:
        run_evaluation(limit=args.limit)

    except (FileNotFoundError, ValueError, TypeError, RuntimeError) as exc:
        logger.error("Real LLM evaluation failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
