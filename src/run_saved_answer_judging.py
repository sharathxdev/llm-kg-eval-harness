"""Run real LLM judging against previously generated GraphRAG answers."""

from __future__ import annotations

import argparse
import json
import logging
import time
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

from src.answer_evaluator import LLMJudgeEvaluator
from src.config import get_openai_model
from src.run_answer_evaluation import (
    DEFAULT_REPORT_PATH,
    _create_pipeline,
    format_evidence_context,
)

logger = logging.getLogger(__name__)


def _average(values: list[float]) -> float | None:
    if not values:
        return None
    return round(mean(values), 4)


def load_answer_report(path: Path) -> dict[str, Any]:
    """Load and validate a previous answer-quality report."""

    if not path.is_file():
        raise FileNotFoundError(f"Answer report not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            report = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid answer report JSON.") from exc

    if not isinstance(report, dict):
        raise TypeError("Answer report must be a JSON object.")

    metadata = report.get("metadata")

    if metadata is None:
        raise ValueError("Answer report metadata is missing.")

    if not isinstance(metadata, dict):
        raise TypeError("Answer report metadata must be a JSON object.")

    samples = report.get("per_sample_results")

    if not isinstance(samples, list):
        raise TypeError("per_sample_results must be a list.")

    successful = [
        sample
        for sample in samples
        if isinstance(sample, dict) and sample.get("status") == "success"
    ]

    if not successful:
        raise ValueError("No successfully generated answers to judge.")

    for sample in successful:
        if not isinstance(sample.get("query"), str):
            raise TypeError("A saved sample query must be a string.")

        if not sample["query"].strip():
            raise ValueError("A saved sample query cannot be empty.")

        if not isinstance(sample.get("answer"), str):
            raise TypeError("A saved sample answer must be a string.")

        if not sample["answer"].strip():
            raise ValueError("A saved sample answer cannot be empty.")

    return report


def judge_saved_report(
    input_path: Path = DEFAULT_REPORT_PATH,
    output_path: Path = (Path("evals/reports/answer_quality_llm_judged.json")),
    model: str | None = None,
    retriever: Any = None,
    evaluator: Any = None,
) -> dict[str, Any]:
    """
    Re-evaluate previously generated answers without calling the generator.

    Only retrieval and the two judge dimensions run again.
    """

    input_path = Path(input_path)
    output_path = Path(output_path)

    if input_path.resolve() == output_path.resolve():
        raise ValueError(
            "Input and output paths must differ. The original report must be preserved."
        )

    original = load_answer_report(input_path)
    report = deepcopy(original)
    metadata = report["metadata"]

    retrieval_method = metadata.get("retrieval_method", "tfidf")
    top_k = metadata.get("top_k", 3)
    hops = metadata.get("hops", 2)

    if retrieval_method not in {"tfidf", "bm25", "rrf"}:
        raise ValueError("Saved report has an invalid retrieval method.")

    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("Saved report has an invalid top_k.")

    if not isinstance(hops, int) or isinstance(hops, bool) or hops < 0:
        raise ValueError("Saved report has invalid hops.")

    successful_samples = [
        sample
        for sample in report["per_sample_results"]
        if isinstance(sample, dict) and sample.get("status") == "success"
    ]

    needs_retrieval = any(
        not isinstance(sample.get("evaluation_context"), str)
        for sample in successful_samples
    )

    # Maintain backward compatibility with older reports.
    # New reports reuse their original generation evidence.
    if retriever is None and needs_retrieval:
        retriever = _create_pipeline(retrieval_method).retriever

    if evaluator is None:
        evaluator = LLMJudgeEvaluator(
            model=model,
            mock_mode=False,
        )

    faithfulness_scores = []
    relevance_scores = []
    overall_scores = []
    judge_latencies = []

    judge_successful = 0
    judge_failed = 0

    for sample in report["per_sample_results"]:
        if sample.get("status") != "success":
            sample["judge_status"] = "skipped"
            continue

        logger.info(
            "Judging saved answer: %s",
            sample.get("sample_id"),
        )

        start_time = time.perf_counter()

        try:
            context = sample.get("evaluation_context")

            if not isinstance(context, str):
                if retriever is None:
                    raise RuntimeError(
                        "No saved evidence context or retriever available."
                    )

                retrieval = retriever.retrieve(
                    query=sample["query"],
                    top_k=top_k,
                    hops=hops,
                )

                context = format_evidence_context(retrieval)

            quality = evaluator.evaluate_answer(
                query=sample["query"],
                answer=sample["answer"],
                context=context,
            )

            if hasattr(quality, "model_dump"):
                quality_data = quality.model_dump(mode="json")
            elif isinstance(quality, dict):
                quality_data = quality
            else:
                raise TypeError("Invalid judge response type.")

            faithfulness = float(quality_data["faithfulness"]["score"])
            relevance = float(quality_data["answer_relevance"]["score"])
            overall = float(quality_data["overall_score"])

            for score in (faithfulness, relevance, overall):
                if not 1 <= score <= 5:
                    raise ValueError("Judge returned an invalid score.")

            latency = (time.perf_counter() - start_time) * 1000

            sample["quality_evaluation"] = quality_data
            sample["judge_status"] = "success"
            sample["judge_latency_ms"] = round(latency, 3)
            sample.pop("judge_error", None)

            faithfulness_scores.append(faithfulness)
            relevance_scores.append(relevance)
            overall_scores.append(overall)
            judge_latencies.append(latency)

            judge_successful += 1

        except Exception as exc:
            judge_failed += 1

            sample["judge_status"] = "failed"
            sample["judge_error"] = str(exc)

            logger.exception(
                "Judge evaluation failed for %s.",
                sample.get("sample_id"),
            )

    metadata["judge_mode"] = "llm"
    metadata["judge_model"] = get_openai_model(model)
    metadata["judged_at"] = datetime.now(UTC).isoformat()

    report["judge_successful_samples"] = judge_successful
    report["judge_failed_samples"] = judge_failed

    report["summary"].update(
        {
            "average_faithfulness": _average(faithfulness_scores),
            "average_answer_relevance": _average(relevance_scores),
            "average_overall_score": _average(overall_scores),
            "average_judge_latency_ms": _average(judge_latencies),
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    logger.info("Real judge report saved to %s", output_path)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Judge saved GraphRAG answers using an LLM."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_REPORT_PATH,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evals/reports/answer_quality_llm_judged.json"),
    )

    parser.add_argument("--model", default=None)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    report = judge_saved_report(
        input_path=args.input,
        output_path=args.output,
        model=args.model,
    )

    print("\n========== REAL LLM JUDGE RESULTS ==========")
    print("Generation calls: 0 (reused saved answers)")
    print("Judge successful:", report["judge_successful_samples"])
    print("Judge failed:", report["judge_failed_samples"])

    for name, value in report["summary"].items():
        print(f"{name}: {value}")

    print("\nReport:", args.output)

    if report["judge_successful_samples"] == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
