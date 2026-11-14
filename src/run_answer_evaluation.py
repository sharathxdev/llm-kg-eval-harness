"""Run automated GraphRAG answer-quality evaluation."""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

from src.answer_evaluator import LLMJudgeEvaluator
from src.config import get_openai_model
from src.pipeline import GraphRAGPipeline
from src.retriever import (
    HybridGraphRetriever,
    TfidfEmbeddingProvider,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DATASET_PATH = PROJECT_ROOT / "evals" / "retrieval_golden_dataset.json"

DEFAULT_REPORT_PATH = PROJECT_ROOT / "evals" / "reports" / "answer_quality_report.json"

DOCUMENTS_PATH = PROJECT_ROOT / "data" / "documents.json"


def load_golden_dataset(
    path: Path = DEFAULT_DATASET_PATH,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Load and validate the existing retrieval golden dataset."""

    if limit is not None and (
        not isinstance(limit, int) or isinstance(limit, bool) or limit < 1
    ):
        raise ValueError("Limit must be a positive integer.")

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            dataset = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid dataset JSON: {exc}") from exc

    if not isinstance(dataset, list) or not dataset:
        raise ValueError("Dataset must be a non-empty JSON array.")

    required_fields = {
        "id",
        "query",
        "expected_doc_ids",
        "expected_graph_entities",
    }

    seen_ids: set[str] = set()

    for index, sample in enumerate(dataset):
        if not isinstance(sample, dict):
            raise TypeError(f"Sample at index {index} must be an object.")

        missing = required_fields - sample.keys()

        if missing:
            raise ValueError(
                f"Sample at index {index} is missing fields: {sorted(missing)}"
            )

        sample_id = sample["id"]

        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError(f"Sample at index {index} has an invalid ID.")

        if sample_id in seen_ids:
            raise ValueError(f"Duplicate sample ID: {sample_id}")

        seen_ids.add(sample_id)

        if not isinstance(sample["query"], str) or not sample["query"].strip():
            raise ValueError(f"Invalid query in {sample_id}.")

        for field in (
            "expected_doc_ids",
            "expected_graph_entities",
        ):
            values = sample[field]

            if not isinstance(values, list):
                raise TypeError(f"{field} in {sample_id} must be a list.")

            if not all(isinstance(value, str) and value.strip() for value in values):
                raise ValueError(f"{field} in {sample_id} contains invalid values.")

    return dataset[:limit] if limit is not None else dataset


def format_evidence_context(
    retrieval_result: dict[str, Any],
) -> str:
    """Combine retrieved documents and graph facts into judge context."""

    sections = []

    chunks = retrieval_result.get("retrieved_chunks", [])

    if not isinstance(chunks, list):
        raise TypeError("retrieved_chunks must be a list.")

    document_lines = []

    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue

        document_lines.append(
            "\n".join(
                [
                    f"Document ID: {chunk.get('id', 'UNKNOWN')}",
                    f"Title: {chunk.get('title', '')}",
                    f"Category: {chunk.get('category', '')}",
                    f"Content: {chunk.get('content', '')}",
                ]
            )
        )

    sections.append(
        "[Retrieved Documents]\n"
        + (
            "\n\n".join(document_lines)
            if document_lines
            else "No relevant documents retrieved."
        )
    )

    graph_context = retrieval_result.get("graph_context", [])

    if not isinstance(graph_context, list):
        raise TypeError("graph_context must be a list.")

    graph_lines = [str(item) for item in graph_context if item]

    sections.append(
        "[Knowledge Graph Relations]\n"
        + (
            "\n".join(graph_lines)
            if graph_lines
            else "No graph relationships retrieved."
        )
    )

    return "\n\n".join(sections)


def _average(values: list[float]) -> float | None:
    """Calculate a rounded mean or return None for empty input."""

    if not values:
        return None

    return round(mean(values), 4)


def evaluate_answers(
    dataset: list[dict[str, Any]],
    pipeline: Any,
    evaluator: Any,
    top_k: int = 3,
    hops: int = 2,
) -> dict[str, Any]:
    """Generate and judge answers for every golden dataset sample."""

    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")

    if not isinstance(hops, int) or isinstance(hops, bool) or hops < 0:
        raise ValueError("hops must be a non-negative integer.")

    if not dataset:
        raise ValueError("Evaluation dataset cannot be empty.")

    sample_results = []

    faithfulness_scores = []
    relevance_scores = []
    overall_scores = []
    generation_latencies = []

    successful_samples = 0
    failed_samples = 0

    for index, sample in enumerate(dataset, start=1):
        sample_id = sample["id"]

        logger.info(
            "Evaluating answer %d/%d: %s",
            index,
            len(dataset),
            sample_id,
        )

        start_time = time.perf_counter()

        try:
            generated = pipeline.run(
                query=sample["query"],
                top_k=top_k,
                hops=hops,
            )

            if not isinstance(generated, dict):
                raise TypeError("Pipeline result must be a dictionary.")

            answer = generated.get("answer")

            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("Pipeline returned an empty or invalid answer.")

            context = format_evidence_context(generated)

            quality = evaluator.evaluate_answer(
                query=sample["query"],
                answer=answer,
                context=context,
            )

            if hasattr(quality, "model_dump"):
                quality_data = quality.model_dump(mode="json")
            elif isinstance(quality, dict):
                quality_data = quality
            else:
                raise TypeError("Evaluator must return a Pydantic model or dictionary.")

            faithfulness = quality_data["faithfulness"]["score"]
            relevance = quality_data["answer_relevance"]["score"]
            overall = quality_data["overall_score"]

            for value in (faithfulness, relevance, overall):
                if (
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not 1 <= value <= 5
                ):
                    raise ValueError("Evaluator returned an invalid quality score.")

            latency = generated.get("latency_ms")

            if (
                not isinstance(latency, (int, float))
                or isinstance(latency, bool)
                or latency < 0
            ):
                latency = (time.perf_counter() - start_time) * 1000

            generation_latency = round(float(latency), 3)

            sample_results.append(
                {
                    "sample_id": sample_id,
                    "status": "success",
                    "query": sample["query"],
                    "answer": answer,
                    "evaluation_context": context,
                    "expected_doc_ids": sample["expected_doc_ids"],
                    "retrieved_doc_ids": [
                        chunk.get("id")
                        for chunk in generated.get("retrieved_chunks", [])
                        if isinstance(chunk, dict)
                    ],
                    "expected_graph_entities": sample["expected_graph_entities"],
                    "graph_entities_used": generated.get("graph_entities_used", []),
                    "quality_evaluation": quality_data,
                    "generation_latency_ms": generation_latency,
                }
            )

            faithfulness_scores.append(float(faithfulness))
            relevance_scores.append(float(relevance))
            overall_scores.append(float(overall))
            generation_latencies.append(generation_latency)

            successful_samples += 1

        except Exception as exc:
            failed_samples += 1

            logger.exception(
                "Answer evaluation failed for %s.",
                sample_id,
            )

            sample_results.append(
                {
                    "sample_id": sample_id,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    return {
        "sample_count": len(dataset),
        "successful_samples": successful_samples,
        "failed_samples": failed_samples,
        "summary": {
            "average_faithfulness": _average(faithfulness_scores),
            "average_answer_relevance": _average(relevance_scores),
            "average_overall_score": _average(overall_scores),
            "average_generation_latency_ms": _average(generation_latencies),
        },
        "per_sample_results": sample_results,
    }


def _create_pipeline(retrieval_method: str) -> GraphRAGPipeline:
    """Build the existing GraphRAG pipeline with the selected retriever."""

    if retrieval_method not in {"tfidf", "bm25", "rrf"}:
        raise ValueError("retrieval_method must be tfidf, bm25, or rrf.")

    embedding_provider = (
        TfidfEmbeddingProvider() if retrieval_method in {"tfidf", "rrf"} else None
    )

    retriever = HybridGraphRetriever(
        documents_path=DOCUMENTS_PATH,
        embedding_provider=embedding_provider,
        retrieval_method=retrieval_method,
    )

    return GraphRAGPipeline(retriever=retriever)


def run_answer_evaluation(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
    top_k: int = 3,
    hops: int = 2,
    retrieval_method: str = "tfidf",
    judge_mode: str = "llm",
    model: str | None = None,
    limit: int | None = None,
    pipeline: Any = None,
    evaluator: Any = None,
) -> dict[str, Any]:
    """Execute answer evaluation and persist a JSON report."""

    if judge_mode not in {"llm", "mock"}:
        raise ValueError("judge_mode must be 'llm' or 'mock'.")

    if retrieval_method not in {"tfidf", "bm25", "rrf"}:
        raise ValueError("retrieval_method must be 'tfidf', 'bm25', or 'rrf'.")

    dataset = load_golden_dataset(
        path=dataset_path,
        limit=limit,
    )

    active_pipeline = (
        pipeline if pipeline is not None else _create_pipeline(retrieval_method)
    )

    active_evaluator = (
        evaluator
        if evaluator is not None
        else LLMJudgeEvaluator(
            model=model,
            mock_mode=(judge_mode == "mock"),
        )
    )

    results = evaluate_answers(
        dataset=dataset,
        pipeline=active_pipeline,
        evaluator=active_evaluator,
        top_k=top_k,
        hops=hops,
    )

    report = {
        "metadata": {
            "evaluation_type": "GraphRAG answer quality",
            "generated_at": datetime.now(UTC).isoformat(),
            "retrieval_method": retrieval_method,
            "judge_mode": judge_mode,
            "judge_model": (None if judge_mode == "mock" else get_openai_model(model)),
            "top_k": top_k,
            "hops": hops,
            "dataset": str(dataset_path),
        },
        **results,
    }

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    logger.info("Answer evaluation report saved to %s", report_path)

    return report


def print_summary(report: dict[str, Any]) -> None:
    """Print a human-readable evaluation summary."""

    summary = report["summary"]

    print("\n========== ANSWER QUALITY EVALUATION ==========")
    print(f"Total samples:       {report['sample_count']}")
    print(f"Successful samples:  {report['successful_samples']}")
    print(f"Failed samples:      {report['failed_samples']}")
    print("-----------------------------------------------")
    print(f"Faithfulness:        {summary['average_faithfulness']}")
    print(f"Answer relevance:    {summary['average_answer_relevance']}")
    print(f"Overall score:       {summary['average_overall_score']}")
    print(f"Generation latency:  {summary['average_generation_latency_ms']} ms")
    print("================================================")


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(
        description="Evaluate GraphRAG generated answer quality."
    )

    parser.add_argument(
        "--judge-mode",
        choices=["llm", "mock"],
        default="llm",
        help="LLM uses OpenAI for judging; mock uses lexical heuristics.",
    )

    parser.add_argument(
        "--retrieval-method",
        choices=["tfidf", "bm25", "rrf"],
        default="tfidf",
    )

    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--hops", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", default=None)

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_REPORT_PATH,
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    try:
        report = run_answer_evaluation(
            dataset_path=args.dataset,
            report_path=args.output,
            top_k=args.top_k,
            hops=args.hops,
            retrieval_method=args.retrieval_method,
            judge_mode=args.judge_mode,
            model=args.model,
            limit=args.limit,
        )

        print_summary(report)
        print(f"\nReport saved: {args.output}")

        if report["successful_samples"] == 0:
            raise SystemExit(1)

    except (ValueError, TypeError, FileNotFoundError, RuntimeError) as exc:
        logger.error("Answer evaluation failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
