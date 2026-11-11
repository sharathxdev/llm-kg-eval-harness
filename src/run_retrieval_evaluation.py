"""Run deterministic Knowledge Graph RAG retrieval evaluation."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from src.retrieval_evaluator import RetrievalEvaluationEngine
from src.retriever import HybridGraphRetriever, TfidfEmbeddingProvider

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_DATASET_PATH = PROJECT_ROOT / "evals" / "retrieval_golden_dataset.json"
REPORT_PATH = (
    PROJECT_ROOT
    / "evals"
    / "reports"
    / "retrieval_evaluation_report.json"
)
DOCUMENTS_PATH = PROJECT_ROOT / "data" / "documents.json"


def load_golden_dataset(path: Path = GOLDEN_DATASET_PATH) -> list[dict[str, Any]]:
    """Load and validate the retrieval golden dataset."""

    if not path.exists():
        raise FileNotFoundError(f"Golden dataset not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    if not isinstance(dataset, list) or not dataset:
        raise ValueError("Golden dataset must be a non-empty JSON list.")

    seen_ids = set()

    for index, sample in enumerate(dataset):
        if not isinstance(sample, dict):
            raise TypeError(f"Sample at index {index} must be an object.")

        required_fields = [
            "id",
            "query",
            "expected_doc_ids",
            "expected_graph_entities",
        ]

        missing_fields = [
            field for field in required_fields if field not in sample
        ]

        if missing_fields:
            raise ValueError(
                f"Sample at index {index} is missing fields: {missing_fields}"
            )

        sample_id = sample["id"]

        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError(f"Sample at index {index} has an invalid ID.")

        if sample_id in seen_ids:
            raise ValueError(f"Duplicate golden sample ID: {sample_id}")

        seen_ids.add(sample_id)

        if not isinstance(sample["query"], str) or not sample["query"].strip():
            raise ValueError(f"Sample {sample_id} has an invalid query.")

        for field in ["expected_doc_ids", "expected_graph_entities"]:
            if not isinstance(sample[field], list):
                raise TypeError(
                    f"{field} in sample {sample_id} must be a list."
                )

            if not sample[field]:
                raise ValueError(
                    f"{field} in sample {sample_id} cannot be empty."
                )

    return dataset


def display_path(path: Path) -> str:
    """Return a project-relative path when possible, otherwise an absolute path."""

    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def run_retrieval_evaluation(
    k: int = 3,
    hops: int = 2,
    retrieval_method: str = "tfidf",
) -> dict[str, Any]:
    """Run retrieval evaluation and save the resulting report."""

    if not isinstance(hops, int) or isinstance(hops, bool) or hops < 0:
        raise ValueError("Hops must be a non-negative integer.")

    if retrieval_method not in {"tfidf", "bm25"}:
        raise ValueError(
            "retrieval_method must be either 'tfidf' or 'bm25'."
        )

    golden_dataset = load_golden_dataset()

    embedding_provider = (
        TfidfEmbeddingProvider()
        if retrieval_method == "tfidf"
        else None
    )

    retriever = HybridGraphRetriever(
        documents_path=DOCUMENTS_PATH,
        embedding_provider=embedding_provider,
        retrieval_method=retrieval_method,
    )

    evaluator = RetrievalEvaluationEngine(k=k)

    evaluation_samples = []
    retrieval_evidence = []

    for sample in golden_dataset:
        retrieval_result = retriever.retrieve(
            sample["query"],
            top_k=k,
            hops=hops,
        )

        retrieved_doc_ids = [
            chunk["id"]
            for chunk in retrieval_result["retrieved_chunks"]
        ]

        retrieved_graph_entities = [
            entity["name"]
            for entity in retrieval_result["graph_entities"]
        ]

        expected_doc_ids = sample["expected_doc_ids"]
        expected_graph_entities = sample["expected_graph_entities"]

        expected_doc_set = set(expected_doc_ids)
        retrieved_doc_set = set(retrieved_doc_ids)

        expected_entity_set = {
            name.casefold() for name in expected_graph_entities
        }
        retrieved_entity_set = {
            name.casefold() for name in retrieved_graph_entities
        }

        evaluation_samples.append({
            "id": sample["id"],
            "expected_doc_ids": expected_doc_ids,
            "retrieved_doc_ids": retrieved_doc_ids,
            "expected_graph_entities": expected_graph_entities,
            "retrieved_graph_entities": retrieved_graph_entities,
        })

        retrieval_evidence.append({
            "query_id": sample["id"],
            "query": sample["query"],
            "expected_doc_ids": expected_doc_ids,
            "retrieved_doc_ids": retrieved_doc_ids,
            "relevant_retrieved_doc_ids": [
                doc_id for doc_id in retrieved_doc_ids
                if doc_id in expected_doc_set
            ],
            "missing_expected_doc_ids": [
                doc_id for doc_id in expected_doc_ids
                if doc_id not in retrieved_doc_set
            ],
            "expected_graph_entities": expected_graph_entities,
            "retrieved_graph_entities": retrieved_graph_entities,
            "missing_graph_entities": [
                name for name in expected_graph_entities
                if name.casefold() not in retrieved_entity_set
            ],
            "unexpected_graph_entities": [
                name for name in retrieved_graph_entities
                if name.casefold() not in expected_entity_set
            ],
        })

    report = evaluator.evaluate_dataset(evaluation_samples)
    report["retrieval_evidence"] = retrieval_evidence

    report["metadata"] = {
        "evaluation_type": "Knowledge Graph RAG retrieval",
        "retrieval_method": (
            "TF-IDF" if retrieval_method == "tfidf" else "BM25"
        ),
        "embedding_provider": (
            "TF-IDF" if retrieval_method == "tfidf" else None
        ),
        "golden_dataset": display_path(GOLDEN_DATASET_PATH),
        "documents_path": display_path(DOCUMENTS_PATH),
        "document_count": len(retriever.documents),
        "graph_entity_count": len(retriever.entities),
        "graph_relationship_count": retriever.graph_engine.graph.number_of_edges(),
        "hops": hops,
    }

    report_suffix = (
        f"_bm25_k{k}" if retrieval_method == "bm25" else f"_k{k}"
    )

    report_path = REPORT_PATH.with_name(
        f"{REPORT_PATH.stem}{report_suffix}{REPORT_PATH.suffix}"
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)

    logger.info("Retrieval evaluation report saved to %s", report_path)

    print("\n========== RETRIEVAL EVALUATION ==========")
    print(f"Queries evaluated: {report['configuration']['query_count']}")
    print(f"Documents: {report['metadata']['document_count']}")
    print(f"Graph entities: {report['metadata']['graph_entity_count']}")
    print(f"Graph relationships: {report['metadata']['graph_relationship_count']}")
    print(f"Retrieval method: {report['metadata']['retrieval_method']}")
    print(f"K: {k}")
    print(f"Hops: {hops}")

    print("\nDocument Retrieval Metrics:")
    for metric, value in report["summary"]["document_retrieval"].items():
        print(f"  {metric}: {value}")

    print("\nGraph Entity Metrics:")
    for metric, value in report["summary"]["graph_entities"].items():
        print(f"  {metric}: {value}")

    print(f"\nReport saved: {display_path(report_path)}")
    print("==========================================\n")

    return report


def main() -> None:
    """Parse command-line arguments and run evaluation."""

    parser = argparse.ArgumentParser(
        description="Evaluate Knowledge Graph RAG retrieval quality."
    )

    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Number of top documents to evaluate (default: 3).",
    )

    parser.add_argument(
        "--hops",
        type=int,
        default=2,
        help="Knowledge graph traversal depth (default: 2).",
    )

    parser.add_argument(
        "--retrieval-method",
        choices=["tfidf", "bm25"],
        default="tfidf",
        help="Document retrieval method (default: tfidf).",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    run_retrieval_evaluation(
        k=args.k,
        hops=args.hops,
        retrieval_method=args.retrieval_method,
    )


if __name__ == "__main__":
    main()
