"""Compare TF-IDF and BM25 retrieval evaluation reports."""

import json
import logging
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "evals" / "reports"
COMPARISON_REPORT_PATH = (
    REPORTS_DIR / "retrieval_comparison_report.json"
)

BENCHMARK_REPORTS = {
    "TF-IDF": {
        3: "retrieval_evaluation_report_k3.json",
        5: "retrieval_evaluation_report_k5.json",
        10: "retrieval_evaluation_report_k10.json",
    },
    "BM25": {
        3: "retrieval_evaluation_report_bm25_k3.json",
        5: "retrieval_evaluation_report_bm25_k5.json",
        10: "retrieval_evaluation_report_bm25_k10.json",
    },
}

METRICS = (
    "precision",
    "recall",
    "hit_rate",
    "mrr",
    "ndcg",
)


def load_report(path: Path) -> dict[str, Any]:
    """Load and validate one retrieval evaluation report."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Benchmark report not found: {path}"
        )

    with path.open(encoding="utf-8") as file:
        report = json.load(file)

    if not isinstance(report, dict):
        raise ValueError(
            f"Report must contain a JSON object: {path.name}"
        )

    summary = report.get("summary")
    if not isinstance(summary, dict):
        raise ValueError(
            f"Missing or invalid summary in {path.name}"
        )

    document_metrics = summary.get("document_retrieval")
    if not isinstance(document_metrics, dict):
        raise ValueError(
            f"Missing document retrieval metrics in {path.name}"
        )

    graph_metrics = summary.get("graph_entities")
    if not isinstance(graph_metrics, dict):
        raise ValueError(
            f"Missing graph entity metrics in {path.name}"
        )

    return report


def extract_metrics(
    report: dict[str, Any],
    k: int,
) -> dict[str, float]:
    """Extract ranking and graph entity metrics from a report."""

    document_metrics = report["summary"]["document_retrieval"]
    graph_metrics = report["summary"]["graph_entities"]

    extracted = {}

    for metric in METRICS:
        key = f"{metric}_at_{k}"

        if key not in document_metrics:
            raise ValueError(
                f"Missing metric '{key}' in retrieval report."
            )

        value = document_metrics[key]

        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
        ):
            raise ValueError(
                f"Metric '{key}' must be numeric."
            )

        extracted[key] = float(value)

    graph_f1 = graph_metrics.get("f1_score")

    if (
        isinstance(graph_f1, bool)
        or not isinstance(graph_f1, (int, float))
    ):
        raise ValueError(
            "Graph entity f1_score must be numeric."
        )

    extracted["graph_entity_f1"] = float(graph_f1)

    return extracted


def build_comparison() -> dict[str, Any]:
    """Compare all configured TF-IDF and BM25 benchmark reports."""

    results = {}

    for method, benchmarks in BENCHMARK_REPORTS.items():
        results[method] = {}

        for k, filename in benchmarks.items():
            report = load_report(REPORTS_DIR / filename)
            results[method][k] = {
                "source_report": filename,
                "metrics": extract_metrics(report, k),
            }

    comparisons = {}

    for k in (3, 5, 10):
        comparisons[k] = {}

        for metric in METRICS:
            metric_key = f"{metric}_at_{k}"

            tfidf_value = results["TF-IDF"][k]["metrics"][
                metric_key
            ]
            bm25_value = results["BM25"][k]["metrics"][
                metric_key
            ]

            if tfidf_value > bm25_value:
                winner = "TF-IDF"
            elif bm25_value > tfidf_value:
                winner = "BM25"
            else:
                winner = "Tie"

            comparisons[k][metric_key] = {
                "tfidf": tfidf_value,
                "bm25": bm25_value,
                "winner": winner,
            }

    return {
        "metadata": {
            "comparison_type": "TF-IDF vs BM25 retrieval",
            "methods": ["TF-IDF", "BM25"],
            "k_values": [3, 5, 10],
            "query_count": 10,
            "document_count": 15,
        },
        "results": results,
        "comparisons": comparisons,
    }


def save_comparison_report(
    comparison: dict[str, Any],
) -> Path:
    """Save the comparison report as formatted JSON."""

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    with COMPARISON_REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(comparison, file, indent=4)

    return COMPARISON_REPORT_PATH


def main() -> None:
    """Run the retrieval benchmark comparison."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    comparison = build_comparison()
    report_path = save_comparison_report(comparison)

    print("\n========== RETRIEVAL COMPARISON ==========")

    for k, metrics in comparison["comparisons"].items():
        print(f"\nK = {k}")

        for metric, values in metrics.items():
            print(
                f"{metric}: "
                f"TF-IDF={values['tfidf']:.4f}, "
                f"BM25={values['bm25']:.4f}, "
                f"Winner={values['winner']}"
            )

    print(f"\nComparison report saved: {report_path.relative_to(PROJECT_ROOT)}")
    print("==========================================\n")


if __name__ == "__main__":
    main()
