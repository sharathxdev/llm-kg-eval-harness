"""Compare TF-IDF, BM25, and RRF retrieval evaluation reports."""

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
    "RRF": {
        3: "retrieval_evaluation_report_rrf_k3.json",
        5: "retrieval_evaluation_report_rrf_k5.json",
        10: "retrieval_evaluation_report_rrf_k10.json",
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
    """Compare all configured retrieval methods across their K values."""

    if not BENCHMARK_REPORTS:
        raise ValueError("At least one benchmark method must be configured.")

    k_value_sets = [
        set(benchmarks)
        for benchmarks in BENCHMARK_REPORTS.values()
    ]

    if not k_value_sets[0] or any(
        values != k_value_sets[0]
        for values in k_value_sets[1:]
    ):
        raise ValueError(
            "All retrieval methods must define the same non-empty K values."
        )

    k_values = sorted(k_value_sets[0])

    results = {}
    reference_report = None

    for method, benchmarks in BENCHMARK_REPORTS.items():
        results[method] = {}

        for k in k_values:
            filename = benchmarks[k]
            report = load_report(REPORTS_DIR / filename)

            if reference_report is None:
                reference_report = report

            results[method][k] = {
                "source_report": filename,
                "metrics": extract_metrics(report, k),
            }

    # Create stable dictionary keys while preserving display labels.
    method_keys = {
        method: method.lower().replace("-", "")
        for method in BENCHMARK_REPORTS
    }

    comparisons = {}

    for k in k_values:
        comparisons[k] = {}

        metric_keys = [
            f"{metric}_at_{k}"
            for metric in METRICS
        ]
        metric_keys.append("graph_entity_f1")

        for metric_key in metric_keys:
            values = {
                method_keys[method]: results[method][k]["metrics"][
                    metric_key
                ]
                for method in BENCHMARK_REPORTS
            }

            best_value = max(values.values())

            winners = [
                method
                for method, key in method_keys.items()
                if values[key] == best_value
            ]

            winner = winners[0] if len(winners) == 1 else "Tie"

            comparisons[k][metric_key] = {
                **values,
                "winner": winner,
            }

    configuration = (
        reference_report.get("configuration", {})
        if reference_report is not None
        else {}
    )
    reference_metadata = (
        reference_report.get("metadata", {})
        if reference_report is not None
        else {}
    )

    if not isinstance(configuration, dict):
        configuration = {}

    if not isinstance(reference_metadata, dict):
        reference_metadata = {}

    return {
        "metadata": {
            "comparison_type": "Multi-method retrieval comparison",
            "methods": list(BENCHMARK_REPORTS),
            "k_values": k_values,
            "query_count": configuration.get("query_count", 10),
            "document_count": reference_metadata.get("document_count", 15),
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

    method_keys = {
        method: method.lower().replace("-", "")
        for method in comparison["metadata"]["methods"]
    }

    for k, metrics in comparison["comparisons"].items():
        print(f"\nK = {k}")

        for metric, values in metrics.items():
            method_values = ", ".join(
                f"{method}={values[key]:.4f}"
                for method, key in method_keys.items()
            )

            print(
                f"{metric}: {method_values}, "
                f"Winner={values['winner']}"
            )

    print(f"\nComparison report saved: {report_path.relative_to(PROJECT_ROOT)}")
    print("==========================================\n")


if __name__ == "__main__":
    main()
