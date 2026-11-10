"""Run LLM-based entity and relationship extraction over a document corpus."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

from src.data_processor import DataProcessor
from src.extractor import LLMEntityExtractor
from src.graph_builder import KnowledgeGraphBuilder

logger = logging.getLogger(__name__)

DEFAULT_DOCUMENTS_PATH = Path("data/documents.json")
DEFAULT_EXTRACTIONS_PATH = Path(
    "evals/reports/llm_extractions.json"
)
DEFAULT_GRAPH_PATH = Path(
    "evals/reports/llm_knowledge_graph.json"
)


def _serialize_graph(graph: Any) -> dict[str, Any]:
    """Convert a NetworkX MultiDiGraph into JSON-compatible data."""

    nodes = [
        {
            "id": node_id,
            "name": attributes["name"],
            "entity_type": attributes["entity_type"],
            "description": attributes.get("description", ""),
            "metadata": attributes.get("metadata", {}),
        }
        for node_id, attributes in sorted(
            graph.nodes(data=True),
            key=lambda item: item[0],
        )
    ]

    edges = [
        {
            "source_id": source_id,
            "target_id": target_id,
            "relation_type": attributes["relation_type"],
            "weight": attributes["weight"],
        }
        for source_id, target_id, _, attributes in sorted(
            graph.edges(keys=True, data=True),
            key=lambda item: (item[0], item[1], str(item[2])),
        )
    ]

    return {
        "directed": True,
        "graph_type": "MultiDiGraph",
        "nodes": nodes,
        "edges": edges,
    }


def _write_json(path: Path, data: Any) -> None:
    """Write JSON data, creating parent directories when necessary."""

    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def run_graph_extraction(
    documents_path: str | Path = DEFAULT_DOCUMENTS_PATH,
    extractions_output_path: str | Path = DEFAULT_EXTRACTIONS_PATH,
    graph_output_path: str | Path = DEFAULT_GRAPH_PATH,
    extractor: Any | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """
    Extract entities and relationships and construct a consolidated graph.

    An extractor can be injected for testing. When omitted, the real
    OpenAI-backed LLMEntityExtractor is initialized.

    Output files are written only after all documents have been
    extracted and graph construction has succeeded.
    """

    start_time = time.perf_counter()

    documents_path = Path(documents_path)
    extractions_output_path = Path(extractions_output_path)
    graph_output_path = Path(graph_output_path)

    if extractions_output_path.resolve() == graph_output_path.resolve():
        raise ValueError(
            "Extraction output and graph output paths must be different."
        )

    processor = DataProcessor(strict=True)
    documents = processor.load_documents(documents_path)

    if not documents:
        raise ValueError(
            f"No valid documents found in {documents_path}."
        )

    active_extractor = extractor or LLMEntityExtractor(model=model)

    extraction_results = []

    usage_totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "latency_seconds": 0.0,
    }

    logger.info(
        "Starting extraction for %d documents.",
        len(documents),
    )

    for index, document in enumerate(documents, start=1):
        document_id = document["id"]

        logger.info(
            "Extracting document %d/%d: %s",
            index,
            len(documents),
            document_id,
        )

        try:
            extraction = active_extractor.extract(document)
        except Exception as exc:
            raise RuntimeError(
                f"Extraction failed for document {document_id}. "
                "No output files were written."
            ) from exc

        extraction_results.append(extraction)

        usage = active_extractor.last_usage

        for field in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
        ):
            usage_totals[field] += int(usage.get(field, 0))

        usage_totals["latency_seconds"] += float(
            usage.get("latency_seconds", 0.0)
        )

    # Build the graph only after every document has been extracted.
    builder = KnowledgeGraphBuilder()
    graph = builder.build_from_extractions(extraction_results)

    graph_data = _serialize_graph(graph)
    graph_stats = builder.get_graph_stats()

    processing_seconds = round(
        time.perf_counter() - start_time,
        4,
    )

    usage_totals["latency_seconds"] = round(
        usage_totals["latency_seconds"],
        4,
    )

    extraction_data = {
        "evaluation_type": "llm_extraction",
        "documents_processed": len(documents),
        "model": model,
        "extractions": [
            extraction.model_dump(mode="json")
            for extraction in extraction_results
        ],
        "usage": usage_totals,
        "processing_seconds": processing_seconds,
    }

    graph_data["metadata"] = {
        "documents_processed": len(documents),
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "graph_stats": graph_stats,
        "processing_seconds": processing_seconds,
    }

    # Save only after successful extraction and graph construction.
    _write_json(extractions_output_path, extraction_data)
    _write_json(graph_output_path, graph_data)

    summary = {
        "documents_processed": len(documents),
        "entities": graph.number_of_nodes(),
        "relationships": graph.number_of_edges(),
        "usage": usage_totals,
        "processing_seconds": processing_seconds,
        "extractions_output": str(extractions_output_path),
        "graph_output": str(graph_output_path),
        "graph_stats": graph_stats,
    }

    logger.info("LLM graph extraction completed: %s", summary)

    return {
        "summary": summary,
        "graph": graph,
        "extractions": extraction_results,
    }


def main() -> None:
    """Command-line entry point."""

    parser = argparse.ArgumentParser(
        description=(
            "Extract entities and relationships from documents "
            "and build a knowledge graph using an LLM."
        )
    )

    parser.add_argument(
        "--documents",
        type=Path,
        default=DEFAULT_DOCUMENTS_PATH,
        help="Path to the source documents JSON file.",
    )

    parser.add_argument(
        "--extractions-output",
        type=Path,
        default=DEFAULT_EXTRACTIONS_PATH,
        help="Path for the extracted entities and relationships JSON.",
    )

    parser.add_argument(
        "--graph-output",
        type=Path,
        default=DEFAULT_GRAPH_PATH,
        help="Path for the consolidated knowledge graph JSON.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional OpenAI model override.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    result = run_graph_extraction(
        documents_path=args.documents,
        extractions_output_path=args.extractions_output,
        graph_output_path=args.graph_output,
        model=args.model,
    )

    print("\n========== LLM GRAPH EXTRACTION SUMMARY ==========")
    print(json.dumps(result["summary"], indent=2))
    print("==================================================")


if __name__ == "__main__":
    main()
