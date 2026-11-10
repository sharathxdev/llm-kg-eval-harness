import json

import pytest

from src.models import (
    EntityNode,
    EntityType,
    ExtractionResult,
    RelationEdge,
)
from src.run_graph_extraction import run_graph_extraction


def make_entity(entity_id, name):
    return EntityNode(
        id=entity_id,
        name=name,
        entity_type=EntityType.SERVICE,
        description=f"{name} description",
    )


def make_extraction(document_id, include_relation=False):
    entities = [make_entity("AuthService", "AuthService")]

    relations = []

    if include_relation:
        entities.append(
            make_entity("PaymentGateway", "PaymentGateway")
        )
        relations.append(
            RelationEdge(
                source_id="AuthService",
                target_id="PaymentGateway",
                relation_type="DEPENDS_ON",
            )
        )

    return ExtractionResult(
        document_id=document_id,
        entities=entities,
        relations=relations,
    )


class FakeExtractor:
    """Fake extractor that never makes an external API request."""

    def __init__(self, results, failing_document=None):
        self.results = results
        self.failing_document = failing_document
        self.last_usage = {}

    def extract(self, document):
        document_id = document["id"]

        if document_id == self.failing_document:
            raise RuntimeError("Simulated extraction failure")

        self.last_usage = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "latency_seconds": 0.25,
        }

        return self.results[document_id]


@pytest.fixture
def documents_file(tmp_path):
    documents = [
        {
            "id": "DOC-001",
            "title": "Authentication Architecture",
            "category": "Architecture",
            "content": "AuthService manages authentication.",
        },
        {
            "id": "DOC-002",
            "title": "Payment Architecture",
            "category": "Architecture",
            "content": "AuthService supports PaymentGateway.",
        },
    ]

    path = tmp_path / "documents.json"
    path.write_text(json.dumps(documents), encoding="utf-8")

    return path


def test_run_graph_extraction_builds_graph_and_saves_json(
    documents_file,
    tmp_path,
):
    extractor = FakeExtractor(
        {
            "DOC-001": make_extraction(
                "DOC-001",
                include_relation=True,
            ),
            "DOC-002": make_extraction("DOC-002"),
        }
    )

    extractions_path = tmp_path / "output" / "extractions.json"
    graph_path = tmp_path / "output" / "graph.json"

    result = run_graph_extraction(
        documents_path=documents_file,
        extractions_output_path=extractions_path,
        graph_output_path=graph_path,
        extractor=extractor,
    )

    summary = result["summary"]

    assert summary["documents_processed"] == 2
    assert summary["entities"] == 2
    assert summary["relationships"] == 1

    assert summary["usage"]["prompt_tokens"] == 200
    assert summary["usage"]["completion_tokens"] == 100
    assert summary["usage"]["total_tokens"] == 300
    assert summary["usage"]["latency_seconds"] == 0.5

    assert extractions_path.exists()
    assert graph_path.exists()

    extraction_data = json.loads(
        extractions_path.read_text(encoding="utf-8")
    )
    graph_data = json.loads(
        graph_path.read_text(encoding="utf-8")
    )

    assert extraction_data["documents_processed"] == 2
    assert len(extraction_data["extractions"]) == 2

    assert graph_data["directed"] is True
    assert graph_data["graph_type"] == "MultiDiGraph"
    assert len(graph_data["nodes"]) == 2
    assert len(graph_data["edges"]) == 1

    edge = graph_data["edges"][0]

    assert edge["source_id"] == "AuthService"
    assert edge["target_id"] == "PaymentGateway"
    assert edge["relation_type"] == "DEPENDS_ON"


def test_run_graph_extraction_preserves_document_provenance(
    documents_file,
    tmp_path,
):
    extractor = FakeExtractor(
        {
            "DOC-001": make_extraction("DOC-001"),
            "DOC-002": make_extraction("DOC-002"),
        }
    )

    result = run_graph_extraction(
        documents_path=documents_file,
        extractions_output_path=tmp_path / "extractions.json",
        graph_output_path=tmp_path / "graph.json",
        extractor=extractor,
    )

    graph = result["graph"]

    assert graph.number_of_nodes() == 1

    metadata = graph.nodes["AuthService"]["metadata"]

    assert metadata["document_ids"] == ["DOC-001", "DOC-002"]


def test_run_graph_extraction_stops_on_extraction_failure(
    documents_file,
    tmp_path,
):
    extractor = FakeExtractor(
        {
            "DOC-001": make_extraction("DOC-001"),
        },
        failing_document="DOC-002",
    )

    extractions_path = tmp_path / "output" / "extractions.json"
    graph_path = tmp_path / "output" / "graph.json"

    with pytest.raises(
        RuntimeError,
        match="Extraction failed for document DOC-002",
    ):
        run_graph_extraction(
            documents_path=documents_file,
            extractions_output_path=extractions_path,
            graph_output_path=graph_path,
            extractor=extractor,
        )

    assert not extractions_path.exists()
    assert not graph_path.exists()


def test_run_graph_extraction_rejects_same_output_path(
    documents_file,
    tmp_path,
):
    output_path = tmp_path / "same.json"

    with pytest.raises(
        ValueError,
        match="output paths must be different",
    ):
        run_graph_extraction(
            documents_path=documents_file,
            extractions_output_path=output_path,
            graph_output_path=output_path,
            extractor=FakeExtractor({}),
        )


def test_run_graph_extraction_rejects_empty_document_corpus(
    tmp_path,
):
    documents_path = tmp_path / "empty.json"
    documents_path.write_text("[]", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="No valid documents found",
    ):
        run_graph_extraction(
            documents_path=documents_path,
            extractions_output_path=tmp_path / "extractions.json",
            graph_output_path=tmp_path / "graph.json",
            extractor=FakeExtractor({}),
        )
