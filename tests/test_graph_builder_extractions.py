import pytest

from src.graph_builder import KnowledgeGraphBuilder
from src.models import (
    EntityNode,
    EntityType,
    ExtractionResult,
    RelationEdge,
)


def make_entity(entity_id, name, metadata=None):
    return EntityNode(
        id=entity_id,
        name=name,
        entity_type=EntityType.SERVICE,
        metadata=metadata or {},
    )


def make_extraction(document_id, entities, relations=None):
    return ExtractionResult(
        document_id=document_id,
        entities=entities,
        relations=relations or [],
    )


def test_build_graph_from_extraction():
    builder = KnowledgeGraphBuilder()

    extraction = make_extraction(
        "DOC-001",
        [
            make_entity("AuthService", "AuthService"),
            make_entity("PaymentGateway", "PaymentGateway"),
        ],
        [
            RelationEdge(
                source_id="AuthService",
                target_id="PaymentGateway",
                relation_type="DEPENDS_ON",
            )
        ],
    )

    graph = builder.build_from_extractions([extraction])

    assert graph.number_of_nodes() == 2
    assert graph.number_of_edges() == 1
    assert graph.has_edge("AuthService", "PaymentGateway")


def test_merge_entity_across_documents_and_preserve_provenance():
    builder = KnowledgeGraphBuilder()

    first = make_extraction(
        "DOC-001",
        [make_entity("AuthService", "AuthService")],
    )
    second = make_extraction(
        "DOC-002",
        [make_entity("AuthService", "AuthService")],
    )

    graph = builder.build_from_extractions([first, second])

    assert graph.number_of_nodes() == 1
    assert graph.nodes["AuthService"]["metadata"]["document_ids"] == [
        "DOC-001",
        "DOC-002",
    ]


def test_unknown_relationship_endpoint_is_rejected_without_mutation():
    builder = KnowledgeGraphBuilder()

    extraction = make_extraction(
        "DOC-001",
        [make_entity("AuthService", "AuthService")],
        [
            RelationEdge(
                source_id="AuthService",
                target_id="UnknownService",
                relation_type="DEPENDS_ON",
            )
        ],
    )

    with pytest.raises(ValueError, match="unknown target entity"):
        builder.build_from_extractions([extraction])

    assert builder.graph.number_of_nodes() == 0
    assert builder.graph.number_of_edges() == 0


def test_conflicting_entity_definitions_are_rejected_without_mutation():
    builder = KnowledgeGraphBuilder()

    first = make_extraction(
        "DOC-001",
        [make_entity("Service-1", "AuthService")],
    )
    second = make_extraction(
        "DOC-002",
        [make_entity("Service-1", "PaymentGateway")],
    )

    with pytest.raises(ValueError, match="Conflicting definitions"):
        builder.build_from_extractions([first, second])

    assert builder.graph.number_of_nodes() == 0


def test_invalid_document_ids_metadata_is_rejected_without_mutation():
    builder = KnowledgeGraphBuilder()

    extraction = make_extraction(
        "DOC-001",
        [
            make_entity(
                "AuthService",
                "AuthService",
                metadata={"document_ids": "DOC-OLD"},
            )
        ],
    )

    with pytest.raises(ValueError, match="Invalid document_ids metadata"):
        builder.build_from_extractions([extraction])

    assert builder.graph.number_of_nodes() == 0


def test_non_extraction_input_is_rejected():
    builder = KnowledgeGraphBuilder()

    with pytest.raises(TypeError, match="list"):
        builder.build_from_extractions("not-a-list")

    with pytest.raises(TypeError, match="ExtractionResult"):
        builder.build_from_extractions([{"document_id": "DOC-001"}])


def test_empty_extraction_list_returns_empty_graph():
    builder = KnowledgeGraphBuilder()

    graph = builder.build_from_extractions([])

    assert graph.number_of_nodes() == 0
    assert graph.number_of_edges() == 0
