
import pytest
import networkx as nx

from src.graph_builder import KnowledgeGraphBuilder
from src.graph_engine import GraphQueryEngine


@pytest.fixture
def engine():
    """Build a fresh knowledge graph for each test."""
    builder = KnowledgeGraphBuilder()
    graph = builder.build_from_corpus("data/documents.json")
    return GraphQueryEngine(graph)


def test_graph_initialization(engine):
    assert isinstance(engine.graph, nx.MultiDiGraph)
    assert engine.graph.number_of_nodes() == 12
    assert engine.graph.number_of_edges() == 54


def test_get_existing_entity(engine):
    entity = engine.get_entity("AuthService")

    assert entity is not None
    assert entity["id"] == "AuthService"
    assert entity["entity_type"] == "SERVICE"


def test_get_nonexistent_entity(engine):
    entity = engine.get_entity("UnknownService")

    assert entity is None


def test_get_all_entities(engine):
    entities = engine.get_all_entities()

    assert len(entities) == 12
    assert any(
        entity["id"] == "AuthService"
        for entity in entities
    )


def test_get_neighbors(engine):
    neighbors = engine.get_neighbors(
        "AuthService",
        direction="outgoing",
    )

    neighbor_ids = {
        item["entity"]["id"]
        for item in neighbors
    }

    assert "RedisCache" in neighbor_ids
    assert "PostgresCluster" in neighbor_ids


def test_get_neighbors_invalid_entity(engine):
    with pytest.raises(ValueError):
        engine.get_neighbors("UnknownService")


def test_get_neighbors_invalid_direction(engine):
    with pytest.raises(ValueError):
        engine.get_neighbors(
            "AuthService",
            direction="sideways",
        )


def test_multi_hop_traversal(engine):
    results = engine.multi_hop_traversal(
        "AuthService",
        max_depth=2,
        direction="both",
    )

    entity_ids = {
        item["entity"]["id"]
        for item in results
    }

    assert len(entity_ids) == 11
    assert "AuthService" not in entity_ids
    assert "Incident-304" in entity_ids


def test_multi_hop_invalid_depth(engine):
    with pytest.raises(ValueError):
        engine.multi_hop_traversal(
            "AuthService",
            max_depth=-1,
        )


def test_shortest_path(engine):
    path = engine.shortest_path(
        "AuthService",
        "PostgresCluster",
    )

    assert path is not None
    assert path[0] == "AuthService"
    assert path[-1] == "PostgresCluster"


def test_shortest_path_invalid_entity(engine):
    with pytest.raises(ValueError):
        engine.shortest_path(
            "UnknownService",
            "PostgresCluster",
        )


def test_filter_relationships(engine):
    relationships = engine.filter_relationships(
        "DEPENDS_ON"
    )

    assert len(relationships) == 8

    assert all(
        item["relation_type"] == "DEPENDS_ON"
        for item in relationships
    )


def test_filter_relationships_case_insensitive(engine):
    relationships = engine.filter_relationships(
        "depends_on"
    )

    assert len(relationships) == 8


def test_extract_subgraph(engine):
    subgraph = engine.extract_subgraph(
        "AuthService",
        max_depth=2,
        direction="both",
    )

    assert isinstance(subgraph, nx.MultiDiGraph)
    assert "AuthService" in subgraph.nodes
    assert subgraph.number_of_nodes() == 12


def test_extract_subgraph_invalid_entity(engine):
    with pytest.raises(ValueError):
        engine.extract_subgraph("UnknownService")

