import json

import networkx as nx
import pytest

from src.graph_loader import load_knowledge_graph


def write_graph(tmp_path, data):
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture
def valid_graph_data():
    return {
        "directed": True,
        "graph_type": "MultiDiGraph",
        "nodes": [
            {
                "id": "AuthService",
                "name": "AuthService",
                "entity_type": "SERVICE",
                "description": "Authentication service",
                "metadata": {"document_ids": ["DOC-001"]},
            },
            {
                "id": "PaymentGateway",
                "name": "PaymentGateway",
                "entity_type": "SERVICE",
                "description": "Payment processing service",
                "metadata": {"document_ids": ["DOC-001"]},
            },
        ],
        "edges": [
            {
                "source_id": "AuthService",
                "target_id": "PaymentGateway",
                "relation_type": "DEPENDS_ON",
                "weight": 0.9,
            }
        ],
        "metadata": {
            "documents_processed": 1,
            "node_count": 2,
            "edge_count": 1,
        },
    }


def test_load_knowledge_graph_returns_multidigraph(
    tmp_path,
    valid_graph_data,
):
    path = write_graph(tmp_path, valid_graph_data)

    graph = load_knowledge_graph(path)

    assert isinstance(graph, nx.MultiDiGraph)
    assert graph.number_of_nodes() == 2
    assert graph.number_of_edges() == 1


def test_load_knowledge_graph_preserves_nodes_and_relationships(
    tmp_path,
    valid_graph_data,
):
    path = write_graph(tmp_path, valid_graph_data)

    graph = load_knowledge_graph(path)

    assert graph.nodes["AuthService"]["name"] == "AuthService"
    assert graph.nodes["AuthService"]["entity_type"] == "SERVICE"

    edge = graph["AuthService"]["PaymentGateway"]["DEPENDS_ON"]

    assert edge["relation_type"] == "DEPENDS_ON"
    assert edge["weight"] == 0.9

    assert graph.has_edge("AuthService", "PaymentGateway")
    assert not graph.has_edge("PaymentGateway", "AuthService")


def test_load_knowledge_graph_rejects_non_object_json(tmp_path):
    path = tmp_path / "graph.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="must contain a JSON object",
    ):
        load_knowledge_graph(path)


def test_load_knowledge_graph_rejects_invalid_entity(
    tmp_path,
    valid_graph_data,
):
    valid_graph_data["nodes"][0]["entity_type"] = "UNKNOWN"

    path = write_graph(tmp_path, valid_graph_data)

    with pytest.raises(
        ValueError,
        match="Invalid knowledge graph data",
    ):
        load_knowledge_graph(path)


def test_load_knowledge_graph_rejects_missing_relationship_endpoint(
    tmp_path,
    valid_graph_data,
):
    valid_graph_data["edges"][0]["target_id"] = "MissingService"

    path = write_graph(tmp_path, valid_graph_data)

    with pytest.raises(
        ValueError,
        match="Target entity does not exist",
    ):
        load_knowledge_graph(path)
