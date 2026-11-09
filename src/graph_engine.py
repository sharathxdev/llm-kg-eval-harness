
from collections import deque
from typing import Any

import networkx as nx


class GraphQueryEngine:
    """
    Provides querying and traversal operations over a
    NetworkX knowledge graph.
    """

    def __init__(self, graph: nx.MultiDiGraph):
        if not isinstance(graph, nx.MultiDiGraph):
            raise TypeError("Expected a NetworkX MultiDiGraph.")

        self.graph = graph

    def get_entity(self, entity_id: str) -> dict[str, Any] | None:
        """
        Retrieve an entity by its unique ID.
        """
        if not self.graph.has_node(entity_id):
            return None

        return {
            "id": entity_id,
            **self.graph.nodes[entity_id],
        }

    def get_all_entities(self) -> list[dict[str, Any]]:
        """
        Return all entities in the knowledge graph.
        """
        return [
            {
                "id": node_id,
                **attributes,
            }
            for node_id, attributes in self.graph.nodes(data=True)
        ]

    def get_neighbors(
        self,
        entity_id: str,
        direction: str = "both",
        relation_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve directly connected entities.

        direction:
            outgoing - follow outgoing edges
            incoming - follow incoming edges
            both      - follow incoming and outgoing edges
        """
        if not self.graph.has_node(entity_id):
            raise ValueError(f"Entity does not exist: {entity_id}")

        if direction not in {"outgoing", "incoming", "both"}:
            raise ValueError(
                "direction must be outgoing, incoming, or both."
            )

        neighbors = {}

        if direction in {"outgoing", "both"}:
            for neighbor in self.graph.successors(entity_id):
                neighbors.setdefault(neighbor, set())

        if direction in {"incoming", "both"}:
            for neighbor in self.graph.predecessors(entity_id):
                neighbors.setdefault(neighbor, set())

        results = []

        for neighbor_id in sorted(neighbors):
            if direction in {"outgoing", "both"}:
                edge_groups = self.graph.get_edge_data(
                    entity_id, neighbor_id, default={}
                )
                for attributes in edge_groups.values():
                    rel = attributes.get("relation_type")
                    if relation_type is None or rel == relation_type.upper():
                        results.append({
                            "entity": self.get_entity(neighbor_id),
                            "direction": "outgoing",
                            "relation_type": rel,
                            "weight": attributes.get("weight", 1.0),
                        })

            if direction in {"incoming", "both"}:
                edge_groups = self.graph.get_edge_data(
                    neighbor_id, entity_id, default={}
                )
                for attributes in edge_groups.values():
                    rel = attributes.get("relation_type")
                    if relation_type is None or rel == relation_type.upper():
                        results.append({
                            "entity": self.get_entity(neighbor_id),
                            "direction": "incoming",
                            "relation_type": rel,
                            "weight": attributes.get("weight", 1.0),
                        })

        return results

    def multi_hop_traversal(
        self,
        start_entity_id: str,
        max_depth: int = 2,
        direction: str = "outgoing",
    ) -> list[dict[str, Any]]:
        """
        Traverse the graph using breadth-first search.

        Returns entities discovered within max_depth hops.
        Each entity is returned only once, at its shortest depth.
        """
        if not self.graph.has_node(start_entity_id):
            raise ValueError(
                f"Starting entity does not exist: {start_entity_id}"
            )

        if max_depth < 0:
            raise ValueError("max_depth cannot be negative.")

        if direction not in {"outgoing", "incoming", "both"}:
            raise ValueError(
                "direction must be outgoing, incoming, or both."
            )

        visited = {start_entity_id}
        queue = deque([(start_entity_id, 0)])
        results = []

        while queue:
            current_id, depth = queue.popleft()

            if depth >= max_depth:
                continue

            if direction == "outgoing":
                neighbors = self.graph.successors(current_id)
            elif direction == "incoming":
                neighbors = self.graph.predecessors(current_id)
            else:
                neighbors = set(self.graph.successors(current_id))
                neighbors.update(self.graph.predecessors(current_id))

            for neighbor_id in sorted(neighbors):
                if neighbor_id in visited:
                    continue

                visited.add(neighbor_id)
                next_depth = depth + 1

                results.append({
                    "entity": self.get_entity(neighbor_id),
                    "depth": next_depth,
                })

                queue.append((neighbor_id, next_depth))

        return results

    def shortest_path(
        self,
        source_id: str,
        target_id: str,
    ) -> list[str] | None:
        """
        Find the shortest directed path between two entities.
        """
        if not self.graph.has_node(source_id):
            raise ValueError(f"Source entity does not exist: {source_id}")

        if not self.graph.has_node(target_id):
            raise ValueError(f"Target entity does not exist: {target_id}")

        try:
            return nx.shortest_path(
                self.graph,
                source=source_id,
                target=target_id,
            )
        except nx.NetworkXNoPath:
            return None

    def filter_relationships(
        self,
        relation_type: str,
    ) -> list[dict[str, Any]]:
        """
        Return all graph edges matching a relationship type.
        """
        if not relation_type.strip():
            raise ValueError("relation_type cannot be empty.")

        expected_type = relation_type.strip().upper()
        results = []

        for source, target, key, attributes in self.graph.edges(
            keys=True, data=True
        ):
            actual_type = attributes.get("relation_type", key)

            if actual_type == expected_type:
                results.append({
                    "source": self.get_entity(source),
                    "target": self.get_entity(target),
                    "relation_type": actual_type,
                    "weight": attributes.get("weight", 1.0),
                })

        return results

    def extract_subgraph(
        self,
        entity_id: str,
        max_depth: int = 1,
        direction: str = "both",
    ) -> nx.MultiDiGraph:
        """
        Extract a smaller graph around a selected entity.
        """
        if not self.graph.has_node(entity_id):
            raise ValueError(f"Entity does not exist: {entity_id}")

        if max_depth < 0:
            raise ValueError("max_depth cannot be negative.")

        traversal = self.multi_hop_traversal(
            start_entity_id=entity_id,
            max_depth=max_depth,
            direction=direction,
        )

        node_ids = {entity_id}

        for item in traversal:
            node_ids.add(item["entity"]["id"])

        return self.graph.subgraph(node_ids).copy()

