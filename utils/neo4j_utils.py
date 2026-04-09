"""
Neo4j utility functions for pushing knowledge graph data to Neo4j.

This module provides the push_to_neo4j function which converts entity/relationship
dictionaries to LangChain GraphDocument format and pushes them to Neo4j.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_community.graphs import Neo4jGraph
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship

logger = logging.getLogger(__name__)


def push_to_neo4j(
    data: Dict[str, Any],
    bolt: str | None = None,
    user: str | None = None,
    pw: str | None = None
) -> None:
    """
    Push entities and relationships to Neo4j database.

    Args:
        data: Dictionary containing 'entities' and 'relationships' lists
        bolt: Neo4j bolt URI (defaults to NEO4J_URI env var)
        user: Neo4j username (defaults to NEO4J_USERNAME env var)
        pw: Neo4j password (defaults to NEO4J_PASSWORD env var)

    The data dict should have this structure:
    {
        "entities": [
            {"name": "...", "type": "...", "brief": "...", "description": "...", "properties": {...}}
        ],
        "relationships": [
            {"source": "...", "target": "...", "type": "...", "properties": {...}}
        ],
        "file_path": "..." (optional metadata)
    }
    """
    # Use environment variables if not provided
    bolt = bolt or os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = user or os.getenv("NEO4J_USERNAME", "neo4j")
    pw = pw or os.getenv("NEO4J_PASSWORD", "")

    if not pw:
        raise ValueError("Neo4j password is required. Set NEO4J_PASSWORD env var or pass pw parameter.")

    # Set environment variables for Neo4jGraph
    os.environ.update(NEO4J_URI=bolt, NEO4J_USERNAME=user, NEO4J_PASSWORD=pw)
    graph = Neo4jGraph()

    # Map raw entity dicts to Node objects
    node_objects: Dict[str, Node] = {}
    for ent in data.get("entities", []):
        node_properties = {}

        # Add standard properties
        if "brief" in ent:
            node_properties["brief"] = ent["brief"]
        if "description" in ent:
            node_properties["description"] = ent["description"]

        # Merge the nested 'properties' dictionary into the main node properties
        if "properties" in ent and isinstance(ent["properties"], dict):
            node_properties.update(ent["properties"])

        node = Node(
            id=ent["name"],
            type=ent.get("type", "Unknown"),
            properties=node_properties
        )
        node_objects[ent["name"]] = node

    # Map raw relationship dicts to Relationship objects
    relationship_objects: List[Relationship] = []
    for rel_dict in data.get("relationships", []):
        source_node = node_objects.get(rel_dict.get("source"))
        target_node = node_objects.get(rel_dict.get("target"))

        if source_node and target_node:
            # Start with any top-level primitive properties
            flat_props = {
                k: v for k, v in rel_dict.items()
                if k not in {"source", "target", "type", "properties"}
                and v is not None and v != "NO_VALUE"
            }

            # Merge the nested `properties` map if present
            if "properties" in rel_dict and isinstance(rel_dict["properties"], dict):
                flat_props.update({
                    k: v for k, v in rel_dict["properties"].items()
                    if v is not None and v != "NO_VALUE"
                })

            relationship_objects.append(
                Relationship(
                    source=source_node,
                    target=target_node,
                    type=rel_dict.get("type", "RELATED_TO"),
                    properties=flat_props,
                )
            )
        else:
            logger.warning(
                "Skipping relationship with missing source or target node: %s",
                rel_dict
            )

    # Create a source Document for GraphDocument
    source_document = Document(
        page_content="",
        metadata={"file_path": data.get("file_path", "Unknown")}
    )

    graph_document = GraphDocument(
        nodes=list(node_objects.values()),
        relationships=relationship_objects,
        source=source_document
    )
    graph.add_graph_documents([graph_document])
    logger.info(
        "Loaded %d nodes / %d rels into Neo4j",
        len(node_objects),
        len(relationship_objects)
    )


def clear_neo4j_database(
    bolt: str | None = None,
    user: str | None = None,
    pw: str | None = None
) -> None:
    """
    Clear all nodes and relationships from Neo4j database.
    Use with caution!
    """
    bolt = bolt or os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = user or os.getenv("NEO4J_USERNAME", "neo4j")
    pw = pw or os.getenv("NEO4J_PASSWORD", "")

    os.environ.update(NEO4J_URI=bolt, NEO4J_USERNAME=user, NEO4J_PASSWORD=pw)
    graph = Neo4jGraph()
    graph.query("MATCH (n) DETACH DELETE n")
    logger.info("Cleared all nodes and relationships from Neo4j")
