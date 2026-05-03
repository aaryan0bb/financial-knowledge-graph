"""
Stage 8: Combine merged triplets with enhanced relationships and push to Neo4j.

This module merges the original extracted triplets with the semantically enhanced
relationships and pushes the final combined graph to Neo4j.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
import os

from dotenv import load_dotenv

# Neo4j imports
try:
    from langchain_community.graphs import Neo4jGraph  # Older LC versions
except ImportError:
    from langchain_neo4j import Neo4jGraph  # Newer package name

from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def push_to_neo4j(data: Dict[str, Any], bolt: str, user: str, pw: str):
    os.environ.update(NEO4J_URI=bolt, NEO4J_USERNAME=user, NEO4J_PASSWORD=pw)
    graph = Neo4jGraph()

    # Map raw entity dicts to Node objects
    node_objects = {}
    for ent in data["entities"]:
        node_properties = {}
        if "brief" in ent:
            node_properties["brief"] = ent["brief"]
        if "description" in ent:
            node_properties["description"] = ent["description"]
        
        # Merge the nested 'properties' dictionary into the main node properties
        if "properties" in ent and isinstance(ent["properties"], dict):
            node_properties.update(ent["properties"])
            
        node = Node(id=ent["name"], type=ent.get("type", "Unknown"), properties=node_properties)
        node_objects[ent["name"]] = node

    # Map raw relationship dicts to Relationship objects
    relationship_objects = []
    for rel_dict in data["relationships"]:
        source_node = node_objects.get(rel_dict["source"])
        target_node = node_objects.get(rel_dict["target"])
        if source_node and target_node:
            # Start with any top-level primitive properties (e.g. description)
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
                    type=rel_dict["type"],
                    properties=flat_props,
                )
            )
        else:
            logger.warning("Skipping relationship with missing source or target node: %s", rel_dict)

    # Create a dummy source Document for GraphDocument
    source_document = Document(page_content="", metadata={"file_path": data.get("file_path", "Unknown")})

    graph_document = GraphDocument(nodes=list(node_objects.values()), relationships=relationship_objects, source=source_document)
    graph.add_graph_documents([graph_document])
    logger.info("Loaded %d nodes / %d rels into Neo4j", len(node_objects), len(relationship_objects))

def _clean_properties(props: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively clean properties to be Neo4j-compatible."""
    cleaned = {}
    for k, v in props.items():
        if v is None or v == "NO_VALUE" or v == "": # Filter None, "NO_VALUE", empty strings
            continue
        if isinstance(v, dict):
            # Neo4j properties cannot be nested maps, convert to string or flatten as needed
            # For now, convert nested dicts to a JSON string representation
            cleaned[k] = json.dumps(v)
        elif isinstance(v, list):
            # Ensure list elements are primitive. Convert non-primitive to string.
            cleaned_list = []
            for item in v:
                if isinstance(item, (str, int, float, bool)):
                    cleaned_list.append(item)
                else:
                    cleaned_list.append(str(item))
            cleaned[k] = cleaned_list
        else:
            cleaned[k] = v
    return cleaned

# --- Configuration from environment ---
MERGED_JSON_PATH = Path(os.getenv("MERGED_JSON_PATH", "./data/merged.json"))
ENHANCED_JSON_PATH = Path(os.getenv("CLEANED_RELATIONSHIPS_PATH", "./data/enhanced_relationships_cleaned.json"))

BOLT_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
# ------------------------------------------

# ---------------------------------------------------------------
# Local helpers remain for merging logic only; Neo4j push is done
# by imported push_to_neo4j above.
# ---------------------------------------------------------------


def combine_and_push_graph(
    merged_json_path: Path,
    enhanced_json_path: Path,
    bolt_uri: str,
    user: str,
    password: str,
):
    # 1. Load merged.json
    try:
        with open(merged_json_path, 'r') as f:
            merged_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: merged.json not found at {merged_json_path}")
        return
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from merged.json: {e}")
        return

    if "entities" not in merged_data or "relationships" not in merged_data:
        print("Error: merged.json structure invalid. Expected 'entities' and 'relationships'.")
        return

    # Initialize combined graph with data from merged.json
    combined_entities = merged_data["entities"]
    combined_relationships = merged_data["relationships"]

    # Create a set of existing relationships for conflict checking (source, target, type)
    seen_relationships = set()
    for rel in combined_relationships:
        # Assuming 'type' is the key for relationship type in merged.json after previous cleaning step
        if isinstance(rel, dict) and "source" in rel and "target" in rel:
            seen_relationships.add((rel["source"], rel["target"]))
        else:
            print(f"Warning: Malformed relationship in merged.json, skipping for conflict check: {rel}")

    # 2. Load enhanced_relationships.json
    try:
        with open(enhanced_json_path, 'r') as f:
            enhanced_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: enhanced_relationships.json not found at {enhanced_json_path}")
        return
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from enhanced_relationships.json: {e}")
        return

    if "enhanced_relationships" not in enhanced_data or not isinstance(enhanced_data["enhanced_relationships"], list):
        print("Error: enhanced_relationships.json structure invalid. Expected 'enhanced_relationships' list.")
        return

    # 3. Merge relationships from enhanced_relationships.json (prioritizing merged.json)
    for enhanced_rel in enhanced_data["enhanced_relationships"]:
        # Ensure it's a dict and has required keys for comparison
        if isinstance(enhanced_rel, dict) and "source" in enhanced_rel and "target" in enhanced_rel:
            # Conflict check: (source, target, type) must be identical to be considered a conflict
            current_rel_key = (enhanced_rel["source"], enhanced_rel["target"])
            if current_rel_key not in seen_relationships:
                combined_relationships.append(enhanced_rel)
                seen_relationships.add(current_rel_key)
            else:
                print(f"Conflict detected and discarded from enhanced_relationships.json: {enhanced_rel}")
        else:
            print(f"Warning: Malformed enhanced relationship, skipping: {enhanced_rel}")

    # Prepare data for push_to_neo4j (matching structure expected by knowledge_graph_builder_clean)
    final_data_to_push: Dict[str, Any] = {
        "entities": combined_entities,
        "relationships": combined_relationships,
        "file_path": str(merged_json_path)  # metadata for source document
    }

    # 4. Push to Neo4j using the proven function
    push_to_neo4j(final_data_to_push, bolt_uri, user, password)
  
     

if __name__ == "__main__":
    # Run the combination and push process
    combine_and_push_graph(
        MERGED_JSON_PATH,
        ENHANCED_JSON_PATH,
        BOLT_URI,
        NEO4J_USER,
        NEO4J_PASSWORD,
    ) 