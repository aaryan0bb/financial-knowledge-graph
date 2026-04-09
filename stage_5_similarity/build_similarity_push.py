#!/usr/bin/env python
"""build_similarity_push.py
Add SIMILAR_TO edges based on embedding similarity and push augmented graph to Neo4j.

Prerequisites
-------------
1.   `pip install voyageai faiss-cpu` (or `faiss-gpu`).
2.   Set environment variable `VOYAGEAI_API_KEY`.

Hard-coded paths & connection details are defined in `main()` for now; edit as needed.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np  # type: ignore
import faiss  # type: ignore
from voyageai import Client  # type: ignore
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

voyage_model1 = "voyage-3-large"

# Get API key from environment
VOYAGEAI_API_KEY = os.getenv("VOYAGE_API_KEY", "")

# Import Neo4j utilities from our utils module
from utils.neo4j_utils import push_to_neo4j

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# -------------------------
# Embedding + similarity
# -------------------------

def load_graph(json_path: Path) -> Dict[str, Any]:
    """Read a merged JSON file produced by merge_combined_jsons.py."""
    data = json.loads(json_path.read_text())
    if not ("entities" in data and "relationships" in data):
        raise ValueError("JSON file missing 'entities' / 'relationships' keys")
    logger.info("Loaded %d entities / %d relationships", len(data["entities"]), len(data["relationships"]))
    return data


def build_embeddings(entities: List[Dict[str, Any]], client: Client, model: str = voyage_model1) -> np.ndarray:
    """Create L2-normalised embeddings for each entity using Voyage AI."""
    texts: List[str] = []
    for ent in entities:
        desc = ent.get("description") or ent.get("brief") or ""
        
        # Add properties to the text for embedding
        properties_str = ""
        if "properties" in ent and isinstance(ent["properties"], dict):
            # Exclude hash keys from embedding text
            properties_str = ", ".join(
                f"{k}: {v}" for k, v in ent["properties"].items()
                if v is not None and not k.endswith("_hash")
            )
        
        text = f"{desc}. {properties_str}".strip()
        texts.append(text)

    logger.info("Calling VoyageAI for %d embeddings (model=%s)…", len(texts), model)
    
    all_embeddings = []
    batch_size = 1000  # Voyage AI batch size limit
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        embeds = client.embed(batch_texts, model=model, input_type="query", truncation=True).embeddings  # type: ignore
        all_embeddings.extend(embeds)

    vectors = np.asarray(all_embeddings, dtype="float32")
    # L2 normalise so cosine similarity == dot-product
    faiss.normalize_L2(vectors)
    return vectors


def derive_similarity_edges(
    entities: List[Dict[str, Any]],
    vectors: np.ndarray,
    k: int = 100,
    threshold: float = 0.7,
    same_doc_threshold: float = 0.9,
    method_name: str = "cosine_voyage",
) -> List[Dict[str, Any]]:
    """Build SIMILAR_TO edges using FAISS inner-product search."""
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    # Query top (k+1) to account for self-match at 0
    sims, idxs = index.search(vectors, k + 1)

    edges: List[Dict[str, Any]] = []
    seen_pairs: set[Tuple[str, str]] = set()

    for i, (row_sims, row_idxs) in enumerate(zip(sims, idxs)):
        src_entity = entities[i]
        # Iterate neighbors skipping self-match
        for sim, j in zip(row_sims[1:], row_idxs[1:]):
            tgt_entity = entities[j]
            # Skip if both entities come from the same chunk
            src_props = src_entity.get("properties", {})
            tgt_props = tgt_entity.get("properties", {})
            if src_props.get("chunk_hash") and tgt_props.get("chunk_hash") and src_props["chunk_hash"] == tgt_props["chunk_hash"]:
                continue
            # Choose threshold: higher for same-document pairs
            if src_props.get("doc_hash") and tgt_props.get("doc_hash") and src_props["doc_hash"] == tgt_props["doc_hash"]:
                current_threshold = same_doc_threshold
            else:
                current_threshold = threshold
            if sim < current_threshold:
                continue

            # Use entity names for deduplication logic to ensure unique (undirected) pairs
            # even though the actual 'source' and 'target' in the edge will be the full entity dicts.
            key = tuple(sorted((src_entity.get("name", ""), tgt_entity.get("name", ""))))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)

            edges.append(
                {
                    "source": src_entity,  # Store the entire source entity dictionary
                    "target": tgt_entity,  # Store the entire target entity dictionary
                    "type": "SIMILAR_TO",
                    "properties": {
                        "similarity": round(float(sim), 4),
                        "method": method_name,
                        "confidence": int(sim * 100),
                    },
                }
            )
    logger.info("Derived %d SIMILAR_TO edges (k=%d, τ=%.2f)", len(edges), k, threshold)
    return edges


# -------------------------
# Main routine (hard-coded)
# -------------------------

def main(
    merged_json_path: Path | None = None,
    output_dir: Path | None = None,
    top_k: int = 50,
    sim_threshold: float = 0.73,
    same_doc_threshold: float = 0.85,
) -> None:
    """
    Build similarity edges from entity embeddings.

    Args:
        merged_json_path: Path to merged JSON from stage 4
        output_dir: Directory to save outputs (embeddings.npy, similarity_edges.json)
        top_k: Number of nearest neighbors to consider
        sim_threshold: Similarity threshold for cross-document pairs
        same_doc_threshold: Higher threshold for same-document pairs
    """
    # Use defaults if not provided
    if merged_json_path is None:
        merged_json_path = Path("./data/merged_1.json")
    if output_dir is None:
        output_dir = merged_json_path.parent

    # Paths
    embeddings_path = output_dir / "embeddings.npy"
    augmented_json_path = output_dir / "similarity_edges.json"

    # Neo4j credentials from environment
    bolt_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo_user = os.getenv("NEO4J_USERNAME", "neo4j")
    neo_pw = os.getenv("NEO4J_PASSWORD", "")

    # Voyage model
    voyage_model = voyage_model1
    # -----------------------------------------------

    data = load_graph(merged_json_path)

    # Load or build embeddings
    if embeddings_path.exists():
        logger.info("Loading embeddings from %s", embeddings_path)
        vectors = np.load(embeddings_path)
    else:
        logger.info("Embeddings file not found. Building embeddings...")
        # Initialise VoyageAI
        api_key = VOYAGEAI_API_KEY
        if not api_key:
            raise RuntimeError("VOYAGEAI_API_KEY environment variable not set")
        client = Client(api_key=api_key)  # type: ignore

        vectors = build_embeddings(data["entities"], client, voyage_model)
        np.save(embeddings_path, vectors)  # Save embeddings for future runs
        logger.info("Saved embeddings to %s", embeddings_path)

    # Derive similarity edges
    sim_edges = derive_similarity_edges(
        data["entities"], vectors, top_k, sim_threshold, same_doc_threshold
    )

    # Append new edges and deduplicate with existing ones
    # existing_keys = {
    #     (rel.get("source"), rel.get("target"), rel.get("type")) for rel in data["relationships"]
    # }
    # for rel in sim_edges:
    #     key = (rel["source"], rel["target"], rel["type"])
    #     if key not in existing_keys:
    #         data["relationships"].append(rel)
    #         existing_keys.add(key)

    logger.info(
        "Total relationships after augmentation: %d (added %d SIMILAR_TO)",
        len(data["relationships"]),
        len(sim_edges),
    )

    # Write augmented JSON (only new SIMILAR_TO edges)
    augmented_json_path.write_text(json.dumps(sim_edges, indent=2))
    logger.info("Saved augmented JSON (only new edges) → %s", augmented_json_path)

    #   Push to Neo4j
  


if __name__ == "__main__":
    main() 