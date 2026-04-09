#!/usr/bin/env python
"""merge_combined_jsons.py
Merge all *combined* JSON files in a folder into a single entity / relationship set.

Typical usage
-------------
python merge_combined_jsons.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict
import hashlib


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def merge_folder(folder: Path) -> Dict[str, Any]:
    """Merge every *.json file in *folder* into one combined structure.

    The input JSON files must share the schema produced by
    `knowledge_graph_builder_clean.py combine` (top-level keys: entities, relationships).
    """
    combined: Dict[str, Any] = {"entities": [], "relationships": []}
    seen_e, seen_r = set(), set()

    for path in folder.glob("*.json"):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            logger.warning("Skipping malformed JSON %s", path)
            continue

        for ent in data.get("entities", []):
            if not isinstance(ent, dict):
                continue
            key = (ent.get("name"), ent.get("type"))
            if key not in seen_e:
                combined["entities"].append(ent)
                seen_e.add(key)

        for rel in data.get("relationships", []):
            if not isinstance(rel, dict):
                continue
            key = (rel.get("source"), rel.get("target"), rel.get("type"))
            if key not in seen_r:
                combined["relationships"].append(rel)
                seen_r.add(key)

    return combined

def combine_chunks_with_hash(folder: Path) -> Dict[str, Any]:
    """Combine all chunk JSONs in one folder, adding a chunk_hash property."""
    result = {"entities": [], "relationships": []}
    for chunk in sorted(folder.glob("*.json")):
        chunk_hash = hashlib.sha256(chunk.name.encode()).hexdigest()
        try:
            data = json.loads(chunk.read_text())
        except json.JSONDecodeError:
            logging.warning("Skipping malformed chunk %s", chunk)
            continue
        for ent in data.get("entities", []):
            # ensure entity is dict
            if not isinstance(ent, dict):
                continue
            props = ent.setdefault("properties", {})
            props["chunk_hash"] = chunk_hash
            result["entities"].append(ent)
        for rel in data.get("relationships", []):
            # include relationship unchanged (no chunk_hash)
            if not isinstance(rel, dict):
                continue
            result["relationships"].append(rel)
    return result

def combine_all_docs(root: Path) -> Dict[str, Any]:
    """Combine per-document JSONs in a directory, adding doc_hash to each entry."""
    final = {"entities": [], "relationships": []}
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        doc_hash = hashlib.sha256(sub.name.encode()).hexdigest()
        per_doc = combine_chunks_with_hash(sub)
        for ent in per_doc.get("entities", []):
            if not isinstance(ent, dict):
                continue
            props = ent.setdefault("properties", {})
            props["doc_hash"] = doc_hash
            final["entities"].append(ent)
        for rel in per_doc.get("relationships", []):
            # include relationship unchanged (no doc_hash)
            if not isinstance(rel, dict):
                continue
            final["relationships"].append(rel)
    return final


# Configuration (edit these paths explicitly)
TRIPLETS_ROOT = Path(os.getenv("TRIPLETS_DIR", "./data/triplets"))
FINAL_OUT_PATH = Path(os.getenv("MERGED_JSON_PATH", "./data/merged.json"))


def main():
    # Combine all per-folder JSONs into one final merged JSON
    final = combine_all_docs(TRIPLETS_ROOT)
    FINAL_OUT_PATH.write_text(json.dumps(final, indent=2))
    logger.info("Final merged JSON written to %s", FINAL_OUT_PATH)


if __name__ == "__main__":
    main() 