#!/usr/bin/env python
"""local_community_enrichment.py – end-to-end community workflow

1. Re-uses hierarchical Leiden clustering from `local_community_detection.py`.
2. Computes deterministic per-community features (size, density, top centralities, dominant edge types).
3. Generates an LLM report (title + summary + key entities) for the N largest leaf communities.
4. Writes hierarchy and reports back into Neo4j both as:
   • properties on entity nodes (community_levels, community_leaf)
   • explicit (:Community) nodes with (:BELONGS_TO) and (:HAS_PARENT) relations.
5. Saves artifacts (features, reports, stats) under an output directory.

Edit the CONFIG section at the top and run:
    python local_community_enrichment.py
"""
from __future__ import annotations

import json
import os
import time
import csv
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import networkx as nx
from graspologic.partition import hierarchical_leiden
from neo4j import GraphDatabase, Driver
from openai import OpenAI, RateLimitError, APIError

# ---------------------------------------------------------------------------
# CONFIG – explicit values (no environment variables)
# ---------------------------------------------------------------------------
BOLT_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "")

OPENAI_API_KEY        = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL          = "gpt-4o-mini"            # OpenAI model
OPENAI_RATE_SLEEP     = 2                          # seconds between LLM calls
OPENAI_MAX_RETRY      = 4                          # retry on failures
LLM_COMMUNITIES_LIMIT = 100                        # max leaf clusters to summarise
LLM_TOP_K_ENTITIES    = 10                         # top-K entity briefs
LLM_CONTEXT_CHARS     = 8000                       # chars max for prompt context
PARENT_CONTEXT_CHARS  = 12000                      # parent summary context limit (will chunk if exceeded)

MAX_CLUSTER_SIZE      = 50                         # Leiden resolution parameter
USE_LCC               = True                       # restrict to largest CC first
SEED                  = 42                         # random seed

OUTPUT_DIR            = Path(os.getenv("OUTPUT_DIR", "./output/community_enrichment"))
FEATURES_SUBDIR       = "features"
REPORTS_SUBDIR        = "reports"
REPORT_INDEX_FILENAME = "reports_index.json"
NODE_MAPPING_FILE     = "node_to_comms.json"
CLUSTERS_FILE         = "clusters.json"
STATS_FILE            = "stats.json"
CLUSTERS_CSV_FILE     = "clusters.csv"
NODE_COMM_CSV_FILE    = "node_communities.csv"
EDGES_BY_COMM_FILE    = "edges_by_community.json"
GEXF_FILENAME         = "graph_with_communities.gexf"
# ---------------------------------------------------------------------------

client = OpenAI(api_key=OPENAI_API_KEY)

# ---------------------------------------------------------------------------
# Helper utilities  ---------------------------------------------------------
# ---------------------------------------------------------------------------

def stable_graph(graph: nx.Graph) -> nx.Graph:
    """Return graph with stable node/edge order (copy of logic from original script)."""
    fixed = nx.Graph()
    for n, data in sorted(graph.nodes(data=True), key=lambda x: x[0]):
        fixed.add_node(n, **data)
    edges = list(graph.edges(data=True))
    edges = [(min(u, v), max(u, v), d) for u, v, d in edges]
    edges.sort(key=lambda e: f"{e[0]}->{e[1]}")
    fixed.add_edges_from(edges)
    return fixed


def largest_cc(graph: nx.Graph) -> nx.Graph:
    from graspologic.utils import largest_connected_component
    return stable_graph(largest_connected_component(graph))

# ---------------------------------------------------------------------------
# Neo4j helpers (read + write) ----------------------------------------------
# ---------------------------------------------------------------------------

def fetch_graph(uri: str, user: str, pw: str) -> Tuple[nx.Graph, Dict[str,str], Dict[str,str]]:
    driver: Driver = GraphDatabase.driver(uri, auth=(user, pw))
    with driver.session() as sess:
        nodes = sess.run(
            """
            MATCH (n) RETURN elementId(n) AS id, coalesce(n.name,n.id,'') AS name, coalesce(n.brief,'') AS brief
            """
        ).data()
        rels = sess.run(
            """
            MATCH (a)-[r]-(b) WHERE elementId(a) < elementId(b) RETURN elementId(a) AS s, elementId(b) AS t, type(r) AS type
            """
        ).data()
    G = nx.Graph()
    name_lu, brief_lu = {}, {}
    for n in nodes:
        G.add_node(n["id"], name=n["name"], brief=n["brief"])
        name_lu[n["id"]] = n["name"] or n["id"]
        brief_lu[n["id"]] = n["brief"]
    for r in rels:
        if not G.has_edge(r["s"], r["t"]):
            G.add_edge(r["s"], r["t"], rel_type=r["type"])
    driver.close()
    return G, name_lu, brief_lu


def write_back_properties(driver: Driver, node_to_comms: Dict[str, List[int]]):
    rows = [{"id": nid, "levels": comms, "leaf": comms[-1]} for nid, comms in node_to_comms.items()]
    query = (
        "UNWIND $rows AS row "
        "MATCH (n) WHERE elementId(n)=row.id "
        "SET n.community_levels = row.levels, n.community_leaf = row.leaf"
    )
    with driver.session() as sess:
        sess.run(query, rows=rows)


def write_back_community_nodes(
    driver: Driver,
    communities: List[Tuple[int,int,int,List[str]]],
    comm_props: Dict[Tuple[int,int], Dict[str,str]] | None = None,
    leaf_keys: set[Tuple[int,int]] | None = None,
):
    # prepare community nodes and HAS_PARENT relationships
    comm_rows, parent_rows, belong_rows = [], [], []
    comm_props = comm_props or {}
    leaf_keys = leaf_keys or set()
    for lvl, cid, parent, members in communities:
        props = comm_props.get((lvl, cid), {})
        comm_rows.append({
            "level": lvl,
            "cid": cid,
            "size": len(members),
            "is_leaf": (lvl, cid) in leaf_keys,
            "title": props.get("title"),
            "description": props.get("description"),
        })
        if parent != -1:
            parent_rows.append({"level": lvl, "cid": cid, "parent": parent})
        if (lvl, cid) in leaf_keys:
            for nid in members:
                belong_rows.append({"id": nid, "level": lvl, "cid": cid})

    with driver.session() as sess:
        # create community nodes
        result1 = sess.run(
            """
            UNWIND $rows AS row
            MERGE (c:Community {level:row.level, cid:row.cid})
            ON CREATE SET c.size = row.size,
                          c.is_leaf = row.is_leaf,
                          c.title = coalesce(row.title, c.title),
                          c.description = coalesce(row.description, c.description)
            ON MATCH  SET c.size = row.size,
                          c.is_leaf = row.is_leaf,
                          c.title = coalesce(row.title, c.title),
                          c.description = coalesce(row.description, c.description)
            """,
            rows=comm_rows,
        )
        print(f"[neo4j] upserted Community nodes: {len(comm_rows)}")
        # hierarchy
        if parent_rows:
            result2 = sess.run(
                """
                UNWIND $rows AS row
                MATCH (c:Community {level:row.level, cid:row.cid})
                MATCH (p:Community {level:row.level-1, cid:row.parent})
                MERGE (c)-[:HAS_PARENT]->(p)
                """,
                rows=parent_rows,
            )
            print(f"[neo4j] created HAS_PARENT rels: {len(parent_rows)}")
        # BELONGS_TO edges
        result3 = sess.run(
            """
            UNWIND $rows AS row
            MATCH (n) WHERE elementId(n)=row.id
            MATCH (c:Community {level:row.level, cid:row.cid})
            MERGE (n)-[:BELONGS_TO {level:row.level}]->(c)
            """,
            rows=belong_rows,
        )
        print(f"[neo4j] created BELONGS_TO rels: {len(belong_rows)}")

# ---------------------------------------------------------------------------
# Community analytics --------------------------------------------------------
# ---------------------------------------------------------------------------

def community_features(G: nx.Graph, members: List[str]) -> Dict[str, Any]:
    sub = G.subgraph(members)
    degs = dict(sub.degree())
    betw = nx.betweenness_centrality(sub, normalized=True)
    # Use top-10 nodes by degree and betweenness for richer summaries
    top_deg = sorted(degs, key=degs.get, reverse=True)[:10]
    top_betw = sorted(betw, key=betw.get, reverse=True)[:10]
    edge_hist = Counter(d["rel_type"] for _, _, d in sub.edges(data=True)).most_common(5)
    return {
        "size": sub.number_of_nodes(),
        "density": nx.density(sub),
        "top_degree": top_deg,
        "top_betweenness": top_betw,
        "edge_types": edge_hist,
    }


# ---------------------------------------------------------------------------
# LLM summarisation ----------------------------------------------------------
# ---------------------------------------------------------------------------

def summarise_cluster(feat: Dict[str,Any], entity_briefs: Dict[str,str]) -> Dict[str,str]:
    """Call OpenAI to summarise one community using top-20 entities (10 by degree + 10 by betweenness).

    The summary should be holistic and derived from the full briefs/descriptions of these entities only.
    Edge-type information is intentionally excluded from the prompt context.
    """
    top_degree: List[str] = feat.get("top_degree", [])[:10]
    top_betw: List[str] = feat.get("top_betweenness", [])[:10]
    ordered_entities: List[str] = []
    for eid in top_degree + top_betw:
        if eid not in ordered_entities:
            ordered_entities.append(eid)
    # Build full-context lines (no truncation)
    brief_lines = [f"• {e}: {entity_briefs.get(e,'')}" for e in ordered_entities]
    context = (
        f"Community size: {feat['size']}, density: {feat['density']:.3f}.\n"
        f"Entity set for summary (degree ∪ betweenness, up to 20): {', '.join(ordered_entities)}\n\n"
        "Entity briefs/descriptions (full):\n" + "\n".join(brief_lines)
    )
    prompt = (
        "You are an equity research analyst. Using ONLY the entity briefs/descriptions provided, "
        "write a JSON with fields title and executive_summary. The executive_summary should be a holistic "
        "2-3 sentence theme of this community inferred from these entities; do NOT mention data provenance, "
        "edge types, or missing data.\n\n"
        f"CONTEXT:\n{context}\n\nJSON:"
    )

    for attempt in range(OPENAI_MAX_RETRY):
        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content)
            return {"title": data.get("title",""), "executive_summary": data.get("executive_summary","")}
        except (RateLimitError, APIError) as e:
            wait = OPENAI_RATE_SLEEP * (attempt + 1)
            print(f"[OpenAI] Rate/connection error, retrying in {wait}s … {e}")
            time.sleep(wait)
    return {"title": "", "executive_summary": ""}

# ---------------------------------------------------------------------------
# Main pipeline --------------------------------------------------------------
# ---------------------------------------------------------------------------

def run_pipeline():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Fetching graph from Neo4j …")
    G_raw, name_lu, brief_lu = fetch_graph(BOLT_URI, USERNAME, PASSWORD)
    print(f"Graph: {G_raw.number_of_nodes():,} nodes, {G_raw.number_of_edges():,} edges")

    G_proc = largest_cc(G_raw) if USE_LCC else stable_graph(G_raw)

    print("Running hierarchical Leiden clustering …")
    parts = hierarchical_leiden(G_proc, max_cluster_size=MAX_CLUSTER_SIZE, random_seed=SEED)

    # Build community tuples
    communities: List[Tuple[int,int,int,List[str]]] = []
    level_map: Dict[int, Dict[str,int]] = {}
    parent_map: Dict[int,int] = {}
    for p in parts:
        level_map.setdefault(p.level, {})[p.node] = p.cluster
        parent_map[p.cluster] = p.parent_cluster if p.parent_cluster is not None else -1
    for level, m in level_map.items():
        by_cluster: Dict[int,List[str]] = {}
        for n,cid in m.items():
            by_cluster.setdefault(cid, []).append(n)
        for cid, members in by_cluster.items():
            communities.append((level, cid, parent_map[cid], members))
    # logging: communities per level
    per_level = {}
    for lvl, cid, parent, members in communities:
        per_level.setdefault(lvl, 0)
        per_level[lvl] += 1
    print(f"[cluster] communities per level: {per_level}")

    # Node → list of communities across levels
    max_lvl = max(level_map)
    node_to_comms = {n: [-1]*(max_lvl+1) for n in G_proc.nodes()}
    for lvl, cid, _, members in communities:
        for n in members:
            node_to_comms[n][lvl] = cid

    # ------------------------------------------------------------------
    # Feature extraction + entity-based LLM summaries for communities that
    # have direct entity members and NO child communities (regardless of level)
    # ------------------------------------------------------------------
    feat_dir = OUTPUT_DIR / "features"; feat_dir.mkdir(exist_ok=True)
    report_dir = OUTPUT_DIR / "reports"; report_dir.mkdir(exist_ok=True)

    # Build child lists for every parent (all levels)
    children_of: Dict[Tuple[int,int], List[Tuple[int,int]]] = {}
    for lvl, cid, parent, _ in communities:
        if parent != -1:
            children_of.setdefault((lvl-1, parent), []).append((lvl, cid))

    # Summaries and quick index for later parent roll-ups
    reports_summary: List[Dict[str, Any]] = []
    comm_entity_summary: Dict[Tuple[int,int], Dict[str,str]] = {}

    # Communities with no child communities
    leaf_like = [c for c in communities if (c[0], c[1]) not in children_of]
    leaf_like.sort(key=lambda t: len(t[3]), reverse=True)

    leaf_key_set: set[Tuple[int,int]] = set()
    for idx, (lvl,cid,parent,members) in enumerate(leaf_like):
        feat = community_features(G_proc, members)
        (feat_dir / f"L{lvl}_C{cid}.json").write_text(json.dumps(feat,indent=2))

        report = {}
        if idx < LLM_COMMUNITIES_LIMIT:
            print(f"[entity-summary] idx={idx} L{lvl} C{cid} size={len(members)} …")
            report = summarise_cluster(feat, brief_lu)
            time.sleep(OPENAI_RATE_SLEEP)
            (report_dir / f"L{lvl}_C{cid}.json").write_text(json.dumps(report, indent=2))
        reports_summary.append({"level":lvl, "cid":cid, "size":feat["size"], **report})
        if report:
            comm_entity_summary[(lvl,cid)] = report
        leaf_key_set.add((lvl,cid))

    (OUTPUT_DIR / "reports_index.json").write_text(json.dumps(reports_summary, indent=2))

    # ------------------------------------------------------------------
    # Iterative roll-ups for all parent levels (e.g., L1, then L0, …)
    # ------------------------------------------------------------------
    # Map: (level,cid) -> entity-based executive summary text (for communities with direct entity members)
    entity_summary_map: Dict[Tuple[int,int], str] = {}
    for r in reports_summary:
        if r.get("title") or r.get("executive_summary"):
            entity_summary_map[(r["level"], r["cid"])] = (r.get("executive_summary") or r.get("title") or "").strip()

    parent_props: Dict[Tuple[int,int], Dict[str,str]] = {}

    def _summarise_parent(parent_key: Tuple[int,int], child_texts: List[str]) -> Dict[str,str]:
        # chunk
        payload, block = [], ''
        for s in child_texts:
            if len(block) + len(s) + 2 <= PARENT_CONTEXT_CHARS:
                block += (s + '\n')
            else:
                payload.append(block)
                block = s + '\n'
        if block:
            payload.append(block)
        # call LLM
        inferences: List[str] = []
        for i, chunk in enumerate(payload):
            print(f"[parent] L{parent_key[0]} C{parent_key[1]} chunk {i+1}/{len(payload)} len={len(chunk)}")
            prompt = (
                "You are an equity research analyst. You are given a set of child community summaries.\n"
                "Derive at most five concise bullet inferences capturing the common themes.\n"
                "Return JSON: {\"inferences\": [\"...\"]}.\n\n" + chunk
            )
            for attempt in range(OPENAI_MAX_RETRY):
                try:
                    resp = client.chat.completions.create(
                        model=OPENAI_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        response_format={"type": "json_object"},
                    )
                    data = json.loads(resp.choices[0].message.content)
                    inferences.extend(data.get("inferences", [])[:5])
                    time.sleep(OPENAI_RATE_SLEEP)
                    break
                except (RateLimitError, APIError) as e:
                    wait = OPENAI_RATE_SLEEP * (attempt + 1)
                    print(f"[OpenAI] Parent roll-up retry in {wait}s … {e}")
                    time.sleep(wait)
        # dedupe and cap at 5
        seen, uniq = set(), []
        for s in inferences:
            if s not in seen:
                seen.add(s)
                uniq.append(s)
            if len(uniq) >= 5:
                break
        return {"title": " | ".join(uniq)[:48], "description": "\n".join(uniq)} if uniq else {}

    # roll-up from max_lvl-1 down to 0
    for lvl in range(max_lvl - 1, -1, -1):
        # parents at this level
        keys = [k for k in children_of.keys() if k[0] == lvl]
        for parent_key in keys:
            kids = children_of.get(parent_key, [])
            print(f"[parent] building roll-up for parent={parent_key} children={len(kids)}")
            texts: List[str] = []
            for kid in kids:
                # prefer already-computed parent description of child; else entity-based summary
                child_desc = parent_props.get(kid, {}).get("description")
                if not child_desc:
                    child_desc = entity_summary_map.get(kid)
                if child_desc:
                    texts.append(child_desc)
            if not texts:
                continue
            props = _summarise_parent(parent_key, texts)
            if props:
                parent_props[parent_key] = props
                print(f"[parent] L{parent_key[0]} C{parent_key[1]} inferences={len(props['description'].splitlines())}")

    # ------------------------------------------------------------------
    # Write back to Neo4j
    # ------------------------------------------------------------------
    driver = GraphDatabase.driver(BOLT_URI, auth=(USERNAME, PASSWORD))
    print("Writing properties back to Neo4j …")
    write_back_properties(driver, node_to_comms)
    print("Creating Community nodes + relationships …")
    # Merge leaf entity-based props with parent roll-up props so both get written
    combined_props: Dict[Tuple[int,int], Dict[str,str]] = {}
    combined_props.update({k: {"title": v.get("title",""), "description": v.get("executive_summary","")} for k, v in comm_entity_summary.items()})
    combined_props.update(parent_props)
    write_back_community_nodes(driver, communities, combined_props, leaf_key_set)
    driver.close()

    # Save mapping + stats locally
    (OUTPUT_DIR / "node_to_comms.json").write_text(json.dumps(node_to_comms, indent=2))
    print(f"All artifacts saved under {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    run_pipeline()

