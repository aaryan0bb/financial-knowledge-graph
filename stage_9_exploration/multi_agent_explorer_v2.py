#!/usr/bin/env python3
"""
multi_agent_explorer_v2.py – Simplified Multi-Agent Community Explorer

This is a cleaner, more modular implementation that follows LangGraph best practices.

Workflow:
1. Query → Semantic similarity → Top K leaf communities
2. LLM filters communities to investigate  
3. Parallel community agents explore entities + neighbors
4. Shared knowledge prevents going off-track
5. Final synthesis across all findings

Usage:
    python multi_agent_explorer_v2.py
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Tuple
import math
import os
from dataclasses import dataclass
from enum import Enum

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from voyageai import Client  # type: ignore

from neo4j import GraphDatabase
from openai import OpenAI
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from pathlib import Path

# ---------------------------------------------------------------------------
# CACHING & ENTITY EMBEDDING HELPER
# ---------------------------------------------------------------------------
CACHE_DIR = Path(__file__).parent / "cache"
ENTITY_EMBEDDINGS_FILE = CACHE_DIR / "entities_embeddings.npz"
ENTITY_ITEMS_FILE = CACHE_DIR / "entities_items.json"
CACHE_DIR.mkdir(exist_ok=True)

class EntityEmbeddingHelper:
    """Helper for embedding and indexing entities with caching"""
    def __init__(self, model_name: str):
        if model_name.startswith("voyage-"):
            self.client = Client(VOYAGEAI_API_KEY)
            self.model_name = model_name
            self.use_voyage = True
        else:
            self.model = SentenceTransformer(model_name)
            self.use_voyage = False
        self.items: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.index: Optional[faiss.IndexFlatIP] = None

    def load_or_build(self, entities: List[Dict[str, Any]]):
        if ENTITY_EMBEDDINGS_FILE.exists() and ENTITY_ITEMS_FILE.exists():
            data = np.load(str(ENTITY_EMBEDDINGS_FILE))
            self.embeddings = data['embeddings']
            self.items = json.loads(ENTITY_ITEMS_FILE.read_text())
        else:
            self.items = entities
            texts = []
            for e in entities:
                name = e.get('name','') or ''
                desc = e.get('description','') or ''
                texts.append(f"{name}. {desc}".strip())
            if self.use_voyage:
                all_embeddings = []
                batch_size = 1000
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i+batch_size]
                    embeds = self.client.embed(batch, model=self.model_name, input_type="query", truncation=True).embeddings  # type: ignore
                    all_embeddings.extend(embeds)
                vectors = np.asarray(all_embeddings, dtype="float32")
                faiss.normalize_L2(vectors)
                self.embeddings = vectors
            else:
                emb = self.model.encode(texts, normalize_embeddings=True)
                self.embeddings = emb.astype('float32')
            np.savez(str(ENTITY_EMBEDDINGS_FILE), embeddings=self.embeddings)
            ENTITY_ITEMS_FILE.write_text(json.dumps(self.items))
        dim = self.embeddings.shape[1]
        idx = faiss.IndexFlatIP(dim)
        idx.add(self.embeddings)
        self.index = idx

    def topk(self, query: str, k: int) -> List[Tuple[Dict[str, Any], float]]:
        if self.index is None:
            raise RuntimeError("Entity index not built")
        if self.use_voyage:
            embeds = self.client.embed([query], model=self.model_name, input_type="query", truncation=True).embeddings  # type: ignore
            qv = np.asarray(embeds, dtype="float32")
            faiss.normalize_L2(qv)
        else:
            qv = self.model.encode([query], normalize_embeddings=True).astype('float32')
        scores, idxs = self.index.search(qv, k)
        out: List[Tuple[Dict[str, Any], float]] = []
        for score, i in zip(scores[0], idxs[0]):
            if 0 <= i < len(self.items):
                out.append((self.items[i], float(score)))
        return out
# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-4o-mini"

# Embedding model for community similarity (sentence-transformers)
# Voyage finance embedding model
#EMBEDDING_MODEL = "voyage-finance-2"
EMBEDDING_MODEL = "voyage-3-large"

VOYAGEAI_API_KEY = os.getenv("VOYAGE_API_KEY", "")
# 🔧 ADJUSTABLE PARAMETERS
TOP_K_COMMUNITIES = 5
MAX_COMMUNITIES_TO_INVESTIGATE = 4 # Adjust this based on your needs
TOP_K_ENTITIES_PER_COMMUNITY =  10 # Adjust this based on your needs
SIMILARITY_THRESHOLD = 0.6          # Adjust this based on your needs
MAX_EXPLORATION_DEPTH = 2
RATE_LIMIT_SLEEP = 1
ENTITY_PATH_K = 20  # Number of top entities to explore independently

# Output directory (each run gets its own timestamped folder)
OUTPUT_ROOT = Path(os.getenv("OUTPUT_DIR", "./output/multi_agent_runs"))

# ---------------------------------------------------------------------------
# DATA MODELS
# ---------------------------------------------------------------------------

class AgentState(BaseModel):
    """Global state for the multi-agent workflow"""
    query: str
    
    # Phase 1: Community Discovery
    candidate_communities: List[Dict[str, Any]] = Field(default_factory=list)
    selected_communities: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Phase 2: Parallel Exploration
    exploration_results: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    shared_insights: List[str] = Field(default_factory=list)
    explored_entities: List[str] = Field(default_factory=list)
    
    # Phase 3: Final Answer
    final_answer: str = ""
    error: Optional[str] = None

# ---------------------------------------------------------------------------
# UTILITY CLASSES
# ---------------------------------------------------------------------------

class Neo4jClient:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    def close(self):
        self.driver.close()
    
    def get_leaf_communities(self) -> List[Dict[str, Any]]:
        """Get all leaf communities as written by local_community_enrichment.py (c.is_leaf=true)."""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (c:Community)
                WHERE c.is_leaf = true
                RETURN c.level AS level,
                       c.cid   AS cid,
                       coalesce(c.title, '') AS title,
                       coalesce(c.description, '') AS description,
                       c.size  AS size
                ORDER BY size DESC
                """
            )
            return [dict(record) for record in result]
    
    def get_community_entities(self, level: int, cid: int) -> List[Dict[str, Any]]:
        """Entities are those linked to the community via BELONGS_TO; rank by in-community degree.

        Mirrors the leaf semantics in local_community_enrichment.py where only leaf communities create BELONGS_TO.
        Degree counts relationships to other nodes that also BELONG_TO the same (level,cid).
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (n)-[:BELONGS_TO]->(c:Community {level: $level, cid: $cid})
                OPTIONAL MATCH (n)-[r]-(m)
                WHERE (m)-[:BELONGS_TO]->(c)
                WITH n, count(r) AS degree
                RETURN elementId(n) AS id,
                       coalesce(n.name, '') AS name,
                       coalesce(n.brief, '') AS brief,
                       coalesce(n.description, '') AS description,
                       degree
                ORDER BY degree DESC, name ASC
                LIMIT 100
                """,
                level=level,
                cid=cid,
            )
            rows = [dict(record) for record in result]
            # Ensure numeric degree
            for row in rows:
                row.setdefault("degree", 0)
            return rows
    
    def get_entity_neighbors(self, entity_id: str, max_neighbors: int = 3) -> List[Dict[str, Any]]:
        """Get neighbors of an entity"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (n)-[r]-(neighbor)
                WHERE elementId(n) = $entity_id
                RETURN DISTINCT elementId(neighbor) as id,
                       coalesce(neighbor.name, '') as name,
                       coalesce(neighbor.brief, '') as brief,
                       coalesce(neighbor.description, '') as description,
                       type(r) as relationship_type,
                       properties(r) as edge_props
                LIMIT $max_neighbors
            """, entity_id=entity_id, max_neighbors=max_neighbors)
            return [dict(record) for record in result]
    def get_all_entities(self) -> List[Dict[str, Any]]:
        """Get all non-community nodes as entities with id, name, brief, description"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (n)
                WHERE NOT n:Community
                RETURN elementId(n) AS id,
                       coalesce(n.name, '') AS name,
                       '' AS brief,
                       coalesce(n.content, '') AS description
                """
            )
            return [dict(record) for record in result]

class LLMClient:
    def _parse_json_fallback(self, content: str) -> Dict[str, Any]:
        """Best-effort JSON fallback extractor for resilience."""
        try:
            return json.loads(content)
        except Exception:
            pass
        # crude fallback: look for a score and insights lines
        result: Dict[str, Any] = {}
        try:
            import re
            score_match = re.search(r"relevance[_\s-]*score\D+([01](?:\.\d+)?)", content, flags=re.IGNORECASE)
            if score_match:
                result["relevance_score"] = float(score_match.group(1))
        except Exception:
            pass
        return result
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
    
    def filter_communities(self, query: str, communities: List[Dict[str, Any]]) -> List[int]:
        """Use LLM to select which communities to investigate"""
        community_descriptions = []
        for i, comm in enumerate(communities):
            desc = f"{i}. Community L{comm['level']}_C{comm['cid']} (size: {comm['size']})\n"
            desc += f"   Title: {comm['title']}\n"
            desc += f"   Description: {comm['description'][:150]}...\n"
            community_descriptions.append(desc)
        
        prompt = f"""
Query: "{query}"

Available Communities:
{chr(10).join(community_descriptions)}

Select up to {MAX_COMMUNITIES_TO_INVESTIGATE} communities most relevant to the query.
Consider relevance, size, and diversity.

Return JSON: {{"selected_indices": [0, 1], "reasoning": "explanation"}}
        """
        
        try:
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            raw_content = response.choices[0].message.content
            try:
                result = json.loads(raw_content)
            except Exception:
                result = self._parse_json_fallback(raw_content)
            return result.get("selected_indices", [])[:MAX_COMMUNITIES_TO_INVESTIGATE]
        except Exception as e:
            print(f"LLM filtering failed: {e}")
            return list(range(min(MAX_COMMUNITIES_TO_INVESTIGATE, len(communities))))
    
    def analyze_entity_relevance(self, query: str, entity: Dict[str, Any], context: str = "") -> Tuple[float, List[str]]:
        """Analyze entity relevance to query"""
        prompt = f"""
 Query: "{query}"
 Entity: {entity['name']}
 Entity Description (verbatim): {entity.get('brief', '')} {entity.get('description', '')}
 Context (neighbors/community): {context}
 
 Task: Assess how relevant this entity is to answering the query using BOTH the entity description and the provided context. 
 - Read both carefully and synthesize.
 - Prefer specific, defensible insights grounded in the text.
 - If a point comes from context vs entity, reflect that implicitly in wording.
 
 Return JSON:
 {{
   "relevance_score": 0.0-1.0,
   "insights": ["concise, evidence-based insight 1", "insight 2", "insight 3"]
 }}
         """
        
        try:
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            raw_content = response.choices[0].message.content
            try:
                result = json.loads(raw_content)
            except Exception:
                result = self._parse_json_fallback(raw_content)

            raw_score = result.get("relevance_score", 0.5)
            try:
                score = float(raw_score)
            except Exception:
                score = 0.5
            # clamp to [0.0, 1.0]
            if not math.isfinite(score):
                score = 0.5
            score = max(0.0, min(1.0, score))

            insights = result.get("insights", [])
            if isinstance(insights, str):
                insights = [insights]
            if not isinstance(insights, list):
                insights = []

            return score, insights
        except Exception as e:
            print(f"Entity analysis failed: {e}")
            return 0.5, []
    
    def synthesize_answer(self, query: str, exploration_results: Dict[str, Any]) -> str:
        """Create final answer from exploration results"""
        summaries = []
        for comm_id, results in exploration_results.items():
            if results.get("insights"):
                summary = f"Community {comm_id}: {'; '.join(results['insights'][:3])}"
                summaries.append(summary)
        
        prompt = f"""
Query: "{query}"

Exploration Results:
{chr(10).join(summaries)}

Provide a comprehensive answer based on these findings.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Analysis completed. Found insights across {len(exploration_results)} communities."

class EmbeddingHelper:
    """Small helper around sentence-transformers + FAISS for community similarity."""
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        if model_name.startswith("voyage-"):
            self.client = Client(VOYAGEAI_API_KEY)
            self.model_name = model_name
            self.use_voyage = True
        else:
            self.model = SentenceTransformer(model_name)
            self.use_voyage = False
        self.index = None
        self.items: List[Dict[str, Any]] = []

    def build_index(self, communities: List[Dict[str, Any]]):
        self.items = communities
        texts: List[str] = []
        for c in communities:
            title = (c.get("title", "") or "").strip()
            desc = (c.get("description", "") or "").strip()
            if not title and not desc:
                texts.append(f"Community L{c.get('level')}_C{c.get('cid')}")
            else:
                texts.append(f"{title}. {desc}".strip())
        if self.use_voyage:
            all_embeddings = []
            batch_size = 1000
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                embeds = self.client.embed(batch, model=self.model_name, input_type="query", truncation=True).embeddings  # type: ignore
                all_embeddings.extend(embeds)
            emb = np.asarray(all_embeddings, dtype="float32")
            faiss.normalize_L2(emb)
        else:
            emb = self.model.encode(texts, normalize_embeddings=True).astype("float32")
        dim = emb.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(emb)

    def topk(self, query: str, k: int) -> List[Tuple[Dict[str, Any], float]]:
        if self.index is None:
            raise RuntimeError("Embedding index not built")
        if self.use_voyage:
            embeds = self.client.embed([query], model=self.model_name, input_type="query", truncation=True).embeddings  # type: ignore
            q = np.asarray(embeds, dtype="float32")
            faiss.normalize_L2(q)
        else:
            q = self.model.encode([query], normalize_embeddings=True).astype("float32")
        scores, idxs = self.index.search(q, k)
        out: List[Tuple[Dict[str, Any], float]] = []
        for s, i in zip(scores[0], idxs[0]):
            if 0 <= i < len(self.items):
                out.append((self.items[i], float(s)))
        return out

class CommunityExplorer:
    """Individual agent for exploring one community"""
    
    def __init__(self, community: Dict[str, Any], neo4j_client: Neo4jClient, llm_client: LLMClient):
        self.community = community
        self.neo4j = neo4j_client
        self.llm = llm_client
        self.comm_id = f"L{community['level']}_C{community['cid']}"
        
    def explore(self, query: str, shared_state: Dict[str, Any]) -> Dict[str, Any]:
        """Explore this community"""
        print(f"[{self.comm_id}] Starting exploration...")
        
        results = {
            "community_id": self.comm_id,
            "entities_explored": [],
            "insights": [],
            "neighbors_found": [],
            "exploration_path": []
        }
        
        try:
            # Get entities in this community
            entities = self.neo4j.get_community_entities(
                self.community['level'], 
                self.community['cid']
            )
            
            # Select top entities by in-community degree
            entities.sort(key=lambda e: (e.get("degree", 0), e.get("name", "")), reverse=True)
            top_entities = entities[:TOP_K_ENTITIES_PER_COMMUNITY]
            
            explored_ids_set = set(shared_state.get('explored_entities', []))
            for entity in top_entities:
                if entity['id'] in explored_ids_set:
                    continue
                
                print(f"[{self.comm_id}] Exploring entity: {entity['name']}")
                
                # Fetch neighbors first to enrich relevance context
                neighbors = self.neo4j.get_entity_neighbors(entity['id'])
                neighbor_summaries = []
                for n in neighbors[:5]:
                    nb = (n.get('brief', '') or '').strip()
                    nd = (n.get('description', '') or '').strip()
                    neighbor_summaries.append(f"{n.get('name','')}: {nb} {nd}".strip())
                enriched_context = (self.community.get('description', '') or '').strip()
                if neighbor_summaries:
                    enriched_context = f"{enriched_context}\nNeighbors:\n- " + "\n- ".join(neighbor_summaries)
                
                # Analyze relevance with neighbor briefs/descriptions in context
                relevance, insights = self.llm.analyze_entity_relevance(
                    query, entity, enriched_context
                )
                # ensure numeric relevance for safe comparison
                try:
                    relevance_value = float(relevance)
                except Exception:
                    relevance_value = 0.5
                if not math.isfinite(relevance_value):
                    relevance_value = 0.5
                relevance_value = max(0.0, min(1.0, relevance_value))

                if relevance_value < SIMILARITY_THRESHOLD:
                    print(f"[{self.comm_id}] Entity {entity['name']} not relevant enough ({relevance_value:.3f})")
                    continue
                
                # Add to results
                results["entities_explored"].append({
                    "name": entity['name'],
                    "relevance": relevance_value,
                    "insights": insights
                })
                results["insights"].extend(insights)
                results["exploration_path"].append(entity['name'])
                
                # Reuse fetched neighbors
                results["neighbors_found"].extend([n['name'] for n in neighbors])
                
                # Update shared state
                shared_state.setdefault('explored_entities', [])
                if entity['id'] not in explored_ids_set:
                    shared_state['explored_entities'].append(entity['id'])
                    explored_ids_set.add(entity['id'])
                shared_state.setdefault('shared_insights', []).extend(insights)
                
                time.sleep(RATE_LIMIT_SLEEP)
                
        except Exception as e:
            print(f"[{self.comm_id}] Exploration error: {e}")
            results["error"] = str(e)
        
        print(f"[{self.comm_id}] Completed: {len(results['entities_explored'])} entities, {len(results['insights'])} insights")
        return results

# ---------------------------------------------------------------------------
# LANGGRAPH WORKFLOW NODES
# ---------------------------------------------------------------------------

def community_discovery_node(state: AgentState) -> AgentState:
    """Phase 1: Discover and rank communities by similarity"""
    print(f"[Phase 1] Community Discovery for query: '{state.query}'")
    
    neo4j_client = None
    try:
        neo4j_client = Neo4jClient()
        
        # Get all leaf communities
        communities = neo4j_client.get_leaf_communities()
        print(f"[Phase 1] Found {len(communities)} leaf communities")
        
        if not communities:
            state.error = "No leaf communities found"
            return state

        # Embedding similarity with FAISS (cosine via normalized inner-product)
        embedder = EmbeddingHelper(EMBEDDING_MODEL)
        embedder.build_index(communities)
        ranked = embedder.topk(state.query, TOP_K_COMMUNITIES)
        # attach similarity_score
        cand = []
        for item, score in ranked:
            item = dict(item)
            item["similarity_score"] = score
            cand.append(item)
        state.candidate_communities = cand
        
        print(f"[Phase 1] Selected top {len(state.candidate_communities)} candidates (embedding-based)")
        
    except Exception as e:
        state.error = f"Community discovery failed: {e}"
        print(f"[ERROR] {state.error}")
    finally:
        if neo4j_client is not None:
            try:
                neo4j_client.close()
            except Exception:
                pass
    
    return state

def community_filtering_node(state: AgentState) -> AgentState:
    """Phase 2: LLM filters communities for investigation"""
    print(f"[Phase 2] Community Filtering")
    
    if state.error or not state.candidate_communities:
        return state
    
    try:
        llm_client = LLMClient()
        
        selected_indices = llm_client.filter_communities(state.query, state.candidate_communities)
        
        state.selected_communities = [
            state.candidate_communities[i] 
            for i in selected_indices 
            if i < len(state.candidate_communities)
        ]
        
        print(f"[Phase 2] Selected {len(state.selected_communities)} communities for exploration")
        
    except Exception as e:
        state.error = f"Community filtering failed: {e}"
        print(f"[ERROR] {state.error}")
    
    return state

def parallel_exploration_node(state: AgentState) -> AgentState:
    """Phase 3: Community exploration (sequential in this implementation)"""
    print(f"[Phase 3] Community Exploration (sequential)")
    
    if state.error or not state.selected_communities:
        return state
    
    neo4j_client = None
    try:
        neo4j_client = Neo4jClient()
        llm_client = LLMClient()
        
        # Shared state for coordination
        shared_state = {
            'explored_entities': [],
            'shared_insights': []
        }
        
        # Explore each community (sequentially for simplicity)
        for community in state.selected_communities:
            explorer = CommunityExplorer(community, neo4j_client, llm_client)
            
            results = explorer.explore(state.query, shared_state)
            
            comm_id = results['community_id']
            state.exploration_results[comm_id] = results
            state.shared_insights.extend(results.get('insights', []))
        # -------------------------------------------------------------------
        # Phase 3b: Entity-only exploration path (direct similarity to query)
        print(f"[Phase 3b] Direct Entity Exploration for top {ENTITY_PATH_K} entities")
        try:
            # Load or build entity embeddings cache
            all_entities = neo4j_client.get_all_entities()
            entity_emb = EntityEmbeddingHelper(EMBEDDING_MODEL)
            entity_emb.load_or_build(all_entities)
            top_entities = entity_emb.topk(state.query, ENTITY_PATH_K)
            for ent_item, ent_score in top_entities:
                ent_id = ent_item['id']
                ent_name = ent_item.get('name','')
                # 1-hop neighbors: fetch all, then rank top 5 by similarity
                nbors1 = neo4j_client.get_entity_neighbors(ent_id)
                emb1 = EntityEmbeddingHelper(EMBEDDING_MODEL)
                emb1.load_or_build(nbors1)
                top_nbors1 = emb1.topk(state.query, 5)
                first_neighbors = [item for item, _ in top_nbors1]
                # 2-hop neighbors: for each first neighbor, rank its neighbors
                second_neighbors = []
                for nb in first_neighbors:
                    nbors2 = neo4j_client.get_entity_neighbors(nb['id'])
                    emb2 = EntityEmbeddingHelper(EMBEDDING_MODEL)
                    emb2.load_or_build(nbors2)
                    top_nbors2 = emb2.topk(state.query, 5)
                    second_neighbors.extend([item for item, _ in top_nbors2])
                # build context from entity and neighbors
                ctx_lines = [f"Entity: {ent_name}"]
                for nb in first_neighbors:
                    ctx_lines.append(
                        f"1-hop: {nb.get('name','')} - Desc: {nb.get('description','')} - rel: {nb.get('relationship_type','')} - props: {nb.get('edge_props',{})}"
                    )
                for nb in second_neighbors:
                    ctx_lines.append(
                        f"2-hop: {nb.get('name','')} - Desc: {nb.get('description','')} - rel: {nb.get('relationship_type','')} - props: {nb.get('edge_props',{})}"
                    )
                enriched_ctx = "\n".join(ctx_lines)
                # Analyze with LLM
                rel_score, insights = llm_client.analyze_entity_relevance(
                    state.query, ent_item, enriched_ctx
                )
                key = f"entity_{ent_id}"
                state.exploration_results[key] = {
                    'entity': ent_name,
                    'similarity': ent_score,
                    'relevance': rel_score,
                    'insights': insights,
                    'neighbors': {
                        'first_hop': first_neighbors,
                        'second_hop': second_neighbors
                    }
                }
                state.shared_insights.extend(insights)
                time.sleep(RATE_LIMIT_SLEEP)
            print(f"[Phase 3b] Completed exploration of {len(top_entities)} entities")
        except Exception as e:
            print(f"[Phase 3b] Entity exploration error: {e}")
            # continue without blocking
        
        state.explored_entities = shared_state['explored_entities']
        
        print(f"[Phase 3] Completed exploration of {len(state.exploration_results)} communities")
        
    except Exception as e:
        state.error = f"Parallel exploration failed: {e}"
        print(f"[ERROR] {state.error}")
    finally:
        if neo4j_client is not None:
            try:
                neo4j_client.close()
            except Exception:
                pass
    
    return state

def synthesis_node(state: AgentState) -> AgentState:
    """Phase 4: Synthesize final answer"""
    print(f"[Phase 4] Final Synthesis")
    
    if state.error:
        return state
    
    try:
        llm_client = LLMClient()
        
        state.final_answer = llm_client.synthesize_answer(state.query, state.exploration_results)
        
        print(f"[Phase 4] Generated final answer ({len(state.final_answer)} chars)")
        
    except Exception as e:
        state.error = f"Synthesis failed: {e}"
        print(f"[ERROR] {state.error}")
    
    return state

# ---------------------------------------------------------------------------
# LANGGRAPH WORKFLOW CONSTRUCTION
# ---------------------------------------------------------------------------

def build_workflow() -> StateGraph:
    """Build the LangGraph workflow"""
    
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("discover", community_discovery_node)
    workflow.add_node("filter", community_filtering_node)
    workflow.add_node("explore", parallel_exploration_node)
    workflow.add_node("synthesize", synthesis_node)
    
    # Add edges
    workflow.set_entry_point("discover")
    workflow.add_edge("discover", "filter")
    workflow.add_edge("filter", "explore")
    workflow.add_edge("explore", "synthesize")
    workflow.add_edge("synthesize", END)
    
    # Compile with checkpointer
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------

@dataclass
class ExplorationResult:
    query: str
    communities_explored: List[str]
    total_insights: int
    final_answer: str
    error: Optional[str] = None

def explore_communities(query: str) -> ExplorationResult:
    """Main entry point for multi-agent community exploration"""
    
    print(f"\n{'='*60}")
    print(f"🔍 Multi-Agent Community Explorer")
    print(f"Query: {query}")
    print(f"{'='*60}")
    
    try:
        # Build and run workflow
        workflow = build_workflow()
        
        # Initialize state
        initial_state = AgentState(query=query)
        
        # Run workflow
        config = {"configurable": {"thread_id": f"exploration_{int(time.time())}"}}
        final_state = workflow.invoke(initial_state, config)

        # Normalize state to dict for robust access
        if hasattr(final_state, "model_dump"):
            fs: Dict[str, Any] = final_state.model_dump()
        elif isinstance(final_state, dict):
            fs = final_state
        else:
            # Fallback best-effort conversion
            try:
                fs = json.loads(json.dumps(final_state, default=lambda o: getattr(o, "__dict__", str(o))))
            except Exception:
                fs = {}
        
        # Extract results
        exploration_results: Dict[str, Any] = fs.get("exploration_results", {}) or {}
        communities_explored = list(exploration_results.keys())
        total_insights = len(fs.get("shared_insights", []) or [])

        # Prepare output directories
        run_dir = OUTPUT_ROOT / f"run_{int(time.time())}"
        comm_dir = run_dir / "communities"
        run_dir.mkdir(parents=True, exist_ok=True)
        comm_dir.mkdir(parents=True, exist_ok=True)

        # Save selected communities list
        selected_path = run_dir / "selected_communities.json"
        selected_payload = {
            "query": query,
            "selected": fs.get("selected_communities", []) or [],
            "candidate_count": len(fs.get("candidate_communities", []) or [])
        }
        # Convert pydantic models to dict if needed
        try:
            to_write = json.loads(json.dumps(selected_payload, default=lambda o: o.__dict__))
        except Exception:
            to_write = {
                "query": query,
                "selected": [
                    {"level": c.get("level"), "cid": c.get("cid"), "title": c.get("title", ""), "size": c.get("size")}
                    for c in (fs.get("selected_communities", []) or [])
                ],
                "candidate_count": len(fs.get("candidate_communities", []) or [])
            }
        selected_path.write_text(json.dumps(to_write, indent=2))
        print(f"💾 Saved selected communities → {selected_path}")

        # Save per-community exploration results
        for comm_id, results in (exploration_results or {}).items():
            comm_path = comm_dir / f"{comm_id.replace(':','_')}.json"
            try:
                payload = json.loads(json.dumps(results))
            except Exception:
                payload = json.loads(json.dumps(results, default=lambda o: getattr(o, "__dict__", str(o))))
            comm_path.write_text(json.dumps(payload, indent=2))
            print(f"💾 Saved community results → {comm_path}")

        # Save final answer
        final_path = run_dir / "final_answer.txt"
        final_path.write_text((fs.get("final_answer") or ""))
        print(f"💾 Saved final answer → {final_path}")
        print(f"📁 Run artifacts folder → {run_dir}")
        
        return ExplorationResult(
            query=query,
            communities_explored=communities_explored,
            total_insights=total_insights,
            final_answer=fs.get("final_answer", ""),
            error=fs.get("error")
        )
        
    except Exception as e:
        return ExplorationResult(
            query=query,
            communities_explored=[],
            total_insights=0,
            final_answer="",
            error=f"Workflow failed: {e}"
        )

# ---------------------------------------------------------------------------
# CLI INTERFACE
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("🚀 Multi-Agent Community Explorer")
    print("Enter your query and press Enter. Ctrl+C to exit.\n")
    
    while True:
        try:
            query = input("Q> ").strip()
            if not query:
                continue
            
            result = explore_communities(query)
            
            print(f"\n{'='*60}")
            print("📊 RESULTS")
            print(f"{'='*60}")
            
            if result.error:
                print(f"❌ Error: {result.error}")
            else:
                print(f"🏘️  Communities Explored: {len(result.communities_explored)}")
                for comm in result.communities_explored:
                    print(f"   • {comm}")
                
                print(f"\n💡 Total Insights Generated: {result.total_insights}")
                
                print(f"\n🎯 Final Answer:")
                print(f"{result.final_answer}")
            
            print(f"\n{'='*60}\n")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
