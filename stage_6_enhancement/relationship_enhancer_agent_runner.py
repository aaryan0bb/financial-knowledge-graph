#!/usr/bin/env python
"""relationship_enhancer_agent_runner.py

Re-implementation of the massive *SIMILAR_TO* upgrader using the **OpenAI
Agents SDK** instead of manual OpenAI calls.

• Validates both input & output with strict Pydantic models.
• Uses `Agent` with `output_type=` so the JSON schema is sent to the model and
  enforced by the SDK (no manual parsing / retries needed).
• Batches 10 edges at a time; sleeps 2 seconds to respect RPM.
"""
from __future__ import annotations

import asyncio, json, os, time
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field, ValidationError, field_validator
from agents import Agent, Runner  # openai-agents SDK
from agents.agent_output import AgentOutputSchema
from agents.exceptions import InputGuardrailTripwireTriggered
from agents.model_settings import ModelSettings # Import ModelSettings

from relationship_enhancer import rel_enhancer_prompt  # long prompt string

# ---------------------------------------------------------------------------
# Pydantic I/O schemas – strict
# ---------------------------------------------------------------------------

class Properties(BaseModel):
    probability: float | None = None
    lag_days: int | None = None
    impact_value: float | None = None
    unit: str | None = None
    non_linear: bool | None = None
    confidence: int

    model_config = dict(extra="forbid")

class Edge(BaseModel):
    source: dict
    target: dict
    type: str
    properties: dict

    model_config = dict(extra="forbid")

class InputBatch(BaseModel):
    edges: list[Edge]

    @field_validator("edges")
    @classmethod
    def _check_len(cls, v):
        if not v:
            raise ValueError("edges empty")
        return v

class RelationshipOut(BaseModel):
    source: str
    target: str
    rel_type: str
    custom_rel_type: str | None = None
    rel_explanation: str
    properties: Properties
    description: str

    model_config = dict(extra="forbid")

class EnhancedBatch(BaseModel):
    enhanced_relationships: List[RelationshipOut]
    model_config = dict(extra="forbid")

# Let the Agents SDK wrap this into an output-schema object
StrictEnhancedBatch = AgentOutputSchema(EnhancedBatch)

# ---------------------------------------------------------------------------
# Build the Agent
# ---------------------------------------------------------------------------

enhancer_agent = Agent(
    name="Relationship Enhancer",
    instructions=rel_enhancer_prompt,
    output_type=StrictEnhancedBatch,  # schema automatically enforced
    model="o4-mini",
    model_settings=ModelSettings(
        reasoning_effort="high",
        max_completion_tokens=15000,
        max_window_tokens=80000,
        sleep_between_calls=2,
    ),
)

# ---------------------------------------------------------------------------
# Runner logic – batches of 10
# ---------------------------------------------------------------------------

INPUT_JSON = Path(os.getenv("MERGED_JSON_PATH", "./data/merged.json"))
OUTPUT_JSON = Path(os.getenv("ENHANCED_RELATIONSHIPS_PATH", "./data/enhanced_relationships.json"))
BATCH_SIZE = 20

if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable is required. Set it in your .env file.")

all_edges = json.loads(INPUT_JSON.read_text())
edges = all_edges["edges"] if isinstance(all_edges, dict) else all_edges

async def run_batches():
    enhanced_total: List[dict] = []
    for i in range(0, len(edges), BATCH_SIZE):
        print(f"Processing batch {i} of {len(edges)}")
        # Slice the input edges for this batch
        batch_edges = edges[i : i + BATCH_SIZE]
        # Validate input via Pydantic
        try:
            input_model = InputBatch(edges=batch_edges)
        except ValidationError as e:
            print("⚠️ Skipping bad input batch due to Pydantic validation error:", e)
            continue

        # Convert validated input to JSON string for the agent
        json_input_for_agent = input_model.model_dump_json()

        result = await Runner.run(enhancer_agent, json_input_for_agent)
        try:
            batch_out: EnhancedBatch = result.final_output_as(StrictEnhancedBatch)
            enhanced_total.extend([rel.model_dump() for rel in batch_out.enhanced_relationships])
            # Print total enhanced edges so far to track progress
            print(f"✔ After batch starting at index {i}: cumulative enhanced edges = {len(enhanced_total)}")
        except Exception as e:
            print("⚠️ failed to validate agent output –", e)
            continue
        # time.sleep(enhancer_agent.model_config["sleep_between_calls"])

    OUTPUT_JSON.write_text(json.dumps({"enhanced_relationships": enhanced_total}, indent=2))
    print("✔ Saved", len(enhanced_total), "edges →", OUTPUT_JSON)

if __name__ == "__main__":
    asyncio.run(run_batches()) 