# Financial Knowledge Graph Pipeline

End-to-end pipeline for building knowledge graphs from financial documents with multi-agent exploration capabilities.

## Features

- **PDF to Knowledge Graph**: Extract structured entities and relationships from financial PDFs
- **Semantic Enhancement**: AI-powered relationship classification and enrichment
- **Graph Storage**: Neo4j integration with community detection
- **Multi-Agent Exploration**: LangGraph-based intelligent graph exploration

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FINANCIAL KNOWLEDGE GRAPH PIPELINE                        │
└─────────────────────────────────────────────────────────────────────────────┘

STAGE 1: PDF Conversion               STAGE 2: Segmentation
┌──────────────────────┐              ┌──────────────────────┐
│  LLMWhisperer        │              │  Page Splitting      │
│  + MinerU Layout     │ ──────────>  │  + Theme Extraction  │
│  + Gemini Vision     │              │  + Disclaimer Filter │
└──────────────────────┘              └──────────────────────┘
         │                                       │
         v                                       v
    [Enriched Text]                    [Chunks + Themes]
                                                 │
STAGE 3: Extraction                              │
┌──────────────────────┐                         │
│  o4-mini             │ <───────────────────────┘
│  High Reasoning      │
│  Theme-Anchored      │
└──────────────────────┘
         │
         v
    [Triplet JSONs]
         │
STAGE 4: Merge         STAGE 5: Similarity      STAGE 6: Enhancement
┌──────────────┐       ┌──────────────┐         ┌──────────────┐
│  Deduplicate │ ───>  │  VoyageAI    │ ──────> │  OpenAI      │
│  Add Hashes  │       │  + FAISS     │         │  Agents SDK  │
└──────────────┘       └──────────────┘         └──────────────┘
                              │                        │
                              v                        v
                       [SIMILAR_TO]            [Semantic Rels]
                                                       │
STAGE 7: Cleanup       STAGE 8: Neo4j          STAGE 9: Exploration
┌──────────────┐       ┌──────────────┐        ┌──────────────┐
│  Normalize   │ ───>  │  Push Graph  │ ─────> │  Communities │
│  Validate    │       │  Documents   │        │  + LangGraph │
└──────────────┘       └──────────────┘        └──────────────┘
                              │                        │
                              v                        v
                       [Neo4j DB]              [Multi-Agent Q&A]
```

## Project Structure

```
last_mile/
├── config.py                          # Centralized configuration
├── run_pipeline.py                    # Main orchestrator
├── requirements.txt                   # Dependencies
├── .env.example                       # Environment template
├── .gitignore                         # Git ignore rules
├── README.md                          # This file
│
├── stage_1_pdf_conversion/            # PDF → Enriched Text
│   └── pdf_to_text_converter.py       # LLMWhisperer + MinerU + Gemini
│
├── stage_2_segmentation/              # Document Processing
│   ├── split_pages.py                 # Page splitting + classification
│   └── extract_theme_summary.py       # Theme extraction
│
├── stage_3_extraction/                # Triplet Extraction
│   ├── extract_triplets.py            # Single document extraction
│   ├── batch_extraction.py            # Batch processing
│   ├── full_pipeline.py               # Stages 1-3 orchestrator
│   ├── add_publication_date.py        # Metadata enrichment
│   └── prompts/
│       └── prompt_v1.py               # Extraction guidelines
│
├── stage_4_merge/                     # Triplet Aggregation
│   └── merge_combined_jsons.py        # Merge + deduplicate
│
├── stage_5_similarity/                # Embedding Similarity
│   └── build_similarity_push.py       # VoyageAI + FAISS
│
├── stage_6_enhancement/               # Semantic Enhancement
│   ├── relationship_enhancer.py       # Prompt template
│   └── relationship_enhancer_agent_runner.py  # OpenAI Agents
│
├── stage_7_cleanup/                   # Data Cleanup
│   └── clean_enhanced_relationships.py
│
├── stage_8_neo4j/                     # Graph Database
│   └── combine_and_push_graph.py      # Neo4j integration
│
├── stage_9_exploration/               # Graph Exploration
│   ├── local_community_enrichment.py  # Hierarchical Leiden
│   └── multi_agent_explorer_v2.py     # LangGraph explorer
│
├── utils/                             # Shared Utilities
│   └── neo4j_utils.py                 # Neo4j helper functions
│
└── samples/                           # Sample Data
    ├── input/                         # Sample PDFs
    └── output/                        # Expected outputs
```

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/financial-knowledge-graph.git
cd financial-knowledge-graph

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For PDF conversion with MinerU (optional, large package)
pip install mineru
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys
nano .env  # or use your preferred editor
```

Required API keys:
- `OPENAI_API_KEY` - For triplet extraction and enhancement
- `GEMINI_API_KEY` - For PDF figure analysis
- `VOYAGE_API_KEY` - For embedding generation
- `LLMWHISPERER_API_KEY` - For PDF text extraction
- `NEO4J_*` - Database credentials

### 3. Run the Pipeline

```bash
# Run full pipeline
python run_pipeline.py --input ./data/pdfs --output ./output

# Run specific stages (e.g., stages 4-8 if you have triplets)
python run_pipeline.py --stages 4,5,6,7,8 --input ./data/triplets

# Validate configuration only
python run_pipeline.py --validate-only
```

## Stage Details

### Stage 1: PDF Conversion

Converts PDF documents to enriched text using:
- **LLMWhisperer**: High-quality text extraction with form mode
- **MinerU**: Document layout analysis and figure detection
- **Gemini Vision**: Chart/figure analysis with 3-page context

### Stage 2: Document Segmentation

- Splits documents on `page_end` delimiters
- Classifies last 3 pages as disclaimer vs. relevant content
- Extracts 3-sentence document theme for context

### Stage 3: Triplet Extraction

Uses OpenAI `o4-mini` with high reasoning effort to extract:
- **Entities**: Companies, Events, Factors, Metrics, Instruments, Tables
- **Relationships**: EXPOSED_TO, CAUSES, TRIGGERS, IMPACTS, OWNS, HOLDS
- **Scenarios**: Bull/bear cases with probabilities

### Stage 4: Merge Triplets

- Combines chunk-level JSONs per document
- Deduplicates by (name, type) for entities
- Deduplicates by (source, target, type) for relationships
- Adds `doc_hash` and `chunk_hash` for provenance

### Stage 5: Similarity Edges

- Generates embeddings using VoyageAI (`voyage-3-large`)
- FAISS cosine similarity search
- Creates `SIMILAR_TO` edges with configurable thresholds

### Stage 6: Semantic Enhancement

Uses OpenAI Agents SDK to upgrade generic `SIMILAR_TO` edges to:
- `ALIAS_OF`, `SERIES_CONTINUATION`, `CORRELATED_WITH`
- `INVERSE_OF`, `CHILD_OF`, `THEMATIC_OVERLAP`
- `INFORMS`, `CAUSES`, `IMPACTS`, `TRIGGERS`

### Stage 7: Cleanup

- Resolves `rel_type` vs `custom_rel_type` conflicts
- Filters null/undefined values
- Standardizes relationship format

### Stage 8: Neo4j Push

- Merges original triplets with enhanced relationships
- Converts to LangChain GraphDocument format
- Pushes to Neo4j database

### Stage 9: Community Detection & Exploration

- **Hierarchical Leiden**: Multi-level community detection using graspologic
- **LLM Summaries**: AI-generated community descriptions
- **Multi-Agent Explorer**: LangGraph-based parallel agent exploration

## Technology Stack

| Component | Technology |
|-----------|------------|
| PDF Extraction | LLMWhisperer |
| Layout Analysis | MinerU |
| Figure Analysis | Gemini Vision (gemini-2.5-flash) |
| Page Classification | GPT-4.1-mini |
| Theme Extraction | GPT-4o-mini |
| Triplet Extraction | o4-mini (high reasoning) |
| Embeddings | VoyageAI (voyage-3-large) |
| Vector Search | FAISS |
| Graph Database | Neo4j |
| Relationship Enhancement | OpenAI Agents SDK |
| Community Detection | graspologic (hierarchical Leiden) |
| Multi-Agent Framework | LangGraph |

## Configuration Options

See `.env.example` for all available configuration options including:
- API keys and credentials
- Model selections
- Similarity thresholds
- Batch sizes
- File paths

## Usage Examples

### Running Individual Stages

```python
from stage_4_merge.merge_combined_jsons import combine_all_docs

# Merge triplets from a directory
result = combine_all_docs("/path/to/triplets")
```

### Using the Neo4j Utilities

```python
from utils.neo4j_utils import push_to_neo4j

data = {
    "entities": [...],
    "relationships": [...]
}
push_to_neo4j(data, bolt="bolt://localhost:7687", user="neo4j", pw="password")
```

### Multi-Agent Exploration

```python
from stage_9_exploration.multi_agent_explorer_v2 import GraphExplorer

explorer = GraphExplorer()
result = explorer.explore("What factors impact Apple's stock price?")
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- OpenAI for GPT models and Agents SDK
- Google for Gemini Vision
- Anthropic for Claude (used in development)
- VoyageAI for embedding models
- Neo4j for graph database
