"""
Centralized configuration for the Knowledge Graph Pipeline.

All configuration is loaded from environment variables with sensible defaults.
Copy .env.example to .env and fill in your values.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class APIConfig:
    """API keys configuration."""
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    voyage_api_key: str = field(default_factory=lambda: os.getenv("VOYAGE_API_KEY", ""))
    llmwhisperer_api_key: str = field(default_factory=lambda: os.getenv("LLMWHISPERER_API_KEY", ""))


@dataclass
class Neo4jConfig:
    """Neo4j database configuration."""
    uri: str = field(default_factory=lambda: os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    username: str = field(default_factory=lambda: os.getenv("NEO4J_USERNAME", "neo4j"))
    password: str = field(default_factory=lambda: os.getenv("NEO4J_PASSWORD", ""))


@dataclass
class PathConfig:
    """File path configuration."""
    # Base directories
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent)
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", "./data")))
    output_dir: Path = field(default_factory=lambda: Path(os.getenv("OUTPUT_DIR", "./output")))

    # Stage-specific paths
    pdf_input_dir: Path = field(default_factory=lambda: Path(os.getenv("PDF_INPUT_DIR", "./data/pdfs")))
    converted_text_dir: Path = field(default_factory=lambda: Path(os.getenv("CONVERTED_TEXT_DIR", "./data/converted")))
    chunks_dir: Path = field(default_factory=lambda: Path(os.getenv("CHUNKS_DIR", "./data/chunks")))
    triplets_dir: Path = field(default_factory=lambda: Path(os.getenv("TRIPLETS_DIR", "./data/triplets")))
    merged_json_path: Path = field(default_factory=lambda: Path(os.getenv("MERGED_JSON_PATH", "./data/merged.json")))
    embeddings_path: Path = field(default_factory=lambda: Path(os.getenv("EMBEDDINGS_PATH", "./data/embeddings.npy")))
    similarity_edges_path: Path = field(default_factory=lambda: Path(os.getenv("SIMILARITY_EDGES_PATH", "./data/similarity_edges.json")))
    enhanced_relationships_path: Path = field(default_factory=lambda: Path(os.getenv("ENHANCED_RELATIONSHIPS_PATH", "./data/enhanced_relationships.json")))
    cleaned_relationships_path: Path = field(default_factory=lambda: Path(os.getenv("CLEANED_RELATIONSHIPS_PATH", "./data/enhanced_relationships_cleaned.json")))


@dataclass
class ModelConfig:
    """Model configuration."""
    # PDF conversion
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    pdf_dpi: int = field(default_factory=lambda: int(os.getenv("PDF_DPI", "300")))
    pdf_timeout: int = field(default_factory=lambda: int(os.getenv("PDF_TIMEOUT", "300")))

    # Triplet extraction
    extraction_model: str = field(default_factory=lambda: os.getenv("EXTRACTION_MODEL", "o4-mini"))
    extraction_reasoning_effort: str = field(default_factory=lambda: os.getenv("EXTRACTION_REASONING_EFFORT", "high"))
    extraction_max_tokens: int = field(default_factory=lambda: int(os.getenv("EXTRACTION_MAX_TOKENS", "32000")))

    # Classification
    classification_model: str = field(default_factory=lambda: os.getenv("CLASSIFICATION_MODEL", "gpt-4.1-mini"))
    theme_model: str = field(default_factory=lambda: os.getenv("THEME_MODEL", "gpt-4o-mini"))

    # Embeddings
    voyage_model: str = field(default_factory=lambda: os.getenv("VOYAGE_MODEL", "voyage-3-large"))
    voyage_batch_size: int = field(default_factory=lambda: int(os.getenv("VOYAGE_BATCH_SIZE", "1000")))

    # Similarity
    similarity_top_k: int = field(default_factory=lambda: int(os.getenv("SIMILARITY_TOP_K", "50")))
    similarity_threshold: float = field(default_factory=lambda: float(os.getenv("SIMILARITY_THRESHOLD", "0.73")))
    same_doc_similarity_threshold: float = field(default_factory=lambda: float(os.getenv("SAME_DOC_SIMILARITY_THRESHOLD", "0.85")))

    # Relationship enhancement
    enhancement_model: str = field(default_factory=lambda: os.getenv("ENHANCEMENT_MODEL", "o4-mini"))
    enhancement_batch_size: int = field(default_factory=lambda: int(os.getenv("ENHANCEMENT_BATCH_SIZE", "20")))

    # Community detection
    community_model: str = field(default_factory=lambda: os.getenv("COMMUNITY_MODEL", "gpt-4o-mini"))

    # Multi-agent exploration
    explorer_model: str = field(default_factory=lambda: os.getenv("EXPLORER_MODEL", "gpt-4o-mini"))


@dataclass
class PipelineConfig:
    """Main pipeline configuration combining all sub-configs."""
    api: APIConfig = field(default_factory=APIConfig)
    neo4j: Neo4jConfig = field(default_factory=Neo4jConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    models: ModelConfig = field(default_factory=ModelConfig)

    # Pipeline control
    request_pause_sec: float = field(default_factory=lambda: float(os.getenv("REQUEST_PAUSE_SEC", "2.0")))
    debug: bool = field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")

    def validate(self) -> list[str]:
        """Validate configuration and return list of missing required values."""
        errors = []
        if not self.api.openai_api_key:
            errors.append("OPENAI_API_KEY is required")
        if not self.neo4j.password:
            errors.append("NEO4J_PASSWORD is required for database operations")
        return errors


# Global config instance
config = PipelineConfig()


def get_config() -> PipelineConfig:
    """Get the global configuration instance."""
    return config


def validate_config() -> None:
    """Validate configuration and raise error if invalid."""
    errors = config.validate()
    if errors:
        raise ValueError(f"Configuration errors: {', '.join(errors)}")
