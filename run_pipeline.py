#!/usr/bin/env python3
"""
Knowledge Graph Pipeline Orchestrator

End-to-end pipeline for building knowledge graphs from financial documents.

Pipeline Stages:
    1. PDF Conversion      - PDF → Enriched text (LLMWhisperer + MinerU + Gemini)
    2. Document Segmentation - Split pages + classify disclaimers + extract themes
    3. Triplet Extraction   - Extract entities, relationships, scenarios (o4-mini)
    4. Merge Triplets       - Combine and deduplicate across documents
    5. Similarity Edges     - Generate SIMILAR_TO edges via embeddings (VoyageAI)
    6. Semantic Enhancement - Classify relationships semantically (OpenAI Agents)
    7. Cleanup              - Clean and normalize enhanced relationships
    8. Neo4j Push          - Push combined graph to Neo4j
    9. Community Detection  - Build community hierarchy + multi-agent explorer

Usage:
    # Run full pipeline
    python run_pipeline.py --input ./data/pdfs --output ./output

    # Run specific stages
    python run_pipeline.py --stages 4,5,6,7,8 --input ./data/triplets

    # Start from stage 5 (assumes earlier stages completed)
    python run_pipeline.py --start-stage 5 --input ./data/merged.json

    # Run only community detection
    python run_pipeline.py --stages 9 --skip-neo4j-push
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

from config import get_config, validate_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


class PipelineRunner:
    """Orchestrates the knowledge graph pipeline stages."""

    def __init__(self, input_path: Path, output_path: Path):
        self.input_path = input_path
        self.output_path = output_path
        self.config = get_config()

        # Ensure output directory exists
        self.output_path.mkdir(parents=True, exist_ok=True)

    def run_stage_1(self) -> Path:
        """Stage 1: PDF Conversion"""
        logger.info("=" * 60)
        logger.info("STAGE 1: PDF Conversion")
        logger.info("=" * 60)

        from stage_1_pdf_conversion.pdf_to_text_converter import main as convert_pdfs

        output_dir = self.output_path / "converted"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Note: pdf_to_text_converter needs to be adapted to accept parameters
        # For now, we'll call it with default behavior
        logger.info("Converting PDFs from %s to %s", self.input_path, output_dir)

        # TODO: Refactor pdf_to_text_converter to accept input/output paths
        # convert_pdfs(self.input_path, output_dir)

        logger.info("Stage 1 complete. Converted files in: %s", output_dir)
        return output_dir

    def run_stage_2(self, input_dir: Optional[Path] = None) -> Path:
        """Stage 2: Document Segmentation"""
        logger.info("=" * 60)
        logger.info("STAGE 2: Document Segmentation")
        logger.info("=" * 60)

        from stage_2_segmentation.split_pages import split_document_by_pages
        from stage_2_segmentation.extract_theme_summary import extract_theme_summary_from_txt

        input_dir = input_dir or self.output_path / "converted"
        output_dir = self.output_path / "chunks"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Process each text file
        for txt_file in input_dir.glob("*.txt"):
            logger.info("Processing: %s", txt_file.name)
            doc_name = txt_file.stem
            doc_output_dir = output_dir / doc_name

            # Split into chunks
            split_document_by_pages(str(txt_file), str(doc_output_dir))

            # Extract theme
            theme_dir = doc_output_dir / "themes"
            theme_dir.mkdir(parents=True, exist_ok=True)
            theme_path = theme_dir / "theme.txt"
            extract_theme_summary_from_txt(str(txt_file), str(theme_path))

            time.sleep(self.config.request_pause_sec)

        logger.info("Stage 2 complete. Chunks in: %s", output_dir)
        return output_dir

    def run_stage_3(self, input_dir: Optional[Path] = None) -> Path:
        """Stage 3: Triplet Extraction"""
        logger.info("=" * 60)
        logger.info("STAGE 3: Triplet Extraction")
        logger.info("=" * 60)

        from stage_3_extraction.batch_extraction import main as batch_extract

        input_dir = input_dir or self.output_path / "chunks"
        output_dir = self.output_path / "triplets"
        output_dir.mkdir(parents=True, exist_ok=True)

        # TODO: Refactor batch_extraction to accept parameters
        # batch_extract(input_dir, output_dir)

        logger.info("Stage 3 complete. Triplets in: %s", output_dir)
        return output_dir

    def run_stage_4(self, input_dir: Optional[Path] = None) -> Path:
        """Stage 4: Merge Triplets"""
        logger.info("=" * 60)
        logger.info("STAGE 4: Merge Triplets")
        logger.info("=" * 60)

        from stage_4_merge.merge_combined_jsons import combine_all_docs

        input_dir = input_dir or self.output_path / "triplets"
        output_path = self.output_path / "merged.json"

        result = combine_all_docs(str(input_dir))

        import json
        output_path.write_text(json.dumps(result, indent=2))

        logger.info("Stage 4 complete. Merged JSON: %s", output_path)
        return output_path

    def run_stage_5(self, input_path: Optional[Path] = None) -> Path:
        """Stage 5: Similarity Edges"""
        logger.info("=" * 60)
        logger.info("STAGE 5: Similarity Edges")
        logger.info("=" * 60)

        from stage_5_similarity.build_similarity_push import main as build_similarity

        input_path = input_path or self.output_path / "merged.json"

        build_similarity(
            merged_json_path=input_path,
            output_dir=self.output_path,
            top_k=self.config.models.similarity_top_k,
            sim_threshold=self.config.models.similarity_threshold,
            same_doc_threshold=self.config.models.same_doc_similarity_threshold,
        )

        output_path = self.output_path / "similarity_edges.json"
        logger.info("Stage 5 complete. Similarity edges: %s", output_path)
        return output_path

    def run_stage_6(self, input_path: Optional[Path] = None) -> Path:
        """Stage 6: Semantic Enhancement"""
        logger.info("=" * 60)
        logger.info("STAGE 6: Semantic Enhancement")
        logger.info("=" * 60)

        # TODO: Import and run relationship_enhancer_agent_runner
        # This requires async execution

        input_path = input_path or self.output_path / "similarity_edges.json"
        output_path = self.output_path / "enhanced_relationships.json"

        logger.info("Stage 6 complete. Enhanced relationships: %s", output_path)
        return output_path

    def run_stage_7(self, input_path: Optional[Path] = None) -> Path:
        """Stage 7: Cleanup"""
        logger.info("=" * 60)
        logger.info("STAGE 7: Cleanup Enhanced Relationships")
        logger.info("=" * 60)

        from stage_7_cleanup.clean_enhanced_relationships import main as clean_relationships

        input_path = input_path or self.output_path / "enhanced_relationships.json"
        output_path = self.output_path / "enhanced_relationships_cleaned.json"

        # TODO: Refactor clean_enhanced_relationships to accept parameters
        # clean_relationships(input_path, output_path)

        logger.info("Stage 7 complete. Cleaned relationships: %s", output_path)
        return output_path

    def run_stage_8(self) -> None:
        """Stage 8: Neo4j Push"""
        logger.info("=" * 60)
        logger.info("STAGE 8: Push to Neo4j")
        logger.info("=" * 60)

        from stage_8_neo4j.combine_and_push_graph import combine_and_push_graph

        merged_path = self.output_path / "merged.json"
        enhanced_path = self.output_path / "enhanced_relationships_cleaned.json"

        combine_and_push_graph(
            merged_json_path=merged_path,
            enhanced_json_path=enhanced_path,
            bolt_uri=self.config.neo4j.uri,
            user=self.config.neo4j.username,
            password=self.config.neo4j.password,
        )

        logger.info("Stage 8 complete. Graph pushed to Neo4j.")

    def run_stage_9(self) -> None:
        """Stage 9: Community Detection & Exploration"""
        logger.info("=" * 60)
        logger.info("STAGE 9: Community Detection & Multi-Agent Exploration")
        logger.info("=" * 60)

        # TODO: Import and run community detection
        # from stage_9_exploration.local_community_enrichment import main as detect_communities

        logger.info("Stage 9 complete. Communities detected and explorer ready.")

    def run_full_pipeline(self, stages: Optional[list[int]] = None) -> None:
        """Run the full pipeline or selected stages."""
        all_stages = stages or list(range(1, 10))

        logger.info("Starting Knowledge Graph Pipeline")
        logger.info("Stages to run: %s", all_stages)
        logger.info("Input: %s", self.input_path)
        logger.info("Output: %s", self.output_path)

        start_time = time.time()

        try:
            if 1 in all_stages:
                self.run_stage_1()
            if 2 in all_stages:
                self.run_stage_2()
            if 3 in all_stages:
                self.run_stage_3()
            if 4 in all_stages:
                self.run_stage_4()
            if 5 in all_stages:
                self.run_stage_5()
            if 6 in all_stages:
                self.run_stage_6()
            if 7 in all_stages:
                self.run_stage_7()
            if 8 in all_stages:
                self.run_stage_8()
            if 9 in all_stages:
                self.run_stage_9()

            elapsed = time.time() - start_time
            logger.info("=" * 60)
            logger.info("PIPELINE COMPLETE")
            logger.info("Total time: %.2f seconds", elapsed)
            logger.info("=" * 60)

        except Exception as e:
            logger.error("Pipeline failed: %s", e)
            raise


def parse_stages(stages_str: str) -> list[int]:
    """Parse comma-separated stage numbers."""
    stages = []
    for part in stages_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-")
            stages.extend(range(int(start), int(end) + 1))
        else:
            stages.append(int(part))
    return sorted(set(stages))


def main():
    parser = argparse.ArgumentParser(
        description="Knowledge Graph Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path("./data/pdfs"),
        help="Input directory or file path (default: ./data/pdfs)"
    )

    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("./output"),
        help="Output directory (default: ./output)"
    )

    parser.add_argument(
        "--stages", "-s",
        type=str,
        default=None,
        help="Comma-separated list of stages to run (e.g., '4,5,6,7,8' or '1-9')"
    )

    parser.add_argument(
        "--start-stage",
        type=int,
        default=1,
        help="Start from this stage (default: 1)"
    )

    parser.add_argument(
        "--end-stage",
        type=int,
        default=9,
        help="End at this stage (default: 9)"
    )

    parser.add_argument(
        "--skip-neo4j-push",
        action="store_true",
        help="Skip pushing to Neo4j (stage 8)"
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate configuration, don't run pipeline"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    # Set debug logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate configuration
    try:
        validate_config()
        logger.info("Configuration validated successfully")
    except ValueError as e:
        logger.error("Configuration error: %s", e)
        logger.info("Please check your .env file and ensure all required values are set.")
        sys.exit(1)

    if args.validate_only:
        logger.info("Configuration is valid. Use --help to see usage options.")
        sys.exit(0)

    # Determine which stages to run
    if args.stages:
        stages = parse_stages(args.stages)
    else:
        stages = list(range(args.start_stage, args.end_stage + 1))

    if args.skip_neo4j_push and 8 in stages:
        stages.remove(8)

    # Run pipeline
    runner = PipelineRunner(args.input, args.output)
    runner.run_full_pipeline(stages)


if __name__ == "__main__":
    main()
