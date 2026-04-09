import os
import glob
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import time

from dotenv import load_dotenv
from openai import OpenAI

# Local imports --------------------------------------------------------------
# We reuse helper logic from the existing scripts by importing them.
import sys
SCRIPT_DIR = Path(__file__).parent
sys.path.append(str(SCRIPT_DIR))

from split_pages import (
    split_document_by_pages,
    generate_folder_name,
    classify_page_content_with_llm,  # re-exposed for completeness
)
from extract_theme_summary import extract_theme_summary_from_txt
from test_batch_extraction import (
    load_prompt_template,
    load_chunk_content,
    load_page_classification_mapping,
    extract_triplets_with_gpt4_mini,
    save_triplets_to_json,
)

load_dotenv()

# ---------------------------------------------------------------------------
# High-level pipeline functions
# ---------------------------------------------------------------------------

# seconds to wait between OpenAI calls to avoid rate limits
REQUEST_PAUSE_SEC = 2  # adjust as needed

def process_report(txt_path: Path, chunks_root: Path, triplets_root: Path):
    """Process a single report txt → chunks (+theme) → triplets."""

    folder_name = generate_folder_name(txt_path.name)
    chunks_dir = chunks_root / folder_name
    triplet_dir = triplets_root / folder_name

    print(f"\n=== Processing report: {txt_path.name} ===")
    print(f"Chunks dir   : {chunks_dir}")
    print(f"Triplets dir : {triplet_dir}\n")

    # 1) Split into pages & classify
    classification_mapping = split_document_by_pages(str(txt_path), str(chunks_dir))

    # 2) Extract theme summary
    theme_dir = chunks_dir / "themes"
    theme_dir.mkdir(parents=True, exist_ok=True)
    theme_path = theme_dir / "theme.txt"
    extract_theme_summary_from_txt(str(txt_path), str(theme_path))

    # small pause after theme extraction
    time.sleep(REQUEST_PAUSE_SEC)

    # 3) Extract triplets per chunk (exclude disclaimers)
    prompt_template = load_prompt_template()
    chunk_theme = theme_path.read_text(encoding="utf-8")

    # Gather chunks after disclaimer filtering
    all_chunk_files = sorted(chunks_dir.glob("chunk_*.txt"), key=lambda p: int(p.stem.split('_')[1]))
    mapping = load_page_classification_mapping(str(chunks_dir))
    chunk_files = [p for p in all_chunk_files if mapping.get(p.stem, "relevant_content") != "disclaimer"]

    print(f"\nTotal chunks to process: {len(chunk_files)} (excluding disclaimers)\n")

    results = []
    for chunk_path in chunk_files:
        chunk_name = chunk_path.stem
        chunk_content = load_chunk_content(str(chunk_path))
        triplets = extract_triplets_with_gpt4_mini(chunk_content, prompt_template, chunk_theme)
        out_path = save_triplets_to_json(triplets, str(triplet_dir), chunk_name)
        results.append({
            "chunk": chunk_name,
            "entities": len(triplets.get("entities", [])),
            "relationships": len(triplets.get("relationships", [])),
            "scenarios": len(triplets.get("scenarios", [])),
            "file": os.path.basename(out_path),
        })
        print(f"Saved triplets for {chunk_name} → {out_path}")

        # Throttle between chunk calls
        time.sleep(REQUEST_PAUSE_SEC)

    # 4) Batch summary
    summary = {
        "report": txt_path.name,
        "timestamp": datetime.now().isoformat(),
        "total_chunks": len(chunk_files),
        "total_entities": sum(r["entities"] for r in results),
        "total_relationships": sum(r["relationships"] for r in results),
        "total_scenarios": sum(r["scenarios"] for r in results),
        "chunk_breakdown": results,
    }
    summary_path = triplet_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    triplet_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSummary written to {summary_path}\n")


# ---------------------------------------------------------------------------
# CLI-style runner (edit paths below or call programmatically)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ----- user configuration -----
    INPUT_TXT_DIR = Path(os.getenv("CONVERTED_TEXT_DIR", "./data/converted"))
    CHUNKS_ROOT = Path(os.getenv("CHUNKS_DIR", "./data/chunks"))
    TRIPLETS_ROOT = Path(os.getenv("TRIPLETS_DIR", "./data/triplets"))
    # --------------------------------

    INPUT_TXT_DIR = INPUT_TXT_DIR.expanduser().resolve()
    CHUNKS_ROOT = CHUNKS_ROOT.expanduser().resolve()
    TRIPLETS_ROOT = TRIPLETS_ROOT.expanduser().resolve()

    txt_files = list(INPUT_TXT_DIR.glob("*.txt"))
    if not txt_files:
        print("No .txt files found in", INPUT_TXT_DIR)
        exit()

    for txt in txt_files:
        try:
            process_report(txt, CHUNKS_ROOT, TRIPLETS_ROOT)
        except Exception as exc:
            print(f"ERROR processing {txt.name}: {exc}") 