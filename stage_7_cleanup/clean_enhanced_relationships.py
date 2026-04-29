import json
import os
from pathlib import Path

# Define input and output file paths
INPUT_FILE = Path(os.getenv("ENHANCED_RELATIONSHIPS_PATH", "./data/enhanced_relationships.json"))
OUTPUT_FILE = Path(os.getenv("CLEANED_RELATIONSHIPS_PATH", "./data/enhanced_relationships_cleaned.json"))

def clean_relationships(input_path: Path, output_path: Path):
    if not input_path.exists():
        print(f"Error: Input file not found at {input_path}")
        return

    try:
        with open(input_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {input_path}: {e}")
        return

    if "enhanced_relationships" not in data or not isinstance(data["enhanced_relationships"], list):
        print("Error: JSON structure invalid. Expected a top-level 'enhanced_relationships' list.")
        return

    cleaned_relationships = []
    for rel in data["enhanced_relationships"]:
        # Determine the final relationship type: use rel_type unless it is
        # "OTHER" (case-insensitive) and a more specific custom_rel_type is
        # provided.
        base_type = rel.get("rel_type")
        final_type = (
            rel.get("custom_rel_type")
            if (isinstance(base_type, str) and base_type.upper() == "OTHER" and rel.get("custom_rel_type"))
            else base_type
        )

        cleaned_rel = {
            "source": rel.get("source"),
            "target": rel.get("target"),
            "type": final_type,
            "properties": rel.get("properties"),
            "description": rel.get("description"),
        }
        # Filter out None values that might result from missing keys if not using .get safely
        cleaned_relationships.append({k: v for k, v in cleaned_rel.items() if v is not None})

    data["enhanced_relationships"] = cleaned_relationships

    try:
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Successfully cleaned and saved relationships to {output_path}")
    except IOError as e:
        print(f"Error writing to file {output_path}: {e}")

if __name__ == "__main__":
    clean_relationships(INPUT_FILE, OUTPUT_FILE) 