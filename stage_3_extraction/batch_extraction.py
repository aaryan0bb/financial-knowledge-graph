import json
import os
import glob
from typing import List, Dict, Any
from openai import OpenAI
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def load_prompt_template() -> str:
    """Load the prompt template from prompt_v1.py"""
    import sys
    from stage_3_extraction.prompts.prompt_v1 import prompt_1
    return prompt_1

def load_chunk_content(chunk_path: str) -> str:
    """Load the chunk content from specified chunk file"""
    with open(chunk_path, 'r', encoding='utf-8') as f:
        return f.read()

def load_document_theme() -> str:
    """Load the document theme from theme.txt"""
    theme_path = os.path.join(os.getenv("CHUNKS_DIR", "./data/chunks"), "themes/theme.txt")
    with open(theme_path, 'r', encoding='utf-8') as f:
        return f.read()

def load_page_classification_mapping(chunks_dir: str) -> Dict[str, str]:
    """Load page classification mapping to filter out disclaimer chunks"""
    mapping_path = os.path.join(chunks_dir, "page_classification_mapping.json")
    try:
        with open(mapping_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: Page classification mapping not found at {mapping_path}")
        return {}

def get_all_chunk_files(chunks_dir: str) -> List[str]:
    """Get all chunk files from the specified directory, excluding disclaimer chunks"""
    chunk_pattern = os.path.join(chunks_dir, "chunk_*.txt")
    chunk_files = glob.glob(chunk_pattern)
    
    # Load page classification mapping
    classification_mapping = load_page_classification_mapping(chunks_dir)
    
    # Filter out disclaimer chunks
    filtered_chunk_files = []
    excluded_chunks = []
    
    for chunk_file in chunk_files:
        filename = os.path.basename(chunk_file).replace('.txt', '')
        classification = classification_mapping.get(filename, 'unknown')
        
        if classification == 'disclaimer':
            excluded_chunks.append(filename)
            print(f"Excluding {filename} (classified as disclaimer)")
        else:
            filtered_chunk_files.append(chunk_file)
    
    # Sort by chunk number for consistent processing order
    def extract_chunk_number(filepath):
        filename = os.path.basename(filepath)
        # Extract number from chunk_X.txt
        chunk_num = filename.replace('chunk_', '').replace('.txt', '')
        return int(chunk_num)
    
    filtered_chunk_files.sort(key=extract_chunk_number)
    
    if excluded_chunks:
        print(f"Excluded {len(excluded_chunks)} disclaimer chunks: {', '.join(excluded_chunks)}")
    
    return filtered_chunk_files

def extract_triplets_with_gpt4_mini(chunk_content: str, prompt_template: str, chunk_theme: str) -> Dict[str, Any]:
    """Extract triplets using GPT-4 mini with high reasoning effort"""
    
    # Initialize OpenAI client
    client = OpenAI()
    
    # Create the extraction prompt
    extraction_prompt = f"""
    Can you please extract node entity triplets from the chunk below? Please follow guidelines as mentioned in the <triplet_extraction_prompt> for extracting the triplets from <chunk>, use <chunk_theme> to extract the relevant names for all entities

    <triplet_extraction_prompt>
    {prompt_template}
    </triplet_extraction_prompt>

    <chunk_theme>
    {chunk_theme}
    </chunk_theme>

    <chunk>
    {chunk_content}
    </chunk>


"""
    
    try:
        print("Making API call to OpenAI...")
        # Make API call to GPT-4 mini with high reasoning effort
        # First try with GPT-4.1
        # response = client.chat.completions.create(
        #     model="gpt-4.1",
        #     messages=[
        #         {
        #             "role": "user",
        #             "content": extraction_prompt
        #         }
        #     ],
        #     max_tokens=32000
        # )

        # Fallback to GPT-4 mini if 4.1 fails
        response = client.chat.completions.create(
            model="o4-mini", 
            reasoning_effort="high",
            messages=[
                {
                    "role": "user",
                    "content": extraction_prompt
                }
            ],
            max_completion_tokens=32000
)
        
        print("API call completed successfully.")

        print(response)
        
        # Check if response has content
        if not response.choices or not response.choices[0].message.content:
            print("WARNING: Empty response from API")
            return {}
        
        # Extract JSON content from response
        json_content = response.choices[0].message.content.strip()
        
        # Debug: print the raw response
        print(f"Raw API response: {json_content[:500]}...")
        
        # Clean up the JSON content (remove markdown code blocks if present)
        if json_content.startswith("```json"):
            json_content = json_content[7:]
        if json_content.endswith("```"):
            json_content = json_content[:-3]
        json_content = json_content.strip()
        
        # Parse JSON directly without Pydantic validation
        parsed_data = json.loads(json_content)
        
        return parsed_data
        
    except Exception as e:
        print(f"Error during triplet extraction: {e}")
        print(f"Response content: {response.choices[0].message.content if 'response' in locals() else 'No response'}")
        # Return empty extraction on error
        return {}

def save_triplets_to_json(triplets: Dict[str, Any], output_dir: str, chunk_name: str) -> str:
    """Save extracted triplets to JSON file"""
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename with timestamp and chunk name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"o4_triplets_{chunk_name}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Save to JSON file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(triplets, f, indent=2, ensure_ascii=False)
    
    return filepath

def process_single_chunk(chunk_path: str, prompt_template: str, chunk_theme: str, output_dir: str) -> Dict[str, Any]:
    """Process a single chunk file and extract triplets"""
    
    chunk_name = os.path.basename(chunk_path).replace('.txt', '')
    print(f"\n{'='*60}")
    print(f"Processing {chunk_name}...")
    print(f"{'='*60}")
    
    # Load chunk content
    chunk_content = load_chunk_content(chunk_path)
    print(f"Chunk content length: {len(chunk_content)} characters")
    
    # Extract triplets
    print(f"Extracting triplets for {chunk_name}...")
    triplets = extract_triplets_with_gpt4_mini(chunk_content, prompt_template, chunk_theme)
    
    # Display extraction summary
    print(f"\nExtraction completed for {chunk_name}:")
    print(f"- Entities: {len(triplets.get('entities', []))}")
    print(f"- Relationships: {len(triplets.get('relationships', []))}")
    print(f"- Scenarios: {len(triplets.get('scenarios', []))}")
    
    # Save triplets to JSON
    filepath = save_triplets_to_json(triplets, output_dir, chunk_name)
    print(f"Triplets saved to: {filepath}")
    
    return {
        'chunk_name': chunk_name,
        'chunk_path': chunk_path,
        'output_path': filepath,
        'triplets': triplets,
        'entity_count': len(triplets.get('entities', [])),
        'relationship_count': len(triplets.get('relationships', [])),
        'scenario_count': len(triplets.get('scenarios', []))
    }

def generate_batch_summary(results: List[Dict[str, Any]], output_dir: str) -> str:
    """Generate a summary report of the batch processing"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(output_dir, f"batch_summary_{timestamp}.json")
    
    # Calculate totals
    total_entities = sum(r['entity_count'] for r in results)
    total_relationships = sum(r['relationship_count'] for r in results)
    total_scenarios = sum(r['scenario_count'] for r in results)
    
    summary = {
        'batch_timestamp': timestamp,
        'total_chunks_processed': len(results),
        'total_entities_extracted': total_entities,
        'total_relationships_extracted': total_relationships,
        'total_scenarios_extracted': total_scenarios,
        'chunk_results': [
            {
                'chunk_name': r['chunk_name'],
                'output_file': os.path.basename(r['output_path']),
                'entity_count': r['entity_count'],
                'relationship_count': r['relationship_count'],
                'scenario_count': r['scenario_count']
            }
            for r in results
        ]
    }
    
    # Save summary
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    return summary_path

def main():
    """Main workflow for batch triplet extraction"""
    
    # Configuration
    chunks_dir = os.getenv("CHUNKS_DIR", "./data/chunks")
    output_dir = os.getenv("TRIPLETS_DIR", "./data/triplets")
    
    print("Starting batch triplet extraction workflow...")
    print(f"Input directory: {chunks_dir}")
    print(f"Output directory: {output_dir}")
    
    # Get all chunk files
    chunk_files = get_all_chunk_files(chunks_dir)
    print(f"\nFound {len(chunk_files)} chunk files to process:")
    for chunk_file in chunk_files:
        print(f"  - {os.path.basename(chunk_file)}")
    
    # Load shared resources
    print("\nLoading shared resources...")
    prompt_template = load_prompt_template()
    chunk_theme = load_document_theme()
    print(f"Document theme: {chunk_theme[:100]}...")
    
    # Process each chunk
    results = []
    failed_chunks = []
    
    for i, chunk_path in enumerate(chunk_files, 1):
        try:
            print(f"\n\nProgress: {i}/{len(chunk_files)}")
            result = process_single_chunk(chunk_path, prompt_template, chunk_theme, output_dir)
            results.append(result)
        except Exception as e:
            chunk_name = os.path.basename(chunk_path)
            print(f"ERROR: Failed to process {chunk_name}: {e}")
            failed_chunks.append(chunk_name)
    
    # Generate summary report
    print(f"\n{'='*60}")
    print("BATCH PROCESSING COMPLETE")
    print(f"{'='*60}")
    
    print(f"Successfully processed: {len(results)}/{len(chunk_files)} chunks")
    if failed_chunks:
        print(f"Failed chunks: {', '.join(failed_chunks)}")
    
    # Calculate and display totals
    total_entities = sum(r['entity_count'] for r in results)
    total_relationships = sum(r['relationship_count'] for r in results)
    total_scenarios = sum(r['scenario_count'] for r in results)
    
    print(f"\nTotal extraction results:")
    print(f"- Total entities: {total_entities}")
    print(f"- Total relationships: {total_relationships}")
    print(f"- Total scenarios: {total_scenarios}")
    
    # Generate and save summary
    if results:
        summary_path = generate_batch_summary(results, output_dir)
        print(f"\nBatch summary saved to: {summary_path}")
    
    return results, failed_chunks

if __name__ == "__main__":
    results, failed_chunks = main()