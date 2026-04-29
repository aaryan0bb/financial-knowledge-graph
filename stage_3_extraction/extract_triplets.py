import json
import os
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from openai import OpenAI
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Simple Pydantic model for raw JSON output
class TripletExtraction(BaseModel):
    """Raw JSON output without structure validation"""
    pass

def load_prompt_template() -> str:
    """Load the prompt template from prompt_v1.py"""
    import sys
    sys.path.append(os.getenv("PROMPTS_DIR", "./prompts"))
    from prompt_v1 import prompt_1
    return prompt_1

def load_chunk_content() -> str:
    """Load the chunk content from chunk_2.txt"""
    chunk_path = os.getenv("CHUNK_PATH", "./data/chunks/sample_chunk.txt")
    with open(chunk_path, 'r', encoding='utf-8') as f:
        return f.read()

def load_document_theme() -> str:
    """Load the document theme from theme.txt"""
    theme_path = os.getenv("THEME_PATH", "./data/themes/sample_theme.txt")
    with open(theme_path, 'r', encoding='utf-8') as f:
        return f.read()

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

def save_triplets_to_json(triplets: Dict[str, Any], output_dir: str) -> str:
    """Save extracted triplets to JSON file"""
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"o4_triplets_chunk2_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Save to JSON file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(triplets, f, indent=2, ensure_ascii=False)
    
    return filepath

def main():
    """Main workflow for triplet extraction"""
    
    print("Starting triplet extraction workflow...")
    
    # Load prompt template, chunk content, and document theme
    print("Loading prompt template, chunk content, and document theme...")
    prompt_template = load_prompt_template()
    chunk_content = load_chunk_content()
    chunk_theme = load_document_theme()
    
    print(f"Chunk content length: {len(chunk_content)} characters")
    print(f"Document theme: {chunk_theme[:100]}...")
    
    # Extract triplets using GPT-4 mini
    print("Extracting triplets with GPT-4 mini (high reasoning effort)...")
    triplets = extract_triplets_with_gpt4_mini(chunk_content, prompt_template, chunk_theme)
    
    # Display extraction summary
    print(f"\nExtraction completed:")
    print(f"- Entities: {len(triplets.get('entities', []))}")
    print(f"- Relationships: {len(triplets.get('relationships', []))}")
    print(f"- Scenarios: {len(triplets.get('scenarios', []))}")
    
    # Save triplets to JSON
    output_dir = os.getenv("TRIPLETS_DIR", "./data/triplets")
    filepath = save_triplets_to_json(triplets, output_dir)
    
    print(f"\nTriplets saved to: {filepath}")
    
    return triplets, filepath

if __name__ == "__main__":
    triplets, filepath = main()