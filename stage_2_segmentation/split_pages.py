#!/usr/bin/env python3
import json
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def classify_page_content_with_llm(page_text):
    """Classify page as disclaimer or relevant financial content using LLM."""
    
    # Initialize OpenAI client
    client = OpenAI()
    
    # Create classification prompt
    classification_prompt = f"""You are analyzing a page from a financial research report. Your task is to classify whether this page contains primarily legal/regulatory content or substantive financial analysis.

Analyze the following page content and classify it into one of two categories:
1. "disclaimer" - if the page primarily contains disclosures, legal notices, regulatory information, compliance statements, risk warnings, or other non-analytical content
2. "relevant_content" - if the page contains financial analysis, company information, market data, investment recommendations, or other substantive research content

Page content to analyze:
{page_text[:3000]}  # Limit to first 3000 characters to manage token usage

Please respond with ONLY one of these two classifications: "disclaimer" or "relevant_content"
"""
    
    try:
        # Make API call to GPT-4o-mini
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "user",
                    "content": classification_prompt
                }
            ],
            temperature=0.1,  # Low temperature for consistent classification
            max_tokens=10
        )
        
        # Extract classification from response
        classification = response.choices[0].message.content.strip().lower()
        
        # Validate response
        if classification in ['disclaimer', 'relevant_content']:
            return classification
        else:
            print(f"Unexpected LLM response: {classification}, defaulting to relevant_content")
            return 'relevant_content'
            
    except Exception as e:
        print(f"Error during LLM classification: {e}")
        # Fallback to simple heuristic if LLM fails
        return classify_page_content_fallback(page_text)

def classify_page_content_fallback(page_text):
    """Simple fallback classification based on content length and structure."""
    # If page is very short or mostly tables/charts, likely disclaimer
    if len(page_text.strip()) < 500:
        return 'disclaimer'
    
    # Default to relevant content
    return 'relevant_content'

def split_document_by_pages(input_file, output_dir):
    """Split document into pages based on 'page_end' markers and classify content."""
    
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Split by page_end, keeping the content before each page_end
    pages = content.split('page_end')
    
    # Remove the last empty part if it exists
    if pages[-1].strip() == '':
        pages = pages[:-1]
    
    # Classification mapping
    classification_mapping = {}
    
    # Process last 3 pages for classification if document has 3+ pages
    if len(pages) >= 3:
        print(f"\nAnalyzing last 3 pages using LLM for disclaimer content...")
        for i in range(len(pages) - 3, len(pages)):
            page_num = i + 1
            print(f"Classifying chunk_{page_num}...")
            classification = classify_page_content_with_llm(pages[i])
            classification_mapping[f'chunk_{page_num}'] = classification
            print(f"chunk_{page_num}: {classification}")
    
    # Save each page as a separate chunk
    for i, page in enumerate(pages, 1):
        chunk_filename = f"{output_dir}/chunk_{i}.txt"
        with open(chunk_filename, 'w', encoding='utf-8') as f:
            f.write(page.strip())
        print(f"Saved {chunk_filename}")
        
        # For pages not in last 3, default to relevant_content
        if f'chunk_{i}' not in classification_mapping:
            classification_mapping[f'chunk_{i}'] = 'relevant_content'
    
    # Save classification mapping
    mapping_filename = f"{output_dir}/page_classification_mapping.json"
    with open(mapping_filename, 'w', encoding='utf-8') as f:
        json.dump(classification_mapping, f, indent=2)
    print(f"\nSaved classification mapping to {mapping_filename}")
    
    print(f"\nSplit document into {len(pages)} pages")
    
    # Print summary of classifications
    disclaimer_chunks = [chunk for chunk, classification in classification_mapping.items() 
                        if classification == 'disclaimer']
    relevant_chunks = [chunk for chunk, classification in classification_mapping.items() 
                      if classification == 'relevant_content']
    
    print(f"\nClassification Summary:")
    print(f"Relevant content chunks ({len(relevant_chunks)}): {', '.join(sorted(relevant_chunks, key=lambda x: int(x.split('_')[1])))}")
    print(f"Disclaimer chunks ({len(disclaimer_chunks)}): {', '.join(sorted(disclaimer_chunks, key=lambda x: int(x.split('_')[1])))}")
    
    return classification_mapping

def generate_folder_name(file_path):
    """Generate folder name from file name."""
    # Extract filename without path and extension
    filename = os.path.basename(file_path)
    # Remove .txt extension
    if filename.endswith('.txt'):
        filename = filename[:-4]
    
    # Clean up the name to be folder-friendly
    folder_name = filename.replace(' ', '_').replace(',', '').replace('-', '_').lower()
    # Remove multiple underscores
    while '__' in folder_name:
        folder_name = folder_name.replace('__', '_')
    
    return folder_name

if __name__ == "__main__":
    import sys
    
    # Default to alphabet file if no arguments provided
    if len(sys.argv) < 2:
        input_file = os.path.join(os.getenv("CONVERTED_TEXT_DIR", "./data/converted"), "sample.txt")
        output_dir = os.path.join(os.getenv("CHUNKS_DIR", "./data/chunks"), "sample")
    else:
        input_file = sys.argv[1]

        # Generate output directory based on filename
        base_dir = os.getenv("CHUNKS_DIR", "./data/chunks")
        folder_name = generate_folder_name(input_file)
        output_dir = os.path.join(base_dir, folder_name)
    
    classification_mapping = split_document_by_pages(input_file, output_dir)