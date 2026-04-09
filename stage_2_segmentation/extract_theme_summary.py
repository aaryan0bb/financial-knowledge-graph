#!/usr/bin/env python3

import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def extract_theme_summary_from_txt(txt_path: str, output_path: str) -> str:
    """Extract theme summary from a plain text file using GPT-4o-mini."""

    client = OpenAI()

    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"Text file not found: {txt_path}")

    print(f"Processing text file: {txt_path}")

    try:
        # Read text and build prompt inline (OpenAI doesn't accept text/plain uploads)
        with open(txt_path, "r", encoding="utf-8") as txt_file:
            raw_text = txt_file.read()

        snippet = raw_text[:20000]  # limit to keep token usage reasonable

        prompt = (

            "Please extract summary and key important theme of the markdown  file that is part of sell/buy side research report on financial markets. "
            "**When you extract the theme please make sure to no include the details of the publisher or author.** "
            "I just want you to focus on the theme of the document. Make sure your output does not exceed 3 sentences"
            + snippet
        )

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )

        theme_summary = completion.choices[0].message.content.strip()

        # Persist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(theme_summary)

        print("Theme summary extracted and saved.")

        return theme_summary

    except Exception as e:
        print(f"Error during theme extraction: {e}")
        raise

def main():
    """Main function to extract theme summary"""
    
    # Input TXT path
    txt_path = os.path.join(os.getenv("CONVERTED_TEXT_DIR", "./data/converted"), "sample_report.txt")

    # Output theme file path
    output_path = os.path.join(os.getenv("CHUNKS_DIR", "./data/chunks"), "themes/theme.txt")
    
    print("Starting PDF theme extraction...")
    
    try:
        theme_summary = extract_theme_summary_from_txt(txt_path, output_path)
        
        print("\n" + "="*50)
        print("THEME EXTRACTION COMPLETE")
        print("="*50)
        print(f"Text processed: {txt_path}")
        print(f"Theme saved to: {output_path}")
        print(f"\nTheme Summary:")
        print(f"{theme_summary}")

        
        return theme_summary
        
    except Exception as e:
        print(f"Failed to extract theme: {e}")
        return None

if __name__ == "__main__":
    theme_summary = main()
    theme_summary_file = os.path.join(os.getenv("CHUNKS_DIR", "./data/chunks"), "themes/theme_sample_1.txt")
    with open(theme_summary_file, 'w') as f:
        f.write(theme_summary)