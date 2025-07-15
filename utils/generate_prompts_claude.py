#!/usr/bin/env python
"""
Generate diverse prompts for concepts using Claude API and save to CSV files
"""

import os
import csv
import json
import argparse
from pathlib import Path
from anthropic import Anthropic
from dotenv import load_dotenv

def get_claude_prompts(client, concept, num_prompts=100):
    """Get diverse prompts from Claude for a specific concept"""
    
    # Determine concept type for better prompting
    artist_concepts = ["van gogh", "picasso", "andy warhol", "monet", "banksy"]
    animal_concepts = ["english springer spaniel", "dog", "cat", "bird"]
    vehicle_concepts = ["airliner", "garbage truck", "car", "bus", "truck"]
    object_concepts = ["french horn", "golf ball", "chainsaw"]
    building_concepts = ["church", "house", "building", "bridge"]
    
    if concept.lower() in artist_concepts:
        concept_type = "artist"
        context = f"{concept} is a famous artist known for their distinctive style."
    elif concept.lower() in animal_concepts:
        concept_type = "animal"
        context = f"{concept} is an animal/breed."
    elif concept.lower() in vehicle_concepts:
        concept_type = "vehicle"
        context = f"{concept} is a type of vehicle."
    elif concept.lower() in object_concepts:
        concept_type = "object"
        context = f"{concept} is a physical object."
    elif concept.lower() in building_concepts:
        concept_type = "building/structure"
        context = f"{concept} is a type of building or structure."
    else:
        concept_type = "general"
        context = f"{concept} is the subject."
    
    prompt = f"""Generate exactly {num_prompts} diverse and creative text prompts for generating images of '{concept}'. 
{context}

Requirements:
1. Each prompt should be unique and varied
2. Prompts should range from simple to detailed
3. Include different perspectives, settings, styles, and contexts
4. For artists, vary the subjects, styles, and types of artwork
5. For objects/animals, vary the settings, actions, and descriptions
6. Avoid repetitive patterns
7. Make prompts suitable for text-to-image generation
8. Keep each prompt under 100 characters

Return ONLY a JSON array of strings, with no additional text or explanation. Example format:
["prompt 1", "prompt 2", "prompt 3"]"""

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}]
    )
    
    # Parse the response
    try:
        content = response.content[0].text.strip()
        # Try to extract JSON from the response
        if content.startswith('['):
            prompts = json.loads(content)
        else:
            # Try to find JSON array in the response
            start = content.find('[')
            end = content.rfind(']') + 1
            if start != -1 and end != 0:
                prompts = json.loads(content[start:end])
            else:
                raise ValueError("Could not find JSON array in response")
        
        # Ensure we have exactly num_prompts
        if len(prompts) > num_prompts:
            prompts = prompts[:num_prompts]
        elif len(prompts) < num_prompts:
            print(f"Warning: Only got {len(prompts)} prompts for {concept}, requesting more...")
            # Request additional prompts
            additional_needed = num_prompts - len(prompts)
            additional_prompt = f"Generate {additional_needed} more diverse prompts for '{concept}', different from these: {prompts[:5]}... Return ONLY a JSON array."
            
            additional_response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2000,
                messages=[{"role": "user", "content": additional_prompt}]
            )
            
            additional_content = additional_response.content[0].text.strip()
            if additional_content.startswith('['):
                additional_prompts = json.loads(additional_content)
                prompts.extend(additional_prompts[:additional_needed])
        
        return prompts[:num_prompts]
        
    except Exception as e:
        print(f"Error parsing Claude response for {concept}: {e}")
        print(f"Response was: {response.content[0].text[:500]}...")
        return []

def save_prompts_to_csv(concept, prompts, output_dir):
    """Save prompts to a CSV file"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    csv_path = output_dir / f"{concept.replace(' ', '_')}_prompts.csv"
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['index', 'prompt'])  # Header
        for i, prompt in enumerate(prompts):
            writer.writerow([i, prompt])
    
    print(f"Saved {len(prompts)} prompts to {csv_path}")
    return csv_path

def main():
    parser = argparse.ArgumentParser(description="Generate prompts using Claude API")
    parser.add_argument("--concepts", nargs="+", 
                        default=["van gogh", "picasso", "andy warhol", "airliner", 
                                "golf ball", "french horn", "english springer spaniel", 
                                "garbage truck", "church"],
                        help="Concepts to generate prompts for")
    parser.add_argument("--num_prompts", type=int, default=100, 
                        help="Number of prompts per concept")
    parser.add_argument("--output_dir", type=str, default="../datasets/prompt_csvs",
                        help="Output directory for CSV files")
    parser.add_argument("--api_key", type=str, default=None,
                        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")
    
    args = parser.parse_args()
    
    # Load environment variables from .env file in root directory
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded environment variables from {env_path}")
    
    # Initialize Claude client
    api_key = args.api_key or os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print("Error: Please provide an API key via --api_key or ANTHROPIC_API_KEY environment variable")
        print("You can get your API key from: https://console.anthropic.com/settings/keys")
        print(f"Checked for .env file at: {env_path}")
        return
    
    client = Anthropic(api_key=api_key)
    
    print(f"Generating {args.num_prompts} prompts for {len(args.concepts)} concepts...")
    
    # Process each concept
    all_results = {}
    for concept in args.concepts:
        print(f"\nProcessing: {concept}")
        prompts = get_claude_prompts(client, concept, args.num_prompts)
        
        if prompts:
            csv_path = save_prompts_to_csv(concept, prompts, args.output_dir)
            all_results[concept] = {
                'csv_path': str(csv_path),
                'num_prompts': len(prompts)
            }
        else:
            print(f"Failed to generate prompts for {concept}")
    
    # Save summary
    summary_path = Path(args.output_dir) / "generation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nAll prompts generated!")
    print(f"CSV files saved in: {args.output_dir}/")
    print(f"Summary saved to: {summary_path}")

if __name__ == "__main__":
    main()