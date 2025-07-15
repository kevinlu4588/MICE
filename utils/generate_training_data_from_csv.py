#!/usr/bin/env python
"""
Generate training data using prompts from CSV files
"""

import os
import csv
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import argparse
from pathlib import Path
from tqdm import tqdm

def load_prompts_from_csv(csv_path):
    """Load prompts from a CSV file"""
    prompts = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompts.append(row['prompt'])
    return prompts

def generate_training_images(pipeline, concept, prompts, output_dir, seed=42):
    """Generate training images using provided prompts"""
    
    # Create output directory
    concept_dir = Path(output_dir) / concept.replace(" ", "_")
    os.makedirs(concept_dir, exist_ok=True)
    
    print(f"Generating {len(prompts)} images for '{concept}'")
    
    # Generate images
    generator = torch.Generator(device=pipeline.device).manual_seed(seed)
    
    for i, prompt in enumerate(tqdm(prompts, desc=f"Generating images for {concept}")):
        # Use different seed for each image
        img_generator = torch.Generator(device=pipeline.device).manual_seed(seed + i)
        
        with torch.no_grad():
            image = pipeline(
                prompt, 
                generator=img_generator,
                num_inference_steps=50,
                guidance_scale=7.5
            ).images[0]
        
        # Save with meaningful filename
        safe_prompt = prompt[:50].replace("/", "_").replace(" ", "_")
        image_path = concept_dir / f"{i:03d}_{safe_prompt}.png"
        image.save(image_path)
    
    # Also save the prompts used for reference
    prompts_file = concept_dir / "prompts_used.txt"
    with open(prompts_file, 'w') as f:
        for i, prompt in enumerate(prompts):
            f.write(f"{i:03d}: {prompt}\n")
    
    print(f"Saved {len(prompts)} images to {concept_dir}")
    return concept_dir

def main():
    parser = argparse.ArgumentParser(description="Generate training data from CSV prompts")
    parser.add_argument("--concepts", nargs="+", 
                        help="Concepts to generate images for (must have corresponding CSV files)")
    parser.add_argument("--csv_dir", type=str, default="../datasets/prompt_csvs",
                        help="Directory containing prompt CSV files")
    parser.add_argument("--num_images", type=int, default=None, 
                        help="Number of images per concept (default: use all prompts in CSV)")
    parser.add_argument("--output_dir", type=str, default="../datasets/training_images",
                        help="Output directory for training images")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # If no concepts specified, find all CSV files
    if not args.concepts:
        csv_dir = Path(args.csv_dir)
        if not csv_dir.exists():
            print(f"Error: CSV directory '{args.csv_dir}' not found")
            print("Please run generate_prompts_claude.py first to generate prompt CSVs")
            return
        
        # Find all concept CSV files
        csv_files = list(csv_dir.glob("*_prompts.csv"))
        if not csv_files:
            print(f"No prompt CSV files found in {args.csv_dir}")
            return
        
        args.concepts = []
        for csv_file in csv_files:
            # Extract concept from filename
            concept = csv_file.stem.replace("_prompts", "").replace("_", " ")
            args.concepts.append(concept)
        
        print(f"Found CSV files for concepts: {args.concepts}")
    
    # Load the model
    print("\nLoading Stable Diffusion model...")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    model_id = "CompVis/stable-diffusion-v1-4"
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False
    ).to(device)
    
    print(f"Model loaded on {device}")
    
    # Generate images for each concept
    for i, concept in enumerate(args.concepts):
        print(f"\n[{i+1}/{len(args.concepts)}] Processing concept: {concept}")
        
        # Find corresponding CSV file
        csv_path = Path(args.csv_dir) / f"{concept.replace(' ', '_')}_prompts.csv"
        if not csv_path.exists():
            print(f"Warning: CSV file not found for '{concept}' at {csv_path}")
            continue
        
        # Load prompts
        prompts = load_prompts_from_csv(csv_path)
        print(f"Loaded {len(prompts)} prompts from {csv_path}")
        
        # Limit number of prompts if specified
        if args.num_images and args.num_images < len(prompts):
            prompts = prompts[:args.num_images]
            print(f"Using first {args.num_images} prompts")
        
        # Use different base seed for each concept
        concept_seed = 42 + i * 1000
        
        generate_training_images(
            pipeline,
            concept,
            prompts,
            args.output_dir,
            seed=concept_seed
        )
    
    print(f"\nAll training data generated successfully!")
    print(f"Images saved in: {args.output_dir}/")

if __name__ == "__main__":
    main()