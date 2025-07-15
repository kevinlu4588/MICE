#!/usr/bin/env python
"""
Generate training data for multiple concepts using Stable Diffusion
"""

import os
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import argparse
from pathlib import Path
from tqdm import tqdm

def generate_training_images(pipeline, concept, output_dir, num_images=50, seed=42):
    """Generate training images for a specific concept"""
    
    # Create output directory
    concept_dir = Path(output_dir) / concept.replace(" ", "_")
    os.makedirs(concept_dir, exist_ok=True)
    
    # Define prompts based on concept type
    if concept in ["picasso", "andy warhol", "van gogh"]:
        # Artist concepts
        prompts = [
            f"a painting by {concept}",
            f"artwork in the style of {concept}",
            f"a masterpiece by {concept}",
            f"art by {concept}",
            f"{concept} style painting",
            f"a portrait by {concept}",
            f"a landscape by {concept}",
            f"abstract art by {concept}",
            f"a famous painting by {concept}",
            f"artwork created by {concept}"
        ]
    elif concept in ["english springer spaniel"]:
        # Dog breed
        prompts = [
            f"a photo of an {concept}",
            f"a beautiful {concept}",
            f"a cute {concept} dog",
            f"an {concept} sitting",
            f"an {concept} running",
            f"a portrait of an {concept}",
            f"an {concept} playing",
            f"a professional photo of an {concept}",
            f"an {concept} outdoors",
            f"a close-up of an {concept}"
        ]
    else:
        # Objects (airliner, chainsaw, golf ball)
        article = "an" if concept[0].lower() in 'aeiou' else "a"
        prompts = [
            f"a photo of {article} {concept}",
            f"{article} {concept}",
            f"a detailed photo of {article} {concept}",
            f"{article} {concept} on white background",
            f"a professional photo of {article} {concept}",
            f"a close-up of {article} {concept}",
            f"{article} {concept} in use",
            f"a high-quality image of {article} {concept}",
            f"{article} {concept} product photo",
            f"a realistic {concept}"
        ]
    
    print(f"Generating {num_images} images for concept: {concept}")
    
    # Generate images
    image_count = 0
    generator = torch.Generator(device=pipeline.device).manual_seed(seed)
    
    with tqdm(total=num_images, desc=f"Generating {concept} images") as pbar:
        while image_count < num_images:
            # Cycle through prompts
            prompt = prompts[image_count % len(prompts)]
            
            # Generate with different seeds for variety
            current_generator = torch.Generator(device=pipeline.device).manual_seed(seed + image_count)
            
            try:
                image = pipeline(
                    prompt,
                    generator=current_generator,
                    num_inference_steps=50,
                    guidance_scale=7.5
                ).images[0]
                
                # Save image
                filename = f"{concept.replace(' ', '_')}_{image_count:04d}.png"
                image.save(concept_dir / filename)
                
                image_count += 1
                pbar.update(1)
                
            except Exception as e:
                print(f"Error generating image with prompt '{prompt}': {e}")
                # Continue with next seed
                seed += 1
    
    print(f"Saved {image_count} images to {concept_dir}")
    return concept_dir

def main():
    parser = argparse.ArgumentParser(description="Generate training data for concept erasure")
    parser.add_argument("--concepts", nargs="+", 
                        default=["picasso", "andy warhol", "english springer spaniel", 
                                "airliner", "chainsaw", "golf ball"],
                        help="Concepts to generate images for")
    parser.add_argument("--num_images", type=int, default=50, 
                        help="Number of images per concept")
    parser.add_argument("--output_dir", type=str, default="training_images",
                        help="Output directory for training images")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Load the model
    print("Loading Stable Diffusion model...")
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
        
        # Use different base seed for each concept
        concept_seed = 42 + i * 1000
        
        generate_training_images(
            pipeline,
            concept,
            args.output_dir,
            num_images=args.num_images,
            seed=concept_seed
        )
    
    print(f"\nAll training data generated successfully!")
    print(f"Images saved in: {args.output_dir}/")
    
    # Print summary
    print("\nSummary:")
    for concept in args.concepts:
        concept_dir = Path(args.output_dir) / concept.replace(" ", "_")
        if concept_dir.exists():
            num_files = len(list(concept_dir.glob("*.png")))
            print(f"  - {concept}: {num_files} images")

if __name__ == "__main__":
    main()