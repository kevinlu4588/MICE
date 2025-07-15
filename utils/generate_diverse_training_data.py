#!/usr/bin/env python
"""
Generate diverse training data for concepts using varied prompts
"""

import os
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import argparse
from pathlib import Path
from tqdm import tqdm
import random

def get_diverse_prompts(concept, num_prompts):
    """Generate diverse prompts for a concept"""
    
    prompts = []
    
    # Artist concepts
    if concept.lower() in ["van gogh", "picasso", "andy warhol", "monet", "banksy"]:
        # Art styles and subjects
        styles = ["painting", "artwork", "masterpiece", "art piece", "creation", "work"]
        subjects = [
            "a landscape", "a portrait", "a still life", "flowers", "a city scene",
            "a rural scene", "a night scene", "a day scene", "people", "nature",
            "abstract forms", "geometric shapes", "a building", "trees", "water",
            "mountains", "a garden", "a room interior", "a street scene", "animals",
            "a seascape", "clouds", "a sunset", "a sunrise", "a field",
            "a bridge", "a boat", "a house", "a church", "a cafe scene"
        ]
        
        modifiers = [
            "beautiful", "colorful", "dramatic", "serene", "vibrant", "moody",
            "expressive", "detailed", "bold", "subtle", "dynamic", "peaceful"
        ]
        
        # Generate varied prompts
        for i in range(num_prompts):
            template = random.choice([
                f"{{subject}} by {concept}",
                f"a {{style}} of {{subject}} by {concept}",
                f"{concept} style {{style}} of {{subject}}",
                f"{{modifier}} {{subject}} in the style of {concept}",
                f"{{subject}} painted by {concept}",
                f"art by {concept} depicting {{subject}}",
                f"{{modifier}} {{style}} by {concept} showing {{subject}}",
                f"{concept}'s {{style}} of {{subject}}",
                f"{{subject}}, {{style}} by {concept}",
                f"a {{modifier}} {concept} style {{subject}}"
            ])
            
            prompt = template.format(
                style=random.choice(styles),
                subject=random.choice(subjects),
                modifier=random.choice(modifiers)
            )
            prompts.append(prompt)
    
    # Animal concepts
    elif concept.lower() in ["english springer spaniel", "dog", "cat", "bird"]:
        actions = [
            "sitting", "running", "playing", "sleeping", "jumping", "walking",
            "standing", "lying down", "looking at camera", "playing with a ball",
            "in motion", "resting", "alert", "happy", "relaxed"
        ]
        
        locations = [
            "in a garden", "on grass", "in a park", "indoors", "outdoors",
            "on a beach", "in nature", "on a couch", "in a field", "by a lake",
            "in the snow", "on a path", "in the woods", "on a deck", "in sunlight"
        ]
        
        descriptors = [
            "beautiful", "cute", "majestic", "playful", "elegant", "friendly",
            "energetic", "calm", "happy", "professional photo of", "portrait of",
            "close-up of", "full body shot of", "candid photo of"
        ]
        
        for i in range(num_prompts):
            template = random.choice([
                f"a {{descriptor}} {concept}",
                f"a {concept} {{action}}",
                f"a {concept} {{action}} {{location}}",
                f"{{descriptor}} a {concept} {{location}}",
                f"photo of a {concept} {{action}}",
                f"a {{descriptor}} {concept} {{action}} {{location}}",
                f"{concept} {{location}}",
                f"{{descriptor}} {concept}"
            ])
            
            prompt = template.format(
                descriptor=random.choice(descriptors),
                action=random.choice(actions),
                location=random.choice(locations)
            )
            prompts.append(prompt)
    
    # Vehicle concepts
    elif concept.lower() in ["airliner", "garbage truck", "car", "bus", "truck"]:
        settings = [
            "at an airport", "in flight", "on a runway", "in the sky", "landing",
            "taking off", "parked", "in motion", "from below", "from the side",
            "at sunset", "during daytime", "at an angle", "close-up view", "distant view"
        ] if "airliner" in concept.lower() else [
            "on a street", "in a city", "collecting garbage", "driving", "parked",
            "at work", "from the side", "from the front", "in a neighborhood",
            "during daytime", "on a road", "in motion", "stopped", "turning", "backing up"
        ]
        
        descriptors = [
            "a modern", "a large", "a", "a professional photo of a", "a detailed",
            "a high-quality photo of a", "an image of a", "a picture of a",
            "a photograph of a", "a clear view of a"
        ]
        
        for i in range(num_prompts):
            template = random.choice([
                f"{{descriptor}} {concept}",
                f"{{descriptor}} {concept} {{setting}}",
                f"{concept} {{setting}}",
                f"photo of {{descriptor}} {concept}",
                f"{{descriptor}} {concept}, {{setting}}"
            ])
            
            prompt = template.format(
                descriptor=random.choice(descriptors),
                setting=random.choice(settings)
            )
            prompts.append(prompt)
    
    # Object concepts
    elif concept.lower() in ["french horn", "golf ball", "chainsaw"]:
        contexts = {
            "french horn": [
                "on a table", "being played", "in an orchestra", "close-up", "shiny brass",
                "in a case", "on a stand", "professional photo", "musical instrument",
                "in a music room", "detailed view", "from the side", "polished", "golden"
            ],
            "golf ball": [
                "on grass", "on a tee", "close-up", "white", "with dimples visible",
                "on a golf course", "next to a club", "in flight", "on the green",
                "in a bucket", "professional photo", "detailed texture", "multiple balls", "clean"
            ],
            "chainsaw": [
                "on a workbench", "close-up", "professional tool", "detailed view",
                "industrial", "power tool", "on wood", "in a workshop", "from the side",
                "orange and black", "with chain visible", "heavy duty", "on the ground", "new"
            ]
        }
        
        descriptors = [
            "a", "a professional photo of a", "a detailed", "a high-quality",
            "a clear photo of a", "an image of a", "a picture of a", "a close-up of a"
        ]
        
        context_list = contexts.get(concept.lower(), ["on a table", "close-up", "detailed view"])
        
        for i in range(num_prompts):
            template = random.choice([
                f"{{descriptor}} {concept}",
                f"{{descriptor}} {concept}, {{context}}",
                f"{concept}, {{context}}",
                f"photo of {{descriptor}} {concept}",
                f"{{context}} {concept}"
            ])
            
            prompt = template.format(
                descriptor=random.choice(descriptors),
                context=random.choice(context_list)
            )
            prompts.append(prompt)
    
    # Building/structure concepts
    elif concept.lower() in ["church", "house", "building", "bridge"]:
        styles = [
            "gothic", "modern", "traditional", "old", "historic", "beautiful",
            "small", "large", "stone", "wooden", "brick", "white", "rural", "urban"
        ]
        
        settings = [
            "on a hill", "in a village", "in a city", "surrounded by trees",
            "at sunset", "during daytime", "with blue sky", "in the countryside",
            "from the front", "from an angle", "aerial view", "close-up", "distant view",
            "with people nearby", "empty", "peaceful setting"
        ]
        
        for i in range(num_prompts):
            template = random.choice([
                f"a {{style}} {concept}",
                f"a {{style}} {concept} {{setting}}",
                f"a {concept} {{setting}}",
                f"photo of a {{style}} {concept}",
                f"a beautiful {concept} {{setting}}",
                f"{{style}} {concept}, {{setting}}"
            ])
            
            prompt = template.format(
                style=random.choice(styles),
                setting=random.choice(settings)
            )
            prompts.append(prompt)
    
    else:
        # Generic object prompts
        for i in range(num_prompts):
            descriptors = ["a", "a photo of a", "an image of a", "a picture of a", 
                         "a professional photo of a", "a detailed", "a close-up of a"]
            settings = ["", "on a table", "close-up", "detailed view", "from the side",
                       "well-lit", "high quality", "clear image", "isolated", "centered"]
            
            desc = random.choice(descriptors)
            sett = random.choice(settings)
            
            if sett:
                prompts.append(f"{desc} {concept}, {sett}")
            else:
                prompts.append(f"{desc} {concept}")
    
    # Ensure uniqueness and correct count
    prompts = list(set(prompts))  # Remove duplicates
    while len(prompts) < num_prompts:
        # Add more if needed
        prompts.append(f"a high-quality photo of a {concept} (variation {len(prompts)})")
    
    return prompts[:num_prompts]

def generate_training_images(pipeline, concept, output_dir, num_images=100, seed=42):
    """Generate training images for a specific concept with diverse prompts"""
    
    # Create output directory
    concept_dir = Path(output_dir) / concept.replace(" ", "_")
    os.makedirs(concept_dir, exist_ok=True)
    
    # Get diverse prompts
    prompts = get_diverse_prompts(concept, num_images)
    
    # Save prompts for reference
    prompts_file = concept_dir / "prompts_used.txt"
    with open(prompts_file, 'w') as f:
        for i, prompt in enumerate(prompts):
            f.write(f"{i:03d}: {prompt}\n")
    
    print(f"Generated {len(prompts)} unique prompts for '{concept}'")
    print(f"Prompts saved to: {prompts_file}")
    
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
    
    print(f"Saved {len(prompts)} images to {concept_dir}")
    return concept_dir

def main():
    parser = argparse.ArgumentParser(description="Generate diverse training data for concept erasure")
    parser.add_argument("--concepts", nargs="+", 
                        default=["van gogh"],
                        help="Concepts to generate images for")
    parser.add_argument("--num_images", type=int, default=100, 
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

if __name__ == "__main__":
    main()