#!/usr/bin/env python
"""
Script to evaluate erased models by generating images for the erased concept and an unrelated concept
"""

import os
import torch
import argparse
from diffusers import StableDiffusionPipeline
from PIL import Image
import numpy as np
from pathlib import Path
from datetime import datetime

def create_image_grid(images, grid_size=(5, 2), title=None):
    """Create a grid of images with optional title"""
    if len(images) == 0:
        return None
    
    # Assuming all images are the same size
    img_width, img_height = images[0].size
    grid_width = img_width * grid_size[0]
    
    # Add space for title if provided
    title_height = 60 if title else 0
    grid_height = img_height * grid_size[1] + title_height
    
    # Create grid with white background
    grid = Image.new('RGB', (grid_width, grid_height), color='white')
    
    # Add title if provided
    if title:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(grid)
        
        # Try to use a nice font, fall back to default if not available
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 24)
            except:
                font = ImageFont.load_default()
        
        # Calculate text position (centered)
        text_bbox = draw.textbbox((0, 0), title, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_x = (grid_width - text_width) // 2
        text_y = 10
        
        # Draw text with black color
        draw.text((text_x, text_y), title, fill='black', font=font)
    
    # Paste images into grid
    for idx, img in enumerate(images):
        if idx >= grid_size[0] * grid_size[1]:
            break
        row = idx // grid_size[0]
        col = idx % grid_size[0]
        y_offset = title_height
        grid.paste(img, (col * img_width, row * img_height + y_offset))
    
    return grid

def generate_images(pipeline, prompt, num_images, seed=42):
    """Generate images with a given prompt"""
    images = []
    generator = torch.Generator(device=pipeline.device).manual_seed(seed)
    
    for i in range(num_images):
        # Use different seeds for variety
        current_generator = torch.Generator(device=pipeline.device).manual_seed(seed + i)
        image = pipeline(
            prompt,
            generator=current_generator,
            num_inference_steps=50,
            guidance_scale=7.5
        ).images[0]
        images.append(image)
    
    return images

def main():
    parser = argparse.ArgumentParser(description="Evaluate erased model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the erased model")
    parser.add_argument("--erased_concept", type=str, default="van gogh", help="The erased concept to test")
    parser.add_argument("--unrelated_concept", type=str, default="a beautiful sunset over mountains", help="An unrelated concept to test")
    parser.add_argument("--num_images", type=int, default=10, help="Number of images to generate per concept")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="Output directory for results")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Create output directory structure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = Path(args.output_dir) / f"eval_{timestamp}"
    erased_dir = output_base / "erased_concept"
    unrelated_dir = output_base / "unrelated_concept"
    
    os.makedirs(erased_dir, exist_ok=True)
    os.makedirs(unrelated_dir, exist_ok=True)
    
    # Load the model
    print(f"Loading model from {args.model_path}")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    pipeline = StableDiffusionPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False
    ).to(device)
    
    # Generate images for erased concept
    erased_prompt = f"a painting by {args.erased_concept}" if "style" not in args.erased_concept else args.erased_concept
    print(f"\nGenerating {args.num_images} images for erased concept: '{erased_prompt}'")
    erased_images = generate_images(pipeline, erased_prompt, args.num_images)
    
    # Save individual erased concept images
    for i, img in enumerate(erased_images):
        img.save(erased_dir / f"erased_{i:02d}.png")
    
    # Create and save grid for erased concept
    erased_grid = create_image_grid(erased_images, grid_size=(5, 2), title=f"Prompt: {erased_prompt}")
    if erased_grid:
        erased_grid.save(output_base / "erased_concept_grid.png")
        print(f"Saved erased concept grid to {output_base / 'erased_concept_grid.png'}")
    
    # Generate images for unrelated concept
    print(f"\nGenerating {args.num_images} images for unrelated concept: '{args.unrelated_concept}'")
    unrelated_images = generate_images(pipeline, args.unrelated_concept, args.num_images, seed=123)
    
    # Save individual unrelated concept images
    for i, img in enumerate(unrelated_images):
        img.save(unrelated_dir / f"unrelated_{i:02d}.png")
    
    # Create and save grid for unrelated concept
    unrelated_grid = create_image_grid(unrelated_images, grid_size=(5, 2), title=f"Prompt: {args.unrelated_concept}")
    if unrelated_grid:
        unrelated_grid.save(output_base / "unrelated_concept_grid.png")
        print(f"Saved unrelated concept grid to {output_base / 'unrelated_concept_grid.png'}")
    
    # Save evaluation info
    info = {
        "model_path": args.model_path,
        "erased_concept": args.erased_concept,
        "erased_prompt": erased_prompt,
        "unrelated_concept": args.unrelated_concept,
        "num_images": args.num_images,
        "timestamp": timestamp
    }
    
    import json
    with open(output_base / "evaluation_info.json", "w") as f:
        json.dump(info, f, indent=2)
    
    print(f"\nEvaluation complete! Results saved to {output_base}")
    print(f"- Erased concept images: {erased_dir}")
    print(f"- Unrelated concept images: {unrelated_dir}")
    print(f"- Image grids: {output_base}")

if __name__ == "__main__":
    main()