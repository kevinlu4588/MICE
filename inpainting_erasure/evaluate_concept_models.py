#!/usr/bin/env python
"""
Evaluate the latest GA and ESD models for a specific concept
"""

import os
import sys
import json
import torch
import argparse
from pathlib import Path
from diffusers import StableDiffusionPipeline
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def load_models_json():
    """Load the models.json file"""
    models_json_path = Path(__file__).parent.parent / "utils" / "models.json"
    with open(models_json_path, 'r') as f:
        return json.load(f)

def find_latest_models(concept, models_data):
    """Find the latest GA and ESD models for a concept"""
    ga_model = None
    esd_model = None
    ga_index = -1
    esd_index = -1
    
    # Normalize concept for comparison
    concept_normalized = concept.lower()
    
    for index, model_info in models_data.items():
        model_concept = model_info.get('concept_erased', '').lower()
        if model_concept == concept_normalized:
            idx = int(index)
            if 'Gradient Ascent' in model_info['method'] and idx > ga_index:
                ga_index = idx
                ga_model = model_info
            elif 'ESD' in model_info['method'] and idx > esd_index:
                esd_index = idx
                esd_model = model_info
    
    return ga_model, ga_index, esd_model, esd_index

def generate_evaluation_images(pipeline, prompts, output_dir, model_name):
    """Generate evaluation images for a model"""
    model_dir = output_dir / model_name
    os.makedirs(model_dir, exist_ok=True)
    
    images = []
    for i, prompt in enumerate(tqdm(prompts, desc=f"Generating {model_name} images")):
        with torch.no_grad():
            image = pipeline(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
            image_path = model_dir / f"{i:03d}_{prompt[:30].replace(' ', '_')}.png"
            image.save(image_path)
            images.append(image_path)
    
    return images

def create_comparison_grid(base_images, ga_images, esd_images, prompts, output_path):
    """Create a comparison grid of base, GA, and ESD results"""
    n_prompts = len(prompts)
    fig, axes = plt.subplots(n_prompts, 3, figsize=(15, 5 * n_prompts))
    
    if n_prompts == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_prompts):
        # Base model
        if i < len(base_images):
            img = mpimg.imread(base_images[i])
            axes[i, 0].imshow(img)
            axes[i, 0].set_title(f"Base SD v1.4\n{prompts[i][:40]}...", fontsize=10)
            axes[i, 0].axis('off')
        
        # GA model
        if i < len(ga_images):
            img = mpimg.imread(ga_images[i])
            axes[i, 1].imshow(img)
            axes[i, 1].set_title(f"GA Erased\n{prompts[i][:40]}...", fontsize=10)
            axes[i, 1].axis('off')
        
        # ESD model
        if i < len(esd_images):
            img = mpimg.imread(esd_images[i])
            axes[i, 2].imshow(img)
            axes[i, 2].set_title(f"ESD Erased\n{prompts[i][:40]}...", fontsize=10)
            axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Evaluate concept erasure models")
    parser.add_argument("--concept", type=str, required=True, help="Concept to evaluate")
    parser.add_argument("--num_prompts", type=int, default=5, help="Number of evaluation prompts")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for results")
    
    args = parser.parse_args()
    
    # Load models data
    models_data = load_models_json()
    
    # Find latest models for concept
    ga_model, ga_index, esd_model, esd_index = find_latest_models(args.concept, models_data)
    
    if not ga_model and not esd_model:
        print(f"No models found for concept: {args.concept}")
        sys.exit(1)
    
    print(f"Found models for '{args.concept}':")
    if ga_model:
        print(f"  GA Model: {ga_model['name']} (index {ga_index})")
    if esd_model:
        print(f"  ESD Model: {esd_model['name']} (index {esd_index})")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Define evaluation prompts based on concept type
    if args.concept.lower() in ["van gogh", "picasso", "andy warhol", "monet", "banksy"]:
        # Artist prompts
        prompts = [
            f"a painting by {args.concept}",
            f"artwork in the style of {args.concept}",
            f"a masterpiece by {args.concept}",
            f"{args.concept} style painting of a landscape",
            f"a portrait in the style of {args.concept}"
        ][:args.num_prompts]
    else:
        # Object prompts
        prompts = [
            f"a photo of a {args.concept}",
            f"a {args.concept} on a table",
            f"a close-up of a {args.concept}",
            f"a {args.concept} in nature",
            f"a professional photo of a {args.concept}"
        ][:args.num_prompts]
    
    # Load base model
    print("\nLoading base SD v1.4 model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_pipeline = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False
    ).to(device)
    
    # Generate base model images
    print("\nGenerating base model images...")
    base_images = generate_evaluation_images(base_pipeline, prompts, output_dir, "base_sd14")
    
    ga_images = []
    esd_images = []
    
    # Load and evaluate GA model
    if ga_model:
        print(f"\nLoading GA model: {ga_model['name']}...")
        ga_path = Path(__file__).parent.parent / "models" / ga_model['name']
        if ga_path.exists():
            ga_pipeline = StableDiffusionPipeline.from_pretrained(
                str(ga_path),
                torch_dtype=torch.float16,
                safety_checker=None,
                requires_safety_checker=False
            ).to(device)
            ga_images = generate_evaluation_images(ga_pipeline, prompts, output_dir, "ga_erased")
            del ga_pipeline
            torch.cuda.empty_cache()
        else:
            print(f"Warning: GA model path not found: {ga_path}")
    
    # Load and evaluate ESD model
    if esd_model:
        print(f"\nLoading ESD model: {esd_model['name']}...")
        esd_path = Path(__file__).parent.parent / "models" / esd_model['name']
        if esd_path.exists():
            esd_pipeline = StableDiffusionPipeline.from_pretrained(
                str(esd_path),
                torch_dtype=torch.float16,
                safety_checker=None,
                requires_safety_checker=False
            ).to(device)
            esd_images = generate_evaluation_images(esd_pipeline, prompts, output_dir, "esd_erased")
            del esd_pipeline
            torch.cuda.empty_cache()
        else:
            print(f"Warning: ESD model path not found: {esd_path}")
    
    # Create comparison grid
    print("\nCreating comparison grid...")
    grid_path = output_dir / f"{args.concept.replace(' ', '_')}_comparison_grid.png"
    create_comparison_grid(base_images, ga_images, esd_images, prompts, grid_path)
    
    print(f"\nEvaluation complete!")
    print(f"Results saved to: {output_dir}")
    print(f"Comparison grid: {grid_path}")
    
    # Save evaluation metadata
    metadata = {
        "concept": args.concept,
        "ga_model": ga_model['name'] if ga_model else None,
        "esd_model": esd_model['name'] if esd_model else None,
        "prompts": prompts,
        "num_images": len(prompts)
    }
    
    with open(output_dir / "evaluation_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    main()