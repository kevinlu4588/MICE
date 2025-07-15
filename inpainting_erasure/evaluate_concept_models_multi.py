#!/usr/bin/env python
"""
Evaluate the latest GA and multiple ESD models for a specific concept
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
    ga_index = -1
    esd_models = {}  # Dict to store different ESD methods
    
    # Normalize concept for comparison
    concept_normalized = concept.lower()
    
    for index, model_info in models_data.items():
        model_concept = model_info.get('concept_erased', '').lower()
        if model_concept == concept_normalized:
            idx = int(index)
            if 'Gradient Ascent' in model_info['method'] and idx > ga_index:
                ga_index = idx
                ga_model = model_info
            elif 'ESD' in model_info['method']:
                # Extract ESD method type from weights_trained field
                method_type = model_info.get('weights_trained', 'esd-x')
                if method_type not in esd_models or idx > int(esd_models[method_type]['index']):
                    esd_models[method_type] = {
                        'model': model_info,
                        'index': str(idx)
                    }
    
    return ga_model, ga_index, esd_models

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

def create_comparison_grid(base_images, ga_images, esd_images_dict, prompts, output_path, 
                          unrelated_base_images=None, unrelated_ga_images=None, 
                          unrelated_esd_images_dict=None, unrelated_prompts=None):
    """Create a comparison grid of base, GA, and multiple ESD results including unrelated concepts"""
    
    # Calculate total prompts (target + unrelated)
    total_prompts = prompts.copy()
    if unrelated_prompts:
        total_prompts.extend(unrelated_prompts)
    
    n_prompts = len(total_prompts)
    n_cols = 2 + len(esd_images_dict)  # Base + GA + ESD methods
    
    fig, axes = plt.subplots(n_prompts, n_cols, figsize=(5 * n_cols, 5 * n_prompts))
    
    if n_prompts == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Combine all images
    all_base_images = base_images + (unrelated_base_images or [])
    all_ga_images = ga_images + (unrelated_ga_images or [])
    
    # Combine ESD images
    all_esd_images_dict = {}
    for method_name in esd_images_dict:
        all_esd_images_dict[method_name] = esd_images_dict[method_name].copy()
        if unrelated_esd_images_dict and method_name in unrelated_esd_images_dict:
            all_esd_images_dict[method_name].extend(unrelated_esd_images_dict[method_name])
    
    # Add separator line before unrelated concepts
    separator_row = len(prompts)
    
    for i in range(n_prompts):
        col_idx = 0
        
        # Add visual separator
        if i == separator_row and unrelated_prompts:
            for j in range(n_cols):
                axes[i, j].axhline(y=0.5, color='black', linewidth=3)
                axes[i, j].text(0.5, 0.5, 'UNRELATED CONCEPTS', ha='center', va='center', 
                               fontsize=12, weight='bold', transform=axes[i, j].transAxes)
                axes[i, j].axis('off')
            continue
        
        # Adjust index for images after separator
        img_idx = i if i < separator_row else i - 1
        
        # Base model
        if img_idx < len(all_base_images):
            img = mpimg.imread(all_base_images[img_idx])
            axes[i, col_idx].imshow(img)
            axes[i, col_idx].set_title(f"Base SD v1.4\n{total_prompts[img_idx][:40]}...", fontsize=10)
            axes[i, col_idx].axis('off')
        col_idx += 1
        
        # GA model
        if img_idx < len(all_ga_images):
            img = mpimg.imread(all_ga_images[img_idx])
            axes[i, col_idx].imshow(img)
            axes[i, col_idx].set_title(f"GA Erased\n{total_prompts[img_idx][:40]}...", fontsize=10)
            axes[i, col_idx].axis('off')
        col_idx += 1
        
        # ESD models
        for method_name, images in sorted(all_esd_images_dict.items()):
            if img_idx < len(images):
                img = mpimg.imread(images[img_idx])
                axes[i, col_idx].imshow(img)
                axes[i, col_idx].set_title(f"ESD {method_name.upper()}\n{total_prompts[img_idx][:40]}...", fontsize=10)
                axes[i, col_idx].axis('off')
            col_idx += 1
    
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
    ga_model, ga_index, esd_models_dict = find_latest_models(args.concept, models_data)
    
    if not ga_model and not esd_models_dict:
        print(f"No models found for concept: {args.concept}")
        sys.exit(1)
    
    print(f"Found models for '{args.concept}':")
    if ga_model:
        print(f"  GA Model: {ga_model['name']} (index {ga_index})")
    for method_type, model_info in esd_models_dict.items():
        print(f"  ESD Model ({method_type}): {model_info['model']['name']} (index {model_info['index']})")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Define evaluation prompts based on concept type
    artist_concepts = ["van gogh", "picasso", "andy warhol", "monet", "banksy"]
    animal_concepts = ["english springer spaniel", "french horn"]
    
    # Define unrelated concepts for testing preservation
    unrelated_prompts = [
        "a photo of a red car",
        "a sports car on the road",
        "a vintage automobile",
        "a luxury vehicle in a showroom",
        "a car parked in a driveway"
    ]
    
    if args.concept.lower() in artist_concepts:
        # Artist prompts
        prompts = [
            f"a painting by {args.concept}",
            f"artwork in the style of {args.concept}",
            f"a masterpiece by {args.concept}",
            f"{args.concept} style painting of a landscape",
            f"a portrait in the style of {args.concept}"
        ][:args.num_prompts]
    elif args.concept.lower() in animal_concepts or "dog" in args.concept.lower() or "spaniel" in args.concept.lower():
        # Animal prompts
        prompts = [
            f"a photo of a {args.concept}",
            f"a {args.concept} sitting on grass",
            f"a close-up portrait of a {args.concept}",
            f"a {args.concept} playing in a park",
            f"a professional photo of a {args.concept}"
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
    
    # Use first 5 unrelated prompts regardless of num_prompts
    unrelated_prompts = unrelated_prompts[:5]
    
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
    esd_images_dict = {}
    
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
    
    # Load and evaluate ESD models
    for method_type, model_info in esd_models_dict.items():
        esd_model = model_info['model']
        print(f"\nLoading ESD model ({method_type}): {esd_model['name']}...")
        esd_path = Path(__file__).parent.parent / "models" / esd_model['name']
        if esd_path.exists():
            # Check if it's a diffusers format model or single file
            if (esd_path / "model_index.json").exists():
                # Diffusers format
                esd_pipeline = StableDiffusionPipeline.from_pretrained(
                    str(esd_path),
                    torch_dtype=torch.float16,
                    safety_checker=None,
                    requires_safety_checker=False
                ).to(device)
            elif (esd_path / "model.safetensors").exists():
                # Single file format - load base model and apply weights
                print(f"  Model is in single-file format, loading base SD v1.4 and applying weights...")
                esd_pipeline = StableDiffusionPipeline.from_pretrained(
                    "CompVis/stable-diffusion-v1-4",
                    torch_dtype=torch.float16,
                    safety_checker=None,
                    requires_safety_checker=False
                ).to(device)
                
                # Load the safetensors weights
                from safetensors.torch import load_file
                state_dict = load_file(esd_path / "model.safetensors")
                esd_pipeline.unet.load_state_dict(state_dict, strict=False)
            else:
                print(f"Warning: Unsupported model format for {esd_model['name']}")
                continue
                
            esd_images = generate_evaluation_images(esd_pipeline, prompts, output_dir, f"esd_{method_type}_erased")
            esd_images_dict[method_type] = esd_images
            del esd_pipeline
            torch.cuda.empty_cache()
        else:
            print(f"Warning: ESD model path not found: {esd_path}")
    
    # Generate unrelated concept images to test preservation
    print("\n" + "="*50)
    print("Generating unrelated concept images (cars)...")
    print("="*50)
    
    # Generate unrelated images with base model (reuse the existing pipeline)
    print("\nGenerating unrelated images with base model...")
    unrelated_base_images = generate_evaluation_images(base_pipeline, unrelated_prompts, output_dir, "unrelated_base")
    
    # Generate unrelated images with GA model
    unrelated_ga_images = []
    if ga_model:
        print("\nGenerating unrelated images with GA model...")
        ga_path = Path(__file__).parent.parent / "models" / ga_model['name']
        if ga_path.exists():
            ga_pipeline = StableDiffusionPipeline.from_pretrained(
                str(ga_path),
                torch_dtype=torch.float16,
                safety_checker=None,
                requires_safety_checker=False
            ).to(device)
            unrelated_ga_images = generate_evaluation_images(ga_pipeline, unrelated_prompts, output_dir, "unrelated_ga")
            del ga_pipeline
            torch.cuda.empty_cache()
    
    # Generate unrelated images with ESD models
    unrelated_esd_images_dict = {}
    for method_type, model_info in esd_models_dict.items():
        esd_model = model_info['model']
        print(f"\nGenerating unrelated images with ESD model ({method_type})...")
        esd_path = Path(__file__).parent.parent / "models" / esd_model['name']
        if esd_path.exists():
            # Check if it's a diffusers format model or single file
            if (esd_path / "model_index.json").exists():
                # Diffusers format
                esd_pipeline = StableDiffusionPipeline.from_pretrained(
                    str(esd_path),
                    torch_dtype=torch.float16,
                    safety_checker=None,
                    requires_safety_checker=False
                ).to(device)
            elif (esd_path / "model.safetensors").exists():
                # Single file format - load base model and apply weights
                esd_pipeline = StableDiffusionPipeline.from_pretrained(
                    "CompVis/stable-diffusion-v1-4",
                    torch_dtype=torch.float16,
                    safety_checker=None,
                    requires_safety_checker=False
                ).to(device)
                
                # Load the safetensors weights
                from safetensors.torch import load_file
                state_dict = load_file(esd_path / "model.safetensors")
                esd_pipeline.unet.load_state_dict(state_dict, strict=False)
            else:
                continue
                
            unrelated_images = generate_evaluation_images(esd_pipeline, unrelated_prompts, output_dir, f"unrelated_esd_{method_type}")
            unrelated_esd_images_dict[method_type] = unrelated_images
            del esd_pipeline
            torch.cuda.empty_cache()
    
    # Create comparison grid with both target and unrelated concepts
    print("\nCreating comparison grid...")
    grid_path = output_dir / f"{args.concept.replace(' ', '_')}_comparison_grid.png"
    create_comparison_grid(base_images, ga_images, esd_images_dict, prompts, grid_path,
                          unrelated_base_images, unrelated_ga_images, 
                          unrelated_esd_images_dict, unrelated_prompts)
    
    print(f"\nEvaluation complete!")
    print(f"Results saved to: {output_dir}")
    print(f"Comparison grid: {grid_path}")
    
    # Save evaluation metadata
    metadata = {
        "concept": args.concept,
        "ga_model": ga_model['name'] if ga_model else None,
        "esd_models": {k: v['model']['name'] for k, v in esd_models_dict.items()},
        "prompts": prompts,
        "unrelated_prompts": unrelated_prompts,
        "num_target_images": len(prompts),
        "num_unrelated_images": len(unrelated_prompts)
    }
    
    with open(output_dir / "evaluation_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    main()