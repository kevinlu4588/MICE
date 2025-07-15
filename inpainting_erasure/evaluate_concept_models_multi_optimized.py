#!/usr/bin/env python
"""
Optimized evaluation script that generates both target and unrelated concept images
Reuses loaded pipelines to avoid reloading models
"""

import argparse
import os
import json
import torch
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from diffusers import StableDiffusionPipeline
from safetensors.torch import load_file
from tqdm import tqdm

def get_models_for_concept(concept):
    """Load models.json and find models for a specific concept"""
    models_json_path = Path(__file__).parent.parent / "utils" / "models.json"
    with open(models_json_path, 'r') as f:
        models_data = json.load(f)
    
    ga_model = None
    esd_models = {'cross_attention': None, 'esd-x': None, 'esd-u': None}
    
    # Normalize concept for comparison
    concept_normalized = concept.lower()
    
    # Find the latest models for this concept
    for index in sorted(models_data.keys(), key=int, reverse=True):
        model_info = models_data[index]
        if model_info.get('concept_erased', '').lower() == concept_normalized:
            if 'gradient ascent' in model_info['method'].lower() and ga_model is None:
                ga_model = model_info
            elif 'esd' in model_info['method'].lower():
                # Determine ESD type
                if 'cross_attention' in model_info.get('method', '').lower() or model_info['name'].endswith('_xattn'):
                    if esd_models['cross_attention'] is None:
                        esd_models['cross_attention'] = model_info
                elif 'esd-u' in model_info.get('train_method', ''):
                    if esd_models['esd-u'] is None:
                        esd_models['esd-u'] = model_info
                elif 'esd-x' in model_info.get('train_method', '') or 'esd' in model_info['method'].lower():
                    if esd_models['esd-x'] is None:
                        esd_models['esd-x'] = model_info
    
    # Create a dict with found models
    esd_models_dict = {k: {'model': v} for k, v in esd_models.items() if v is not None}
    
    return ga_model, esd_models_dict

def load_pipeline_smart(model_path, device):
    """Load a pipeline, handling both diffusers format and single file format"""
    if (model_path / "model_index.json").exists():
        # Diffusers format
        pipeline = StableDiffusionPipeline.from_pretrained(
            str(model_path),
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False
        ).to(device)
    elif (model_path / "model.safetensors").exists():
        # Single file format - load base model and apply weights
        print(f"  Model is in single-file format, loading base SD v1.4 and applying weights...")
        pipeline = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False
        ).to(device)
        
        # Load the safetensors weights
        state_dict = load_file(model_path / "model.safetensors")
        pipeline.unet.load_state_dict(state_dict, strict=False)
    else:
        raise ValueError(f"Unsupported model format at {model_path}")
    
    return pipeline

def generate_evaluation_images(pipeline, prompts, output_dir, prefix):
    """Generate images for evaluation"""
    images = []
    os.makedirs(output_dir / prefix, exist_ok=True)
    
    desc = f"Generating {prefix} images"
    for i, prompt in enumerate(tqdm(prompts, desc=desc)):
        # Generate image
        with torch.cuda.amp.autocast():
            image = pipeline(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
        
        # Save image
        filename = f"{i:03d}_{prompt[:30].replace(' ', '_')}.png"
        image_path = output_dir / prefix / filename
        image.save(image_path)
        images.append(image_path)
    
    return images

def create_comparison_grid_with_unrelated(all_results, output_path):
    """Create a comparison grid including unrelated concepts"""
    # Extract data
    target_prompts = all_results['target_prompts']
    unrelated_prompts = all_results['unrelated_prompts']
    models = all_results['models']
    
    # Calculate grid dimensions
    n_target = len(target_prompts)
    n_unrelated = len(unrelated_prompts)
    n_total_rows = n_target + 1 + n_unrelated  # +1 for separator
    n_cols = len(models)
    
    # Create figure
    fig, axes = plt.subplots(n_total_rows, n_cols, figsize=(5 * n_cols, 5 * n_total_rows))
    
    if n_total_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot images
    row = 0
    
    # Target concept images
    for i in range(n_target):
        for col, (model_name, model_data) in enumerate(models.items()):
            if i < len(model_data['target_images']):
                img = mpimg.imread(model_data['target_images'][i])
                axes[row, col].imshow(img)
                axes[row, col].set_title(f"{model_name}\n{target_prompts[i][:40]}...", fontsize=10)
                axes[row, col].axis('off')
        row += 1
    
    # Separator row
    for col in range(n_cols):
        axes[row, col].text(0.5, 0.5, 'UNRELATED CONCEPTS (Cars)', 
                           ha='center', va='center', fontsize=14, weight='bold',
                           transform=axes[row, col].transAxes,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        axes[row, col].axis('off')
    row += 1
    
    # Unrelated concept images
    for i in range(n_unrelated):
        for col, (model_name, model_data) in enumerate(models.items()):
            if i < len(model_data['unrelated_images']):
                img = mpimg.imread(model_data['unrelated_images'][i])
                axes[row, col].imshow(img)
                axes[row, col].set_title(f"{model_name}\n{unrelated_prompts[i][:40]}...", fontsize=10)
                axes[row, col].axis('off')
        row += 1
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Evaluate concept erasure models")
    parser.add_argument("--concept", type=str, required=True, help="Concept to evaluate")
    parser.add_argument("--num_prompts", type=int, default=5, help="Number of prompts to evaluate")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    args = parser.parse_args()
    
    # Find models for this concept
    ga_model, esd_models_dict = get_models_for_concept(args.concept)
    
    print(f"Found models for '{args.concept}':")
    if ga_model:
        print(f"  GA Model: {ga_model['name']} (index {ga_model.get('index', 'N/A')})")
    for method_type, model_info in esd_models_dict.items():
        print(f"  ESD Model ({method_type}): {model_info['model']['name']} (index {model_info['model'].get('index', 'N/A')})")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Define prompts
    artist_concepts = ["van gogh", "picasso", "andy warhol", "monet", "banksy"]
    
    # Target prompts
    if args.concept.lower() in artist_concepts:
        target_prompts = [
            f"a painting by {args.concept}",
            f"artwork in the style of {args.concept}",
            f"a masterpiece by {args.concept}",
            f"{args.concept} style painting of a landscape",
            f"a portrait in the style of {args.concept}"
        ][:args.num_prompts]
    else:
        target_prompts = [
            f"a photo of a {args.concept}",
            f"a {args.concept} on a table",
            f"a close-up of a {args.concept}",
            f"a {args.concept} in nature",
            f"a professional photo of a {args.concept}"
        ][:args.num_prompts]
    
    # Unrelated prompts (always cars)
    unrelated_prompts = [
        "a photo of a red car",
        "a sports car on the road",
        "a vintage automobile",
        "a luxury vehicle in a showroom",
        "a car parked in a driveway"
    ][:5]  # Always use 5 unrelated prompts
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_results = {
        'target_prompts': target_prompts,
        'unrelated_prompts': unrelated_prompts,
        'models': {}
    }
    
    # Process each model
    print("\nProcessing models...")
    
    # Base model
    print("\n1. Base SD v1.4 model")
    base_pipeline = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False
    ).to(device)
    
    target_images = generate_evaluation_images(base_pipeline, target_prompts, output_dir, "base_sd14")
    unrelated_images = generate_evaluation_images(base_pipeline, unrelated_prompts, output_dir, "unrelated_base")
    all_results['models']['Base SD v1.4'] = {
        'target_images': target_images,
        'unrelated_images': unrelated_images
    }
    
    # Don't delete base_pipeline yet - we might need it for single-file models
    
    # GA model
    if ga_model:
        print(f"\n2. GA Model: {ga_model['name']}")
        ga_path = Path(__file__).parent.parent / "models" / ga_model['name']
        ga_pipeline = StableDiffusionPipeline.from_pretrained(
            str(ga_path),
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False
        ).to(device)
        
        target_images = generate_evaluation_images(ga_pipeline, target_prompts, output_dir, "ga_erased")
        unrelated_images = generate_evaluation_images(ga_pipeline, unrelated_prompts, output_dir, "unrelated_ga")
        all_results['models']['GA Erased'] = {
            'target_images': target_images,
            'unrelated_images': unrelated_images
        }
        
        del ga_pipeline
        torch.cuda.empty_cache()
    
    # ESD models
    model_idx = 3
    for method_type, model_info in sorted(esd_models_dict.items()):
        esd_model = model_info['model']
        print(f"\n{model_idx}. ESD Model ({method_type}): {esd_model['name']}")
        esd_path = Path(__file__).parent.parent / "models" / esd_model['name']
        
        esd_pipeline = load_pipeline_smart(esd_path, device)
        
        target_images = generate_evaluation_images(esd_pipeline, target_prompts, output_dir, f"esd_{method_type}_erased")
        unrelated_images = generate_evaluation_images(esd_pipeline, unrelated_prompts, output_dir, f"unrelated_esd_{method_type}")
        
        display_name = f"ESD {method_type.replace('_', ' ').title()}"
        all_results['models'][display_name] = {
            'target_images': target_images,
            'unrelated_images': unrelated_images
        }
        
        del esd_pipeline
        torch.cuda.empty_cache()
        model_idx += 1
    
    # Clean up base pipeline if we still have it
    del base_pipeline
    torch.cuda.empty_cache()
    
    # Create comparison grid
    print("\nCreating comparison grid...")
    grid_path = output_dir / f"{args.concept.replace(' ', '_')}_comparison_grid.png"
    create_comparison_grid_with_unrelated(all_results, grid_path)
    
    print(f"\nEvaluation complete!")
    print(f"Results saved to: {output_dir}")
    print(f"Comparison grid: {grid_path}")
    
    # Save metadata
    metadata = {
        "concept": args.concept,
        "ga_model": ga_model['name'] if ga_model else None,
        "esd_models": {k: v['model']['name'] for k, v in esd_models_dict.items()},
        "target_prompts": target_prompts,
        "unrelated_prompts": unrelated_prompts,
        "num_target_images": len(target_prompts),
        "num_unrelated_images": len(unrelated_prompts)
    }
    
    with open(output_dir / "evaluation_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    main()