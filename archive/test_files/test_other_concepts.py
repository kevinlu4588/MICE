#!/usr/bin/env python3
"""
Test the impact of Van Gogh inpainting erasure on other concepts.

This script evaluates whether the Van Gogh erasure model impacts:
1. Other artists (Picasso, Monet, etc.)
2. Non-artistic concepts (animals, landscapes, objects)
3. General image quality and coherence
"""

import os
import torch
import random
import numpy as np
from tqdm.auto import tqdm
from safetensors.torch import load_file
from diffusers import StableDiffusionInpaintPipeline, StableDiffusionPipeline, UNet2DConditionModel
from PIL import Image, ImageDraw
import argparse
import json

class ConceptImpactEvaluator:
    def __init__(
        self, 
        model_id="stabilityai/stable-diffusion-2-inpainting",
        device="cuda:0",
        torch_dtype=torch.bfloat16
    ):
        """Initialize the concept impact evaluator."""
        self.model_id = model_id
        self.device = device
        self.torch_dtype = torch_dtype
        
        print(f"Loading models for concept impact evaluation...")
        
        # Load original models
        self.load_original_models()
        
        # ESD models will be loaded separately
        self.esd_unet = None
        self.esd_pipe = None
        
        print("Concept impact evaluator initialized!")
    
    def load_original_models(self):
        """Load original (non-erased) models."""
        # Original UNet
        self.original_unet = UNet2DConditionModel.from_pretrained(
            self.model_id, subfolder="unet"
        ).to(self.device, self.torch_dtype)
        
        # Original inpainting pipeline
        self.original_pipe = StableDiffusionInpaintPipeline.from_pretrained(
            self.model_id,
            unet=self.original_unet,
            torch_dtype=self.torch_dtype,
            use_safetensors=True
        ).to(self.device)
        self.original_pipe.set_progress_bar_config(disable=True)
        
        # Text-to-image pipeline for generating base images
        self.text2img_pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            torch_dtype=self.torch_dtype,
            use_safetensors=True
        ).to(self.device)
        self.text2img_pipe.set_progress_bar_config(disable=True)
    
    def load_esd_model(self, model_path):
        """Load the trained ESD model."""
        print(f"Loading ESD model from: {model_path}")
        
        # Load ESD UNet
        self.esd_unet = UNet2DConditionModel.from_pretrained(
            self.model_id, subfolder="unet"
        ).to(self.device, self.torch_dtype)
        
        # Load trained parameters
        esd_params = load_file(model_path)
        
        # Apply parameters to ESD UNet
        missing_keys, unexpected_keys = self.esd_unet.load_state_dict(esd_params, strict=False)
        
        if missing_keys:
            print(f"Warning: Missing keys in ESD model: {len(missing_keys)}")
        
        # Create ESD pipeline
        self.esd_pipe = StableDiffusionInpaintPipeline.from_pretrained(
            self.model_id,
            unet=self.esd_unet,
            torch_dtype=self.torch_dtype,
            use_safetensors=True
        ).to(self.device)
        self.esd_pipe.set_progress_bar_config(disable=True)
        
        print("ESD model loaded successfully!")
    
    def create_mask(self, image_size, mask_type="center_square"):
        """Create masks for evaluation."""
        width, height = image_size
        mask = Image.new("RGB", (width, height), "black")
        draw = ImageDraw.Draw(mask)
        
        if mask_type == "center_square":
            square_size = min(width, height) // 3
            x1 = (width - square_size) // 2
            y1 = (height - square_size) // 2
            x2 = x1 + square_size
            y2 = y1 + square_size
            draw.rectangle([x1, y1, x2, y2], fill="white")
        
        return mask
    
    def get_test_prompts(self):
        """Get test prompts for different concept categories."""
        return {
            "other_artists": [
                "A painting by Pablo Picasso",
                "A landscape by Claude Monet",
                "A portrait by Leonardo da Vinci",
                "A sculpture by Auguste Rodin",
                "A painting by Salvador Dalí",
                "A drawing by Rembrandt",
                "A canvas by Jackson Pollock",
                "A work by Georgia O'Keeffe"
            ],
            "art_styles": [
                "An impressionist painting",
                "A cubist artwork",
                "A baroque painting",
                "A surrealist composition",
                "An abstract expressionist piece",
                "A renaissance painting",
                "A romantic landscape",
                "A modernist artwork"
            ],
            "general_objects": [
                "A red sports car",
                "A golden retriever",
                "A mountain landscape",
                "A city skyline",
                "A flower garden",
                "A sailing boat",
                "A wooden chair",
                "A steaming coffee cup"
            ],
            "nature_scenes": [
                "A forest in autumn",
                "A sunset over the ocean",
                "A snow-covered mountain",
                "A tropical beach",
                "A field of sunflowers",
                "A rushing waterfall",
                "A desert landscape",
                "A starry night sky"
            ]
        }
    
    def evaluate_concept_impact(
        self,
        concept_category,
        prompts,
        num_tests_per_prompt=2,
        output_dir="concept_impact_evaluation",
        seed_start=3000
    ):
        """Evaluate impact on a specific concept category."""
        
        if self.esd_pipe is None:
            raise ValueError("ESD model not loaded. Call load_esd_model() first.")
        
        category_dir = os.path.join(output_dir, concept_category)
        os.makedirs(category_dir, exist_ok=True)
        
        results = []
        current_seed = seed_start
        
        print(f"Evaluating impact on {concept_category}...")
        
        for prompt_idx, prompt in enumerate(tqdm(prompts, desc=f"Testing {concept_category}")):
            for test_idx in range(num_tests_per_prompt):
                seed = current_seed
                current_seed += 1
                
                # Generate base image
                with torch.no_grad():
                    generator = torch.Generator().manual_seed(seed)
                    base_result = self.text2img_pipe(
                        prompt=prompt,
                        num_inference_steps=50,
                        guidance_scale=7.5,
                        height=512,
                        width=512,
                        generator=generator
                    )
                    base_image = base_result.images[0]
                
                # Create mask
                mask = self.create_mask(base_image.size, "center_square")
                
                # Generate with original model
                with torch.no_grad():
                    generator = torch.Generator().manual_seed(seed)
                    original_result = self.original_pipe(
                        prompt=prompt,
                        image=base_image,
                        mask_image=mask,
                        num_inference_steps=50,
                        guidance_scale=7.5,
                        height=512,
                        width=512,
                        generator=generator
                    )
                    original_inpainted = original_result.images[0]
                
                # Generate with ESD model
                with torch.no_grad():
                    generator = torch.Generator().manual_seed(seed)
                    esd_result = self.esd_pipe(
                        prompt=prompt,
                        image=base_image,
                        mask_image=mask,
                        num_inference_steps=50,
                        guidance_scale=7.5,
                        height=512,
                        width=512,
                        generator=generator
                    )
                    esd_inpainted = esd_result.images[0]
                
                # Save results
                test_id = f"{concept_category}_p{prompt_idx:02d}_t{test_idx:02d}"
                
                base_path = os.path.join(category_dir, f"{test_id}_base.png")
                mask_path = os.path.join(category_dir, f"{test_id}_mask.png")
                original_path = os.path.join(category_dir, f"{test_id}_original.png")
                esd_path = os.path.join(category_dir, f"{test_id}_esd.png")
                
                base_image.save(base_path)
                mask.save(mask_path)
                original_inpainted.save(original_path)
                esd_inpainted.save(esd_path)
                
                results.append({
                    'test_id': test_id,
                    'prompt': prompt,
                    'seed': seed,
                    'category': concept_category,
                    'base_image': base_path,
                    'mask': mask_path,
                    'original_inpainted': original_path,
                    'esd_inpainted': esd_path
                })
        
        # Save category results
        results_path = os.path.join(category_dir, f"{concept_category}_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Category {concept_category}: {len(results)} tests completed")
        return results
    
    def create_concept_comparison_grid(self, all_results, output_dir="concept_impact_evaluation"):
        """Create comparison grids for each concept category."""
        
        # Group results by category
        by_category = {}
        for result in all_results:
            category = result['category']
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(result)
        
        for category, results in by_category.items():
            print(f"Creating comparison grid for {category}...")
            
            # Take up to 8 examples for the grid
            grid_results = results[:8]
            grid_size = min(4, int(np.ceil(np.sqrt(len(grid_results)))))
            
            # Create grid image (3 images per test: base, original, esd)
            grid_width = grid_size * 512 * 3
            grid_height = grid_size * 512
            
            grid_image = Image.new('RGB', (grid_width, grid_height), 'white')
            
            for i, result in enumerate(grid_results):
                if i >= grid_size * grid_size:
                    break
                
                row = i // grid_size
                col = i % grid_size
                
                # Load images
                base_img = Image.open(result['base_image']).resize((512, 512))
                original_img = Image.open(result['original_inpainted']).resize((512, 512))
                esd_img = Image.open(result['esd_inpainted']).resize((512, 512))
                
                # Paste in grid
                y_offset = row * 512
                
                grid_image.paste(base_img, (col * 512 * 3, y_offset))
                grid_image.paste(original_img, (col * 512 * 3 + 512, y_offset))
                grid_image.paste(esd_img, (col * 512 * 3 + 1024, y_offset))
            
            # Save grid
            grid_path = os.path.join(output_dir, f"comparison_grid_{category}.png")
            grid_image.save(grid_path)
            print(f"Saved: {grid_path}")
    
    def run_full_evaluation(self, esd_model_path, output_dir="concept_impact_evaluation"):
        """Run full concept impact evaluation."""
        
        # Load ESD model
        self.load_esd_model(esd_model_path)
        
        # Get test prompts
        test_prompts = self.get_test_prompts()
        
        # Run evaluation for each category
        all_results = []
        
        for category, prompts in test_prompts.items():
            category_results = self.evaluate_concept_impact(
                category,
                prompts,
                num_tests_per_prompt=1,  # 1 test per prompt for efficiency
                output_dir=output_dir,
                seed_start=3000 + len(all_results) * 100
            )
            all_results.extend(category_results)
        
        # Create comparison grids
        self.create_concept_comparison_grid(all_results, output_dir)
        
        # Save overall results
        overall_results_path = os.path.join(output_dir, "overall_concept_impact.json")
        with open(overall_results_path, 'w') as f:
            json.dump({
                'total_tests': len(all_results),
                'categories': list(test_prompts.keys()),
                'results': all_results
            }, f, indent=2)
        
        # Generate summary
        summary = self.generate_summary(all_results)
        summary_path = os.path.join(output_dir, "impact_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nFull evaluation completed!")
        print(f"Total tests: {len(all_results)}")
        print(f"Categories tested: {list(test_prompts.keys())}")
        print(f"Results saved to: {output_dir}")
        
        return all_results
    
    def generate_summary(self, all_results):
        """Generate a summary of concept impact results."""
        by_category = {}
        for result in all_results:
            category = result['category']
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(result)
        
        summary = {
            'total_tests': len(all_results),
            'categories': {
                category: {
                    'count': len(results),
                    'prompts': list(set(r['prompt'] for r in results))
                }
                for category, results in by_category.items()
            }
        }
        
        return summary


def main():
    parser = argparse.ArgumentParser(description="Test concept impact of inpainting erasure")
    parser.add_argument('--esd_model_path', required=True,
                       help='Path to trained ESD model (.safetensors)')
    parser.add_argument('--output_dir', default="concept_impact_evaluation",
                       help='Directory to save evaluation results')
    parser.add_argument('--device', default="cuda:0",
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = ConceptImpactEvaluator(device=args.device)
    
    # Run full evaluation
    results = evaluator.run_full_evaluation(
        esd_model_path=args.esd_model_path,
        output_dir=args.output_dir
    )
    
    print("\nConcept impact evaluation completed!")
    print(f"Check results in: {args.output_dir}")


if __name__ == "__main__":
    main()