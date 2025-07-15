#!/usr/bin/env python3
"""
Evaluation Script for Inpainting Erasure Model

This script evaluates the effectiveness of the inpainting erasure method by:
1. Loading the trained ESD model
2. Generating test images with Van Gogh prompts
3. Comparing original vs erased model outputs
4. Providing visual and quantitative evaluation
"""

import os
import torch
import pandas as pd
import random
import numpy as np
from tqdm.auto import tqdm
from safetensors.torch import load_file
from diffusers import StableDiffusionInpaintPipeline, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.image_processor import VaeImageProcessor
from PIL import Image, ImageDraw
import argparse
import json
from pathlib import Path

class InpaintingErasureEvaluator:
    def __init__(
        self, 
        model_id="stabilityai/stable-diffusion-2-inpainting",
        device="cuda:0",
        torch_dtype=torch.bfloat16
    ):
        """Initialize the evaluator."""
        self.model_id = model_id
        self.device = device
        self.torch_dtype = torch_dtype
        
        print(f"Loading models from: {model_id}")
        
        # Load original models
        self.load_original_models()
        
        # Will load ESD model separately
        self.esd_unet = None
        self.esd_pipe = None
        
        print("Evaluator initialized successfully!")
    
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
        if unexpected_keys:
            print(f"Warning: Unexpected keys in ESD model: {len(unexpected_keys)}")
        
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
        
        elif mask_type == "circle":
            radius = min(width, height) // 4
            center_x, center_y = width // 2, height // 2
            draw.ellipse([center_x - radius, center_y - radius, 
                         center_x + radius, center_y + radius], fill="white")
        
        elif mask_type == "car_area":
            car_width = int(width * 0.4)
            car_height = int(height * 0.3)
            x1 = (width - car_width) // 2
            y1 = (height - car_height) // 2
            x2 = x1 + car_width
            y2 = y1 + car_height
            draw.rectangle([x1, y1, x2, y2], fill="white")
        
        return mask
    
    def load_prompts(self, prompt_file):
        """Load Van Gogh prompts from CSV file."""
        df = pd.read_csv(prompt_file)
        return df['prompt'].tolist()
    
    def generate_comparison_images(
        self,
        prompts,
        num_tests=20,
        mask_types=["center_square", "circle", "car_area"],
        output_dir="evaluation_results",
        seed_start=1000
    ):
        """Generate comparison images between original and ESD models."""
        
        if self.esd_pipe is None:
            raise ValueError("ESD model not loaded. Call load_esd_model() first.")
        
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        
        print(f"Generating {num_tests} comparison sets...")
        
        for test_idx in tqdm(range(num_tests), desc="Generating comparisons"):
            # Select random prompt
            prompt = random.choice(prompts)
            seed = seed_start + test_idx
            
            # Generate base image (without inpainting)
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
            
            # Save base image
            base_path = os.path.join(output_dir, f"test_{test_idx:03d}_base.png")
            base_image.save(base_path)
            
            test_result = {
                'test_id': test_idx,
                'prompt': prompt,
                'seed': seed,
                'base_image': base_path,
                'masks': {}
            }
            
            # Test each mask type
            for mask_type in mask_types:
                mask = self.create_mask(base_image.size, mask_type)
                
                # Save mask
                mask_path = os.path.join(output_dir, f"test_{test_idx:03d}_mask_{mask_type}.png")
                mask.save(mask_path)
                
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
                original_path = os.path.join(output_dir, f"test_{test_idx:03d}_original_{mask_type}.png")
                esd_path = os.path.join(output_dir, f"test_{test_idx:03d}_esd_{mask_type}.png")
                
                original_inpainted.save(original_path)
                esd_inpainted.save(esd_path)
                
                test_result['masks'][mask_type] = {
                    'mask_path': mask_path,
                    'original_inpainted': original_path,
                    'esd_inpainted': esd_path
                }
            
            results.append(test_result)
        
        # Save results metadata
        results_path = os.path.join(output_dir, "evaluation_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Generated {len(results)} comparison sets")
        print(f"Results saved to: {output_dir}")
        print(f"Metadata saved to: {results_path}")
        
        return results
    
    def create_comparison_grid(self, results, output_dir="evaluation_results", grid_size=4):
        """Create comparison grids for visual evaluation."""
        
        print(f"Creating comparison grids...")
        
        for mask_type in ["center_square", "circle", "car_area"]:
            # Create grid image
            grid_width = grid_size * 512 * 3  # 3 images per test (base, original, esd)
            grid_height = grid_size * 512
            
            grid_image = Image.new('RGB', (grid_width, grid_height), 'white')
            
            for i in range(min(grid_size * grid_size, len(results))):
                row = i // grid_size
                col = i % grid_size
                
                result = results[i]
                
                # Load images
                base_img = Image.open(result['base_image']).resize((512, 512))
                original_img = Image.open(result['masks'][mask_type]['original_inpainted']).resize((512, 512))
                esd_img = Image.open(result['masks'][mask_type]['esd_inpainted']).resize((512, 512))
                
                # Paste in grid
                y_offset = row * 512
                
                grid_image.paste(base_img, (col * 512 * 3, y_offset))
                grid_image.paste(original_img, (col * 512 * 3 + 512, y_offset))
                grid_image.paste(esd_img, (col * 512 * 3 + 1024, y_offset))
            
            # Save grid
            grid_path = os.path.join(output_dir, f"comparison_grid_{mask_type}.png")
            grid_image.save(grid_path)
            print(f"Saved comparison grid: {grid_path}")
    
    def analyze_results(self, results, output_dir="evaluation_results"):
        """Analyze and summarize evaluation results."""
        
        print("Analyzing results...")
        
        analysis = {
            'total_tests': len(results),
            'mask_types': list(results[0]['masks'].keys()) if results else [],
            'prompts_used': [r['prompt'] for r in results],
            'unique_prompts': len(set(r['prompt'] for r in results))
        }
        
        # Save analysis
        analysis_path = os.path.join(output_dir, "analysis.json")
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"Analysis saved to: {analysis_path}")
        print(f"Total tests: {analysis['total_tests']}")
        print(f"Mask types: {analysis['mask_types']}")
        print(f"Unique prompts: {analysis['unique_prompts']}")
        
        return analysis


def main():
    parser = argparse.ArgumentParser(description="Evaluate Inpainting Erasure Model")
    parser.add_argument('--model_id', default="stabilityai/stable-diffusion-2-inpainting",
                       help='Base model ID')
    parser.add_argument('--esd_model_path', required=True,
                       help='Path to trained ESD model (.safetensors)')
    parser.add_argument('--prompts_file', default="data/vangogh_prompts.csv",
                       help='CSV file with Van Gogh prompts')
    parser.add_argument('--num_tests', type=int, default=20,
                       help='Number of test cases to generate')
    parser.add_argument('--output_dir', default="evaluation_results",
                       help='Directory to save evaluation results')
    parser.add_argument('--device', default="cuda:0",
                       help='Device to use')
    parser.add_argument('--seed_start', type=int, default=1000,
                       help='Starting seed for reproducible tests')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = InpaintingErasureEvaluator(
        model_id=args.model_id,
        device=args.device
    )
    
    # Load ESD model
    evaluator.load_esd_model(args.esd_model_path)
    
    # Load prompts
    prompts = evaluator.load_prompts(args.prompts_file)
    print(f"Loaded {len(prompts)} Van Gogh prompts for evaluation")
    
    # Generate comparison images
    results = evaluator.generate_comparison_images(
        prompts=prompts,
        num_tests=args.num_tests,
        output_dir=args.output_dir,
        seed_start=args.seed_start
    )
    
    # Create comparison grids
    evaluator.create_comparison_grid(results, args.output_dir)
    
    # Analyze results
    analysis = evaluator.analyze_results(results, args.output_dir)
    
    print("\nEvaluation completed successfully!")
    print(f"Check results in: {args.output_dir}")
    print("\nFiles generated:")
    print("- evaluation_results.json: Detailed results metadata")
    print("- comparison_grid_*.png: Visual comparison grids")
    print("- analysis.json: Summary analysis")
    print("- test_*_*.png: Individual test images")


if __name__ == "__main__":
    main()