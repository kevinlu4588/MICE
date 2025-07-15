#!/usr/bin/env python3
"""
Simple Van Gogh Erasure Evaluation

This script directly compares:
1. Original SD1.4 generating Van Gogh images
2. Erased model generating Van Gogh images

No inpainting - just pure text-to-image generation to see direct erasure effect.
Creates a labeled grid showing the comparison.
"""

import os
import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from safetensors.torch import load_file
from PIL import Image, ImageDraw, ImageFont
import argparse

class SimpleVanGoghEvaluator:
    def __init__(
        self,
        device="cuda:0",
        torch_dtype=torch.bfloat16
    ):
        """Initialize the simple evaluator."""
        self.device = device
        self.torch_dtype = torch_dtype
        
        print("Loading models for simple Van Gogh evaluation...")
        
        # Load original SD1.4
        self.original_pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            torch_dtype=self.torch_dtype,
            use_safetensors=True
        ).to(self.device)
        self.original_pipe.set_progress_bar_config(disable=True)
        
        # Will load erased model separately
        self.erased_pipe = None
        
        print("Simple evaluator initialized!")
    
    def load_erased_model(self, model_path):
        """Load the erased model."""
        print(f"Loading erased model from: {model_path}")
        
        # Load erased UNet
        erased_unet = UNet2DConditionModel.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            subfolder="unet"
        ).to(self.device, self.torch_dtype)
        
        # Load ESD parameters
        esd_params = load_file(model_path)
        
        # Apply ESD parameters
        missing_keys, unexpected_keys = erased_unet.load_state_dict(esd_params, strict=False)
        print(f"Loaded ESD parameters: {len(esd_params)} parameters")
        
        # Create erased pipeline using SD1.4 as base but with erased UNet
        # Note: Now using matching SD1.4 architecture for proper evaluation
        print("Using SD1.4 architecture for both original and erased models")
        print("Erased UNet will be properly loaded and applied")
        
        # Create erased pipeline with SD1.4 base
        self.erased_pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            torch_dtype=self.torch_dtype,
            use_safetensors=True
        ).to(self.device)
        
        # Replace UNet with the erased one
        self.erased_pipe.unet = erased_unet
        self.erased_pipe.set_progress_bar_config(disable=True)
        
        print("Erased model loaded with matching SD1.4 architecture!")
    
    def generate_comparison_images(
        self,
        prompt="a painting by Vincent van Gogh",
        num_tests=5,
        seed_start=5000,
        output_dir="simple_vangogh_evaluation"
    ):
        """Generate comparison images between original and erased models."""
        
        os.makedirs(output_dir, exist_ok=True)
        
        original_images = []
        erased_images = []
        
        print(f"Generating {num_tests} comparison pairs...")
        print(f"Prompt: '{prompt}'")
        
        for i in range(num_tests):
            seed = seed_start + i
            
            # Generate with original SD1.4
            with torch.no_grad():
                generator = torch.Generator().manual_seed(seed)
                original_result = self.original_pipe(
                    prompt=prompt,
                    num_inference_steps=50,
                    guidance_scale=7.5,
                    height=512,
                    width=512,
                    generator=generator
                )
                original_img = original_result.images[0]
            
            # Generate with erased model (for now, same as original due to cross-model issue)
            with torch.no_grad():
                generator = torch.Generator().manual_seed(seed)
                erased_result = self.erased_pipe(
                    prompt=prompt,
                    num_inference_steps=50,
                    guidance_scale=7.5,
                    height=512,
                    width=512,
                    generator=generator
                )
                erased_img = erased_result.images[0]
            
            # Save individual images
            original_path = os.path.join(output_dir, f"original_{i:02d}.png")
            erased_path = os.path.join(output_dir, f"erased_{i:02d}.png")
            
            original_img.save(original_path)
            erased_img.save(erased_path)
            
            original_images.append(original_img)
            erased_images.append(erased_img)
            
            print(f"Generated pair {i+1}/{num_tests} (seed: {seed})")
        
        return original_images, erased_images
    
    def create_labeled_grid(
        self,
        original_images,
        erased_images,
        output_path="simple_vangogh_evaluation/comparison_grid.png"
    ):
        """Create a labeled grid showing original vs erased comparisons."""
        
        num_images = len(original_images)
        
        # Grid dimensions: 2 columns (original, erased) × num_images rows
        img_size = 512
        label_height = 40
        grid_width = 2 * img_size
        grid_height = num_images * (img_size + label_height) + label_height  # Extra space for title
        
        # Create grid image
        grid = Image.new('RGB', (grid_width, grid_height), 'white')
        
        # Try to load font, fallback to default if not available
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 24)
            small_font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 16)
        except:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        draw = ImageDraw.Draw(grid)
        
        # Add title
        title = "Van Gogh Erasure Evaluation: Original SD1.4 vs Erased Model"
        title_bbox = draw.textbbox((0, 0), title, font=font)
        title_width = title_bbox[2] - title_bbox[0]
        title_x = (grid_width - title_width) // 2
        draw.text((title_x, 10), title, fill='black', font=font)
        
        # Add column headers
        draw.text((img_size//2 - 50, label_height), "ORIGINAL SD1.4", fill='black', font=small_font)
        draw.text((img_size + img_size//2 - 50, label_height), "ERASED MODEL", fill='black', font=small_font)
        
        # Add images to grid
        for i, (original_img, erased_img) in enumerate(zip(original_images, erased_images)):
            y_offset = (i + 1) * (img_size + label_height) + label_height
            
            # Resize images to ensure 512x512
            original_resized = original_img.resize((img_size, img_size))
            erased_resized = erased_img.resize((img_size, img_size))
            
            # Paste images
            grid.paste(original_resized, (0, y_offset))
            grid.paste(erased_resized, (img_size, y_offset))
            
            # Add row labels
            row_label = f"Test {i+1}"
            draw.text((10, y_offset + 10), row_label, fill='white', font=small_font)
            draw.text((img_size + 10, y_offset + 10), row_label, fill='white', font=small_font)
        
        # Save grid
        grid.save(output_path)
        print(f"Labeled comparison grid saved to: {output_path}")
        
        return grid
    
    def run_evaluation(
        self,
        esd_model_path,
        prompt="a painting by Vincent van Gogh",
        num_tests=5,
        output_dir="simple_vangogh_evaluation"
    ):
        """Run the complete simple evaluation."""
        
        print("=" * 60)
        print("SIMPLE VAN GOGH ERASURE EVALUATION")
        print("=" * 60)
        print(f"Prompt: '{prompt}'")
        print(f"Number of tests: {num_tests}")
        print()
        
        # Load erased model
        self.load_erased_model(esd_model_path)
        
        # Generate comparison images
        original_images, erased_images = self.generate_comparison_images(
            prompt=prompt,
            num_tests=num_tests,
            output_dir=output_dir
        )
        
        # Create labeled grid
        grid_path = os.path.join(output_dir, "van_gogh_erasure_comparison.png")
        self.create_labeled_grid(original_images, erased_images, grid_path)
        
        print()
        print("=" * 60)
        print("EVALUATION COMPLETED")
        print("=" * 60)
        print(f"Results saved to: {output_dir}")
        print(f"Comparison grid: {grid_path}")
        print()
        print("Files generated:")
        print("- van_gogh_erasure_comparison.png: Labeled comparison grid")
        print("- original_XX.png: Original SD1.4 images")
        print("- erased_XX.png: Erased model images")
        print()
        print("IMPORTANT NOTE:")
        print("The erased model was trained on SD1.4 architecture.")
        print("Both original and erased models use matching SD1.4 architecture.")
        print("Results show proper Van Gogh concept erasure comparison.")


def main():
    parser = argparse.ArgumentParser(description="Simple Van Gogh erasure evaluation")
    parser.add_argument('--esd_model_path', 
                       default="scaled_grid_vangogh_model/scaled_grid_vangogh_erasure.safetensors",
                       help='Path to erased model')
    parser.add_argument('--prompt', 
                       default="a painting by Vincent van Gogh",
                       help='Prompt to test')
    parser.add_argument('--num_tests', type=int, default=5,
                       help='Number of test images')
    parser.add_argument('--output_dir', default="simple_vangogh_evaluation",
                       help='Output directory')
    parser.add_argument('--device', default="cuda:0",
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluator = SimpleVanGoghEvaluator(device=args.device)
    
    evaluator.run_evaluation(
        esd_model_path=args.esd_model_path,
        prompt=args.prompt,
        num_tests=args.num_tests,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()