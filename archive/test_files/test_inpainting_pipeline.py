#!/usr/bin/env python3
"""
Test script for HuggingFace inpainting pipeline.
This script tests the inpainting functionality using diffusers library.
"""

import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image, ImageDraw
import requests
import numpy as np
import os
import argparse
from pathlib import Path


def create_mask(image_size, mask_type="center_square"):
    """Create a mask for inpainting."""
    width, height = image_size
    mask = Image.new("RGB", (width, height), "black")
    draw = ImageDraw.Draw(mask)
    
    if mask_type == "center_square":
        # Create a white square in the center
        square_size = min(width, height) // 3
        x1 = (width - square_size) // 2
        y1 = (height - square_size) // 2
        x2 = x1 + square_size
        y2 = y1 + square_size
        draw.rectangle([x1, y1, x2, y2], fill="white")
    elif mask_type == "circle":
        # Create a white circle in the center
        radius = min(width, height) // 4
        center_x, center_y = width // 2, height // 2
        draw.ellipse([center_x - radius, center_y - radius, 
                     center_x + radius, center_y + radius], fill="white")
    elif mask_type == "car_area":
        # Create a mask for the car area (centered)
        car_width = int(width * 0.4)  # 40% of image width
        car_height = int(height * 0.3)  # 30% of image height
        x1 = (width - car_width) // 2
        y1 = (height - car_height) // 2
        x2 = x1 + car_width
        y2 = y1 + car_height
        draw.rectangle([x1, y1, x2, y2], fill="white")
    
    return mask


def load_test_image(image_path=None):
    """Load a test image for inpainting."""
    if image_path and os.path.exists(image_path):
        return Image.open(image_path).convert("RGB")
    
    # Use a default test image from the web
    try:
        url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
        response = requests.get(url)
        if response.status_code == 200:
            from io import BytesIO
            return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        print(f"Could not load image from URL: {e}")
    
    # Create a simple test image
    print("Creating a simple test image...")
    image = Image.new("RGB", (512, 512), "lightblue")
    draw = ImageDraw.Draw(image)
    draw.rectangle([100, 100, 412, 412], fill="green")
    draw.ellipse([200, 200, 312, 312], fill="red")
    return image


def test_inpainting_pipeline(
    model_id="stabilityai/stable-diffusion-2-inpainting",
    prompt="a beautiful landscape with mountains",
    image_path=None,
    mask_type="center_square",
    output_dir="./inpainting_results",
    num_inference_steps=20,
    guidance_scale=7.5,
    device=None
):
    """Test the inpainting pipeline."""
    
    # Setup device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {device}")
    print(f"Loading model: {model_id}")
    
    # Load the pipeline
    try:
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        pipe = pipe.to(device)
        
        # Enable memory efficient attention if available
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
        
        print("Pipeline loaded successfully!")
        
    except Exception as e:
        print(f"Error loading pipeline: {e}")
        return False
    
    # Load test image
    print("Loading test image...")
    original_image = load_test_image(image_path)
    
    # Resize image to appropriate size
    original_image = original_image.resize((512, 512))
    
    # Create mask
    print(f"Creating mask: {mask_type}")
    mask = create_mask(original_image.size, mask_type)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create highlighted versions with red overlay
    def add_red_overlay(image, mask):
        """Add red highlighting to show mask area."""
        highlighted = image.copy()
        # Convert mask to grayscale and create alpha channel
        mask_gray = mask.convert('L')
        # Create red overlay where mask is white (masked area)
        overlay = Image.new('RGBA', image.size, (255, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        
        # Get mask bounds
        mask_array = np.array(mask_gray)
        white_pixels = np.where(mask_array == 255)
        if len(white_pixels[0]) > 0:
            min_y, max_y = np.min(white_pixels[0]), np.max(white_pixels[0])
            min_x, max_x = np.min(white_pixels[1]), np.max(white_pixels[1])
            overlay_draw.rectangle([min_x, min_y, max_x, max_y], fill=(255, 0, 0, 80))
        
        # Composite with original image
        highlighted = highlighted.convert('RGBA')
        highlighted = Image.alpha_composite(highlighted, overlay)
        return highlighted.convert('RGB')
    
    # Create highlighted images
    original_highlighted = add_red_overlay(original_image, mask)
    
    # Save original image, highlighted version, and mask
    original_image.save(os.path.join(output_dir, "original_image.png"))
    original_highlighted.save(os.path.join(output_dir, "original_image_highlighted.png"))
    mask.save(os.path.join(output_dir, "mask.png"))
    
    # Run inpainting
    print(f"Running inpainting with prompt: '{prompt}'")
    print(f"Inference steps: {num_inference_steps}, Guidance scale: {guidance_scale}")
    
    try:
        with torch.no_grad():
            result = pipe(
                prompt=prompt,
                image=original_image,
                mask_image=mask,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=512,
                width=512
            )
        
        # Save result and highlighted version
        output_path = os.path.join(output_dir, "inpainted_result.png")
        result.images[0].save(output_path)
        
        # Create highlighted version of result
        result_highlighted = add_red_overlay(result.images[0], mask)
        result_highlighted.save(os.path.join(output_dir, "inpainted_result_highlighted.png"))
        
        print(f"Inpainting completed successfully!")
        print(f"Results saved to: {output_dir}")
        print(f"Files created:")
        print(f"  - original_image.png")
        print(f"  - original_image_highlighted.png")
        print(f"  - mask.png")
        print(f"  - inpainted_result.png")
        print(f"  - inpainted_result_highlighted.png")
        
        return True
        
    except Exception as e:
        print(f"Error during inpainting: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test HuggingFace inpainting pipeline")
    parser.add_argument("--model", default="stabilityai/stable-diffusion-2-inpainting",
                       help="Model ID to use for inpainting")
    parser.add_argument("--prompt", default="a beautiful landscape with mountains",
                       help="Prompt for inpainting")
    parser.add_argument("--image", help="Path to input image")
    parser.add_argument("--mask-type", choices=["center_square", "circle", "car_area"], 
                       default="center_square", help="Type of mask to create")
    parser.add_argument("--output-dir", default="./inpainting_results",
                       help="Output directory for results")
    parser.add_argument("--steps", type=int, default=20,
                       help="Number of inference steps")
    parser.add_argument("--guidance-scale", type=float, default=7.5,
                       help="Guidance scale for generation")
    parser.add_argument("--device", help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("HuggingFace Inpainting Pipeline Test")
    print("=" * 60)
    
    success = test_inpainting_pipeline(
        model_id=args.model,
        prompt=args.prompt,
        image_path=args.image,
        mask_type=args.mask_type,
        output_dir=args.output_dir,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        device=args.device
    )
    
    if success:
        print("\n✅ Inpainting test completed successfully!")
    else:
        print("\n❌ Inpainting test failed!")
        exit(1)


if __name__ == "__main__":
    main()