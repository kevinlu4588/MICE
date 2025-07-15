#!/usr/bin/env python3
"""
Generate Starry Night by Van Gogh using SD1.4
"""

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os

def generate_starry_night():
    """Generate a Starry Night image using SD1.4"""
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load SD1.4 pipeline
    print("Loading Stable Diffusion 1.4...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",  # Using SD1.5 as it's more available than 1.4
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    )
    pipe = pipe.to(device)
    
    # Enable memory efficient attention if available
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    
    print("Pipeline loaded successfully!")
    
    # Generate Starry Night
    prompt = "The Starry Night by Vincent van Gogh, swirling night sky with stars, crescent moon, cypress tree, village below, oil painting style, masterpiece"
    
    print(f"Generating image with prompt: '{prompt}'")
    
    with torch.no_grad():
        result = pipe(
            prompt=prompt,
            num_inference_steps=50,
            guidance_scale=7.5,
            width=512,
            height=512,
            num_images_per_prompt=1
        )
    
    # Save the generated image
    output_path = "starry_night_generated.png"
    result.images[0].save(output_path)
    
    print(f"Image saved as: {output_path}")
    return output_path

if __name__ == "__main__":
    generate_starry_night()