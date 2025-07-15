#!/bin/bash

# Script to generate training data for a single concept

# Default parameters
CONCEPT=""
NUM_IMAGES=100
OUTPUT_DIR="training_images"
DEVICE="cuda"
MODEL_ID="CompVis/stable-diffusion-v1-4"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --concept)
            CONCEPT="$2"
            shift 2
            ;;
        --num_images)
            NUM_IMAGES="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --model_id)
            MODEL_ID="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 --concept CONCEPT [options]"
            echo ""
            echo "Generate training images for a specific concept using Stable Diffusion"
            echo ""
            echo "Required:"
            echo "  --concept CONCEPT        The concept to generate images for (e.g., 'van gogh', 'picasso')"
            echo ""
            echo "Options:"
            echo "  --num_images NUM        Number of images to generate (default: 50)"
            echo "  --output_dir DIR        Output directory for images (default: training_images)"
            echo "  --device DEVICE         Device to use: cuda or cpu (default: cuda)"
            echo "  --model_id MODEL        Model to use (default: CompVis/stable-diffusion-v1-4)"
            echo "  -h, --help             Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --concept 'van gogh' --num_images 100"
            echo "  $0 --concept 'picasso' --output_dir my_data"
            echo "  $0 --concept 'english springer spaniel' --device cpu"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if concept is provided
if [ -z "$CONCEPT" ]; then
    echo "Error: --concept is required"
    echo "Use --help for usage information"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Convert concept to directory name (replace spaces with underscores)
CONCEPT_DIR="${OUTPUT_DIR}/${CONCEPT// /_}"

# Check if concept directory already exists and has images
if [ -d "$CONCEPT_DIR" ]; then
    EXISTING_IMAGES=$(find "$CONCEPT_DIR" -name "*.png" 2>/dev/null | wc -l)
    if [ $EXISTING_IMAGES -gt 0 ]; then
        echo "Warning: Found $EXISTING_IMAGES existing images in $CONCEPT_DIR"
        read -p "Do you want to continue and add more images? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Aborted."
            exit 0
        fi
    fi
fi

# Print configuration
echo "======================================"
echo "Training Data Generation"
echo "======================================"
echo "Concept: $CONCEPT"
echo "Number of images: $NUM_IMAGES"
echo "Output directory: $OUTPUT_DIR"
echo "Concept directory: $CONCEPT_DIR"
echo "Device: $DEVICE"
echo "Model: $MODEL_ID"
echo "======================================"

# Create a Python script inline to generate the data
PYTHON_SCRIPT=$(cat << 'EOF'
import os
import sys
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import random

def generate_training_images(concept, num_images, output_dir, device_name, model_id):
    """Generate training images for a specific concept"""
    
    # Create output directory
    concept_dir = Path(output_dir) / concept.replace(" ", "_")
    os.makedirs(concept_dir, exist_ok=True)
    
    # Count existing images
    existing_images = list(concept_dir.glob("*.png"))
    start_index = len(existing_images)
    
    # Define prompts based on concept type
    if any(artist in concept.lower() for artist in ["picasso", "warhol", "van gogh", "monet", "dali"]):
        # Artist concepts
        prompts = [
            f"a painting by {concept}",
            f"artwork in the style of {concept}",
            f"a masterpiece by {concept}",
            f"art by {concept}",
            f"{concept} style painting",
            f"a portrait by {concept}",
            f"a landscape by {concept}",
            f"abstract art by {concept}",
            f"a famous painting by {concept}",
            f"artwork created by {concept}",
            f"a beautiful painting in the style of {concept}",
            f"impressionist painting by {concept}",
            f"oil painting by {concept}",
            f"watercolor by {concept}",
            f"a detailed artwork by {concept}"
        ]
    elif any(animal in concept.lower() for animal in ["dog", "cat", "spaniel", "retriever", "terrier"]):
        # Animal/breed concepts
        article = "an" if concept[0].lower() in 'aeiou' else "a"
        prompts = [
            f"a photo of {article} {concept}",
            f"a beautiful {concept}",
            f"a cute {concept}",
            f"{article} {concept} sitting",
            f"{article} {concept} running",
            f"a portrait of {article} {concept}",
            f"{article} {concept} playing",
            f"a professional photo of {article} {concept}",
            f"{article} {concept} outdoors",
            f"a close-up of {article} {concept}",
            f"{article} {concept} in nature",
            f"a happy {concept}",
            f"{article} {concept} on grass",
            f"a studio photo of {article} {concept}",
            f"{article} {concept} with blue sky background"
        ]
    else:
        # Generic objects
        article = "an" if concept[0].lower() in 'aeiou' else "a"
        prompts = [
            f"a photo of {article} {concept}",
            f"{article} {concept}",
            f"a detailed photo of {article} {concept}",
            f"{article} {concept} on white background",
            f"a professional photo of {article} {concept}",
            f"a close-up of {article} {concept}",
            f"{article} {concept} in use",
            f"a high-quality image of {article} {concept}",
            f"{article} {concept} product photo",
            f"a realistic {concept}",
            f"multiple {concept}s",
            f"{article} {concept} from different angle",
            f"a studio photo of {article} {concept}",
            f"{article} {concept} with neutral background",
            f"a clear photo of {article} {concept}"
        ]
    
    print(f"\nGenerating {num_images} images for concept: {concept}")
    if start_index > 0:
        print(f"Starting from index {start_index} (found {start_index} existing images)")
    
    # Load the model
    print("Loading Stable Diffusion model...")
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    ).to(device)
    
    print(f"Model loaded on {device}")
    
    # Generate images
    image_count = 0
    base_seed = random.randint(1000, 10000)
    
    with tqdm(total=num_images, desc=f"Generating {concept} images") as pbar:
        attempt = 0
        while image_count < num_images and attempt < num_images * 2:
            # Cycle through prompts with variations
            prompt_idx = image_count % len(prompts)
            prompt = prompts[prompt_idx]
            
            # Add some variation to prompts
            if random.random() > 0.5:
                variations = [
                    f"{prompt}, high quality",
                    f"{prompt}, detailed",
                    f"{prompt}, professional",
                    f"{prompt}, 4k",
                    f"{prompt}, beautiful"
                ]
                prompt = random.choice(variations)
            
            # Generate with different seeds for variety
            current_seed = base_seed + attempt
            generator = torch.Generator(device=device).manual_seed(current_seed)
            
            try:
                image = pipeline(
                    prompt,
                    generator=generator,
                    num_inference_steps=50,
                    guidance_scale=7.5
                ).images[0]
                
                # Save image with sequential numbering
                filename = f"{concept.replace(' ', '_')}_{start_index + image_count:04d}.png"
                image.save(concept_dir / filename)
                
                image_count += 1
                pbar.update(1)
                
            except Exception as e:
                print(f"\nError generating image with prompt '{prompt}': {e}")
                # Continue with next attempt
            
            attempt += 1
    
    print(f"\nSaved {image_count} new images to {concept_dir}")
    total_images = len(list(concept_dir.glob("*.png")))
    print(f"Total images in directory: {total_images}")
    
    return str(concept_dir)

# Main execution
if __name__ == "__main__":
    concept = sys.argv[1]
    num_images = int(sys.argv[2])
    output_dir = sys.argv[3]
    device = sys.argv[4]
    model_id = sys.argv[5]
    
    try:
        concept_dir = generate_training_images(concept, num_images, output_dir, device, model_id)
        print(f"\nSuccess! Training data generated in: {concept_dir}")
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)
EOF
)

# Run the Python script
echo "$PYTHON_SCRIPT" | python - "$CONCEPT" "$NUM_IMAGES" "$OUTPUT_DIR" "$DEVICE" "$MODEL_ID"

# Check if generation was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "======================================"
    echo "Training data generation complete!"
    echo "======================================"
    
    # Show statistics
    CONCEPT_DIR_FINAL="${OUTPUT_DIR}/${CONCEPT// /_}"
    if [ -d "$CONCEPT_DIR_FINAL" ]; then
        TOTAL_IMAGES=$(find "$CONCEPT_DIR_FINAL" -name "*.png" 2>/dev/null | wc -l)
        echo "Total images for '$CONCEPT': $TOTAL_IMAGES"
        echo "Images location: $CONCEPT_DIR_FINAL"
        
        # Show sample of generated files
        echo ""
        echo "Sample of generated files:"
        ls -la "$CONCEPT_DIR_FINAL" | head -n 6
    fi
else
    echo ""
    echo "Error: Training data generation failed!"
    exit 1
fi