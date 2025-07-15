#!/bin/bash

# Dynamic evaluation script - takes paths and concepts as arguments
# Usage: ./evaluate_dynamic.sh <model_path> <erased_concept> [unrelated_concept] [num_images]

# Check if minimum arguments are provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <model_path> <erased_concept> [unrelated_concept] [num_images]"
    echo ""
    echo "Arguments:"
    echo "  model_path      - Path to the model directory"
    echo "  erased_concept  - The concept that was erased (e.g., 'van gogh')"
    echo "  unrelated_concept - Optional: Concept to test (default: 'a picture of a car')"
    echo "  num_images      - Optional: Number of images to generate (default: 10)"
    echo ""
    echo "Example:"
    echo "  $0 ../models/3_inpainting_erasure_van_gogh 'van gogh' 'a picture of a car' 10"
    exit 1
fi

# Parse arguments
MODEL_PATH="$1"
ERASED_CONCEPT="$2"
UNRELATED_CONCEPT="${3:-a picture of a car}"  # Default to "a picture of a car"
NUM_IMAGES="${4:-10}"  # Default to 10 images

# Check if model directory exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model directory not found: $MODEL_PATH"
    exit 1
fi

# Extract model name from path for output directory
MODEL_NAME=$(basename "$MODEL_PATH")
OUTPUT_BASE_DIR="evaluation_results"

echo "========================================"
echo "Dynamic Model Evaluation"
echo "========================================"
echo "Model Path: $MODEL_PATH"
echo "Model Name: $MODEL_NAME"
echo "Erased Concept: $ERASED_CONCEPT"
echo "Unrelated Concept: $UNRELATED_CONCEPT"
echo "Number of Images: $NUM_IMAGES"
echo "========================================"

# Create output directory
OUTPUT_DIR="../$OUTPUT_BASE_DIR/${MODEL_NAME}"

# Run the evaluation script
python evaluate_erased_model.py \
    --model_path "$MODEL_PATH" \
    --erased_concept "$ERASED_CONCEPT" \
    --unrelated_concept "$UNRELATED_CONCEPT" \
    --num_images "$NUM_IMAGES" \
    --output_dir "$OUTPUT_DIR"

# Check if evaluation was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "Evaluation completed successfully!"
    echo "Results saved in: $OUTPUT_DIR"
    
    # Show the generated grids
    echo ""
    echo "Generated image grids:"
    find "$OUTPUT_DIR" -name "*_grid.png" -type f | while read grid; do
        echo "  - $(basename "$grid")"
    done
else
    echo ""
    echo "Error: Evaluation failed!"
    exit 1
fi