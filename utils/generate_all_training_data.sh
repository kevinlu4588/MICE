#!/bin/bash

# Bash script to generate training data for all concepts
# Usage: ./generate_all_training_data.sh

# Define concepts
CONCEPTS=("picasso" "andy warhol" "english springer spaniel" "airliner" "chainsaw" "golf ball")
NUM_IMAGES=50
OUTPUT_DIR="../training_images"

echo "========================================"
echo "Generating Training Data for Concept Erasure"
echo "========================================"
echo "Concepts: ${CONCEPTS[@]}"
echo "Images per concept: $NUM_IMAGES"
echo "Output directory: $OUTPUT_DIR"
echo "========================================"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Generate training data
python generate_training_data.py \
    --concepts "${CONCEPTS[@]}" \
    --num_images $NUM_IMAGES \
    --output_dir "$OUTPUT_DIR"

# Check if generation was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "Training data generation completed successfully!"
    
    # Show summary of generated images
    echo ""
    echo "Generated training data summary:"
    for concept in "${CONCEPTS[@]}"; do
        concept_dir="${OUTPUT_DIR}/${concept// /_}"
        if [ -d "$concept_dir" ]; then
            num_files=$(ls "$concept_dir"/*.png 2>/dev/null | wc -l)
            echo "  - $concept: $num_files images in $concept_dir"
        fi
    done
else
    echo ""
    echo "Error: Training data generation failed!"
    exit 1
fi