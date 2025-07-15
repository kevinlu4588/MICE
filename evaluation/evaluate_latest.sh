#!/bin/bash

# Automatic evaluation script - loads parameters from most recent model in models.json
# Usage: ./evaluate_latest.sh [num_images]

# Default values
NUM_IMAGES="${1:-10}"  # Default to 10 images if not specified
UNRELATED_CONCEPT="a picture of a car"
OUTPUT_BASE_DIR="evaluation_results"

# Check if models.json exists
if [ ! -f "../utils/models.json" ]; then
    echo "Error: ../utils/models.json file not found!"
    exit 1
fi

# Use Python to parse models.json and find the most recent model
LATEST_MODEL_INFO=$(python3 -c "
import json
import sys

try:
    with open('../utils/models.json', 'r') as f:
        models = json.load(f)
    
    if not models:
        print('ERROR: No models found in models.json')
        sys.exit(1)
    
    # Find the highest index (most recent model)
    max_index = max(int(k) for k in models.keys())
    latest_model = models[str(max_index)]
    
    # Extract information
    model_name = latest_model.get('name', f'{max_index}_unknown')
    concept_erased = latest_model.get('concept_erased', 'unknown')
    
    # Print in format: index|name|concept
    print(f'{max_index}|{model_name}|{concept_erased}')
    
except Exception as e:
    print(f'ERROR: Failed to parse models.json: {e}')
    sys.exit(1)
")

# Check if Python script succeeded
if [[ $LATEST_MODEL_INFO == ERROR:* ]]; then
    echo "$LATEST_MODEL_INFO"
    exit 1
fi

# Parse the model information
IFS='|' read -r MODEL_INDEX MODEL_NAME ERASED_CONCEPT <<< "$LATEST_MODEL_INFO"

# Construct model path
MODEL_PATH="../models/$MODEL_NAME"

# Check if model directory exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model directory not found: $MODEL_PATH"
    echo "Looking for model index $MODEL_INDEX with name $MODEL_NAME"
    exit 1
fi

echo "========================================"
echo "Automatic Evaluation - Latest Model"
echo "========================================"
echo "Model Index: $MODEL_INDEX"
echo "Model Name: $MODEL_NAME"
echo "Model Path: $MODEL_PATH"
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
    
    # Show the generated files summary
    echo ""
    echo "Generated files summary:"
    echo "  - Erased concept images: $(ls "$OUTPUT_DIR"/eval_*/erased_concept/*.png 2>/dev/null | wc -l) files"
    echo "  - Unrelated concept images: $(ls "$OUTPUT_DIR"/eval_*/unrelated_concept/*.png 2>/dev/null | wc -l) files"
    echo "  - Image grids: $(ls "$OUTPUT_DIR"/eval_*/*_grid.png 2>/dev/null | wc -l) files"
    
    # Display grid locations
    echo ""
    echo "Image grids location:"
    find "$OUTPUT_DIR" -name "*_grid.png" -type f | while read grid; do
        echo "  - $grid"
    done
else
    echo ""
    echo "Error: Evaluation failed!"
    exit 1
fi