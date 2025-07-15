#!/bin/bash

# Script to evaluate erased models
# Usage: ./run_evaluation.sh [model_index]

# Set default values
ERASED_CONCEPT="van gogh"
UNRELATED_CONCEPT="a beautiful sunset over mountains"
NUM_IMAGES=10
OUTPUT_BASE_DIR="evaluation_results"

# Check if model index is provided as argument
if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_index>"
    echo "Available models:"
    echo "  0: Advanced Gradient Ascent"
    echo "  1: ESD Van Gogh"
    echo "  2: Gradient Ascent Erasure"
    echo "  3: Inpainting Erasure Van Gogh"
    echo "  5: Inpainting Erasure Van Gogh (v2)"
    exit 1
fi

MODEL_INDEX=$1

# Map model index to model path and name
case $MODEL_INDEX in
    0)
        MODEL_PATH="models/0_advanced_gradient_ascent"
        MODEL_NAME="advanced_gradient_ascent"
        ;;
    1)
        MODEL_PATH="models/1_esd_vangogh_xattn"
        MODEL_NAME="esd_vangogh"
        ;;
    2)
        MODEL_PATH="models/2_gradient_ascent_erasure"
        MODEL_NAME="gradient_ascent_erasure"
        ;;
    3)
        MODEL_PATH="models/3_inpainting_erasure_van_gogh"
        MODEL_NAME="inpainting_erasure_v1"
        ;;
    5)
        MODEL_PATH="models/5_inpainting_erasure_van_gogh"
        MODEL_NAME="inpainting_erasure_v2"
        ;;
    *)
        echo "Error: Invalid model index. Valid indices are: 0, 1, 2, 3, 5"
        exit 1
        ;;
esac

# Check if model directory exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model directory not found: $MODEL_PATH"
    exit 1
fi

echo "========================================"
echo "Evaluating Model: $MODEL_NAME"
echo "Model Path: $MODEL_PATH"
echo "Erased Concept: $ERASED_CONCEPT"
echo "Unrelated Concept: $UNRELATED_CONCEPT"
echo "Number of Images: $NUM_IMAGES"
echo "========================================"

# Run the evaluation script
python evaluate_erased_model.py \
    --model_path "$MODEL_PATH" \
    --erased_concept "$ERASED_CONCEPT" \
    --unrelated_concept "$UNRELATED_CONCEPT" \
    --num_images $NUM_IMAGES \
    --output_dir "$OUTPUT_BASE_DIR/$MODEL_NAME"

# Check if evaluation was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "Evaluation completed successfully!"
    echo "Results saved in: $OUTPUT_BASE_DIR/$MODEL_NAME"
    
    # List the output files
    echo ""
    echo "Generated files:"
    ls -la "$OUTPUT_BASE_DIR/$MODEL_NAME"/eval_*/
else
    echo ""
    echo "Error: Evaluation failed!"
    exit 1
fi