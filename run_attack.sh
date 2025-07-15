#!/bin/bash

# Simple script to run unlearndiffatk attack
# Edit the variables below to configure your attack

# === CONFIGURATION ===
MODEL_PATH="/share/u/kevin/erasing/models/21_inpainting_erasure_esd_van_gogh/unet/diffusion_pytorch_model.safetensors"  # Change this to your model path
CONCEPT="van_gogh"                             # Change this to your concept
ITERATIONS=40
ATTACK_IDX=0
METHOD="esdx"

# Optional: uncomment and set these if needed
# OUTPUT_DIR="./results/my_custom_output"
# DATASET_PATH="/path/to/custom/dataset"

# === RUN ATTACK ===
echo "Running unlearndiffatk attack"
echo "Model: $MODEL_PATH"
echo "Concept: $CONCEPT"
echo "Method: $METHOD"
echo "Iterations: $ITERATIONS"
echo "Attack index: $ATTACK_IDX"
echo ""

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found: $MODEL_PATH"
    echo "Please update MODEL_PATH in this script"
    exit 1
fi

# Build command
CMD="python /share/u/kevin/erasing/run_unlearndiffatk.py"
CMD="$CMD --model-path \"$MODEL_PATH\""
CMD="$CMD --concept \"$CONCEPT\""
CMD="$CMD --iterations $ITERATIONS"
CMD="$CMD --attack-idx $ATTACK_IDX"
CMD="$CMD --method \"$METHOD\""

# Add optional parameters if set
if [ -n "$OUTPUT_DIR" ]; then
    CMD="$CMD --output-dir \"$OUTPUT_DIR\""
fi

if [ -n "$DATASET_PATH" ]; then
    CMD="$CMD --dataset-path \"$DATASET_PATH\""
fi

# Run the attack
echo "Executing: $CMD"
echo ""
eval $CMD

echo ""
echo "Done!"