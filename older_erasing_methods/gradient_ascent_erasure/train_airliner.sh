#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Environment variables
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
BASE_TRAIN_DIR="/share/u/kevin/erasing/inpainting_erasure"
BASE_OUTPUT_DIR="/share/u/kevin/erasing/models/gradient_ascent_erasure"

# List of concepts
concepts=(
    "van_gogh"
)

# Replace "concept" with the first concept in the list
first_concept="${concepts[0]}"

# Prompts to use for validation (replacing "concept" with the first concept)
prompts=(
    "picture of a $first_concept"
    "photo of a $first_concept"
    "$first_concept"
    "portrait of a $first_concept"
    "a picture of a $first_concept"
    "a picture of a car"
)

# Join prompts into a single string separated by semicolons (or any other delimiter)
prompt_args=$(IFS=";"; echo "${prompts[*]}")

# Iterate through each concept and train the model
for concept in "${concepts[@]}"; do
    echo "Starting training for concept: $concept"

    # Convert concept name to lowercase and replace spaces with underscores
    concept_safe=$(echo "$concept" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')

    # Set concept-specific paths
    TRAIN_DIR="${BASE_TRAIN_DIR}/${concept_safe}"
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/ga_${concept_safe}"

    # Ensure output directory exists
    mkdir -p "$OUTPUT_DIR"

    # Check if training data directory exists
    if [ ! -d "$TRAIN_DIR" ]; then
        echo "Training directory $TRAIN_DIR does not exist. Skipping concept: $concept."
        continue
    fi

    # Run training command, passing the joined prompts as a single argument
    echo "Running training for $concept_safe..."
    accelerate launch --mixed_precision="fp16" train_text_to_image.py \
      --pretrained_model_name_or_path="$MODEL_NAME" \
      --train_data_dir="$TRAIN_DIR" \
      --use_ema \
      --resolution=512 --center_crop --random_flip \
      --train_batch_size=5 \
      --gradient_accumulation_steps=4 \
      --gradient_checkpointing \
      --max_train_steps=10 \
      --learning_rate=1e-05 \
      --max_grad_norm=1 \
      --lr_scheduler="constant" --lr_warmup_steps=0 \
      --validation_epochs=1 \
      --output_dir="$OUTPUT_DIR" \
      --validation_prompts="$prompt_args" \
      --step_finisher=10

    # Check for success and provide feedback
    if [ $? -eq 0 ]; then
        echo "Training completed successfully for concept: $concept"
    else
        echo "Training failed for concept: $concept. Check logs for details."
        exit 1
    fi
done

echo "Training completed for all concepts."
