#!/bin/bash

# Example pipeline for generating training data using Claude prompts
# This script demonstrates how to:
# 1. Generate diverse prompts using Claude API
# 2. Use those prompts to generate training images

# The script will automatically load ANTHROPIC_API_KEY from ../.env file
# If you want to override it, uncomment the line below:
# export ANTHROPIC_API_KEY="your-api-key-here"

# Define concepts to process
CONCEPTS=("van gogh" "picasso" "french horn")

# Step 1: Generate prompts using Claude
echo "Step 1: Generating prompts using Claude API..."
python generate_prompts_claude.py \
    --concepts "${CONCEPTS[@]}" \
    --num_prompts 100

# Check if prompts were generated successfully
if [ ! -d "../datasets/prompt_csvs" ]; then
    echo "Error: Failed to generate prompts"
    exit 1
fi

echo ""
echo "Prompts generated successfully! CSV files:"
ls -la ../datasets/prompt_csvs/*.csv

# Step 2: Generate training images using the prompts
echo ""
echo "Step 2: Generating training images from prompts..."
python generate_training_data_from_csv.py \
    --num_images 100  # Optional: limit number of images

echo ""
echo "Training data generation complete!"
echo "Images saved in: ../datasets/training_images/"

# Optional: View some sample prompts
echo ""
echo "Sample prompts for Van Gogh:"
head -5 ../datasets/prompt_csvs/van_gogh_prompts.csv