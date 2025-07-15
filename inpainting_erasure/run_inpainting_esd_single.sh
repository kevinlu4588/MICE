#!/bin/bash

# Script to run inpainting erasure ESD training for a single concept

# Default parameters
CONCEPT="Van Gogh"
DATA_DIR="/share/u/kevin/erasing/datasets/training_images/van_gogh"
NUM_EPOCHS=10
BATCH_SIZE=1
LEARNING_RATE=5e-5
SAVE_EVERY=5
TRAIN_METHOD="esd-x"
NEGATIVE_GUIDANCE=2.0

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --concept)
            CONCEPT="$2"
            shift 2
            ;;
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --num_epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --train_method)
            TRAIN_METHOD="$2"
            shift 2
            ;;
        --negative_guidance)
            NEGATIVE_GUIDANCE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --concept CONCEPT --data_dir DATA_DIR [options]"
            echo "Options:"
            echo "  --concept CONCEPT             Concept to erase (default: 'van gogh')"
            echo "  --data_dir DATA_DIR          Directory containing training images (required)"
            echo "  --num_epochs NUM_EPOCHS      Number of training epochs (default: 10)"
            echo "  --batch_size BATCH_SIZE      Batch size (default: 1)"
            echo "  --learning_rate LR           Learning rate (default: 5e-5)"
            echo "  --train_method METHOD        ESD method: esd-x, esd-u, esd-all, esd-x-strict (default: esd-x)"
            echo "  --negative_guidance NG       Negative guidance value (default: 2.0)"
            exit 1
            ;;
    esac
done

# Check if data directory is provided
if [ -z "$DATA_DIR" ]; then
    echo "Error: --data_dir is required"
    echo "Usage: $0 --concept CONCEPT --data_dir DATA_DIR [options]"
    exit 1
fi

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory '$DATA_DIR' does not exist"
    exit 1
fi

# Print configuration
echo "======================================"
echo "Inpainting Erasure ESD Training"
echo "======================================"
echo "Concept to erase: $CONCEPT"
echo "Data directory: $DATA_DIR"
echo "Number of epochs: $NUM_EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LEARNING_RATE"
echo "Training method: $TRAIN_METHOD"
echo "Negative guidance: $NEGATIVE_GUIDANCE"
echo "======================================"

# Create log directory
LOG_DIR="logs"
mkdir -p $LOG_DIR

# Generate timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/inpainting_esd_${CONCEPT// /_}_${TIMESTAMP}.log"

echo "Starting training... (log: $LOG_FILE)"

# Run the training script
python train_inpainting_erasure_esd.py \
    --concept "$CONCEPT" \
    --data_dir "$DATA_DIR" \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --save_every $SAVE_EVERY \
    --train_method "$TRAIN_METHOD" \
    --negative_guidance $NEGATIVE_GUIDANCE \
    2>&1 | tee "$LOG_FILE"

# Check if training was successful
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo "Training completed successfully!"
    echo "Log saved to: $LOG_FILE"
    
    # Show the output directory
    MODEL_DIR=$(grep "Output directory:" "$LOG_FILE" | awk -F': ' '{print $2}')
    if [ ! -z "$MODEL_DIR" ]; then
        echo "Model saved to: $MODEL_DIR"
    fi
else
    echo ""
    echo "Training failed! Check the log file: $LOG_FILE"
    exit 1
fi