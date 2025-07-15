#!/bin/bash

echo "=== Inpainting ESD Training Monitor & Auto-Evaluator ==="
echo "Monitoring training progress and will run evaluation when complete..."

MODEL_DIR="/share/u/kevin/erasing/inpainting_erasure"
OUTPUT_MODEL="$MODEL_DIR/inpainting_esd_model/inpainting_esd_vangogh.safetensors"
EVAL_DIR="/share/u/kevin/erasing/evaluation"

# Monitor training process
while true; do
    # Check if training process is still running
    if pgrep -f "inpainting_esd_training.py" > /dev/null; then
        # Get current progress from log
        PROGRESS=$(tail -1 "$MODEL_DIR/training.log" | grep -o '[0-9]\+%' | tail -1)
        if [ ! -z "$PROGRESS" ]; then
            echo "$(date '+%H:%M:%S') - Training progress: $PROGRESS"
        fi
        sleep 30  # Check every 30 seconds
    else
        echo "$(date '+%H:%M:%S') - Training process completed!"
        break
    fi
done

# Wait a moment for file system to settle
sleep 5

# Check if model was saved successfully
if [ -f "$OUTPUT_MODEL" ]; then
    echo "$(date '+%H:%M:%S') - Model found at: $OUTPUT_MODEL"
    echo "$(date '+%H:%M:%S') - Starting comprehensive evaluation..."
    
    cd "$EVAL_DIR"
    python comprehensive_evaluation.py \
        --model_path "$OUTPUT_MODEL" \
        --prompt "a painting by Vincent van Gogh" \
        --num_tests 8 \
        --output_dir "inpainting_esd_evaluation_results"
    
    echo "$(date '+%H:%M:%S') - Evaluation completed!"
    echo "Results saved to: $EVAL_DIR/inpainting_esd_evaluation_results/"
else
    echo "$(date '+%H:%M:%S') - ERROR: Model file not found at $OUTPUT_MODEL"
    echo "Checking training log for errors..."
    tail -20 "$MODEL_DIR/training.log"
fi