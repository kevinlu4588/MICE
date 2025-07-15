#!/bin/bash

# Script to run inpainting erasure training with hyperparameter sweep
# This script tests different hyperparameter combinations for a single concept

# Required parameters
CONCEPT=""
DATA_DIR=""
METHOD="esd"  # "esd" or "gradient_ascent"

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
        --method)
            METHOD="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --concept CONCEPT --data_dir DATA_DIR [--method METHOD]"
            echo "Options:"
            echo "  --concept CONCEPT        Concept to erase (required)"
            echo "  --data_dir DATA_DIR     Directory containing training images (required)"
            echo "  --method METHOD         'esd' or 'gradient_ascent' (default: esd)"
            exit 1
            ;;
    esac
done

# Validate required parameters
if [ -z "$CONCEPT" ] || [ -z "$DATA_DIR" ]; then
    echo "Error: --concept and --data_dir are required"
    echo "Usage: $0 --concept CONCEPT --data_dir DATA_DIR [--method METHOD]"
    exit 1
fi

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory '$DATA_DIR' does not exist"
    exit 1
fi

# Define hyperparameter sets
# You can modify these arrays to test different combinations

# Learning rates to test
LEARNING_RATES=(1e-5 5e-5 1e-4)

# Number of epochs to test
EPOCHS=(5 10 20)

# Batch sizes to test  
BATCH_SIZES=(1 2 4)

# For ESD method specific parameters
if [ "$METHOD" == "esd" ]; then
    # ESD training methods
    TRAIN_METHODS=("esd-x" "esd-x-strict")
    
    # Negative guidance values
    NEGATIVE_GUIDANCES=(1.0 2.0 3.0)
    
    SCRIPT="train_inpainting_erasure_esd.py"
else
    SCRIPT="train_inpainting_erasure.py"
fi

# Create results directory
RESULTS_DIR="hyperparam_results_${CONCEPT// /_}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"
LOG_DIR="$RESULTS_DIR/logs"
mkdir -p "$LOG_DIR"

# Summary file
SUMMARY_FILE="$RESULTS_DIR/summary.txt"

# Write header to summary file
echo "Hyperparameter Sweep Results" > "$SUMMARY_FILE"
echo "============================" >> "$SUMMARY_FILE"
echo "Concept: $CONCEPT" >> "$SUMMARY_FILE"
echo "Data Directory: $DATA_DIR" >> "$SUMMARY_FILE"
echo "Method: $METHOD" >> "$SUMMARY_FILE"
echo "Start Time: $(date)" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# Counter for experiments
EXPERIMENT_NUM=0

# Function to run a single experiment
run_experiment() {
    local lr=$1
    local epochs=$2
    local batch_size=$3
    local exp_name=$4
    local extra_args="${@:5}"
    
    EXPERIMENT_NUM=$((EXPERIMENT_NUM + 1))
    
    echo ""
    echo "======================================"
    echo "Experiment $EXPERIMENT_NUM: $exp_name"
    echo "======================================"
    echo "Learning Rate: $lr"
    echo "Epochs: $epochs"
    echo "Batch Size: $batch_size"
    echo "Additional Args: $extra_args"
    echo "======================================"
    
    # Log file for this experiment
    LOG_FILE="$LOG_DIR/exp_${EXPERIMENT_NUM}_${exp_name}.log"
    
    # Run the training
    echo "Starting experiment $EXPERIMENT_NUM..." | tee -a "$SUMMARY_FILE"
    
    START_TIME=$(date +%s)
    
    python "$SCRIPT" \
        --concept "$CONCEPT" \
        --data_dir "$DATA_DIR" \
        --num_epochs "$epochs" \
        --batch_size "$batch_size" \
        --learning_rate "$lr" \
        --save_every 5 \
        $extra_args \
        2>&1 | tee "$LOG_FILE"
    
    TRAINING_EXIT_CODE=${PIPESTATUS[0]}
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    # Extract final loss from log
    FINAL_LOSS=$(grep "average loss:" "$LOG_FILE" | tail -n 1 | awk -F': ' '{print $NF}')
    
    # Extract model directory
    MODEL_DIR=$(grep "Output directory:" "$LOG_FILE" | awk -F': ' '{print $2}')
    
    # Write to summary
    echo "" >> "$SUMMARY_FILE"
    echo "Experiment $EXPERIMENT_NUM: $exp_name" >> "$SUMMARY_FILE"
    echo "  Config: lr=$lr, epochs=$epochs, batch_size=$batch_size" >> "$SUMMARY_FILE"
    echo "  Additional: $extra_args" >> "$SUMMARY_FILE"
    echo "  Status: $([ $TRAINING_EXIT_CODE -eq 0 ] && echo 'SUCCESS' || echo 'FAILED')" >> "$SUMMARY_FILE"
    echo "  Final Loss: $FINAL_LOSS" >> "$SUMMARY_FILE"
    echo "  Duration: $DURATION seconds" >> "$SUMMARY_FILE"
    echo "  Model Dir: $MODEL_DIR" >> "$SUMMARY_FILE"
    
    if [ $TRAINING_EXIT_CODE -ne 0 ]; then
        echo "  ERROR: Training failed! Check log: $LOG_FILE" >> "$SUMMARY_FILE"
    fi
    
    # Small delay between experiments
    sleep 2
}

# Main hyperparameter sweep
if [ "$METHOD" == "esd" ]; then
    # ESD method sweep
    for train_method in "${TRAIN_METHODS[@]}"; do
        for ng in "${NEGATIVE_GUIDANCES[@]}"; do
            for lr in "${LEARNING_RATES[@]}"; do
                for epochs in "${EPOCHS[@]}"; do
                    for batch_size in "${BATCH_SIZES[@]}"; do
                        exp_name="lr${lr}_ep${epochs}_bs${batch_size}_${train_method}_ng${ng}"
                        run_experiment "$lr" "$epochs" "$batch_size" "$exp_name" \
                            "--train_method $train_method --negative_guidance $ng"
                    done
                done
            done
        done
    done
else
    # Gradient ascent method sweep
    for lr in "${LEARNING_RATES[@]}"; do
        for epochs in "${EPOCHS[@]}"; do
            for batch_size in "${BATCH_SIZES[@]}"; do
                exp_name="lr${lr}_ep${epochs}_bs${batch_size}"
                run_experiment "$lr" "$epochs" "$batch_size" "$exp_name"
            done
        done
    done
fi

# Final summary
echo "" | tee -a "$SUMMARY_FILE"
echo "======================================"  | tee -a "$SUMMARY_FILE"
echo "Hyperparameter Sweep Complete!" | tee -a "$SUMMARY_FILE"
echo "======================================"  | tee -a "$SUMMARY_FILE"
echo "Total Experiments: $EXPERIMENT_NUM" | tee -a "$SUMMARY_FILE"
echo "Results Directory: $RESULTS_DIR" | tee -a "$SUMMARY_FILE"
echo "Summary File: $SUMMARY_FILE" | tee -a "$SUMMARY_FILE"
echo "End Time: $(date)" | tee -a "$SUMMARY_FILE"

# Show best performing experiment (by final loss)
echo "" | tee -a "$SUMMARY_FILE"
echo "Best Performing Experiments (by final loss):" | tee -a "$SUMMARY_FILE"
grep -A 3 "Experiment" "$SUMMARY_FILE" | grep -E "(Experiment|Final Loss)" | \
    paste - - | sort -k5 -n | head -5 | tee -a "$SUMMARY_FILE"