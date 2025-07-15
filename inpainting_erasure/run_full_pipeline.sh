#!/bin/bash

# Full pipeline script: Generate data -> Train GA model -> Train ESD model -> Evaluate

# Default parameters
CONCEPT="Van Gogh"
NUM_IMAGES=100
NUM_EPOCHS=10
BATCH_SIZE=1
LEARNING_RATE=5e-5
ESD_METHOD="esd-x"
NEGATIVE_GUIDANCE=2.0
EVAL_PROMPTS=100
SKIP_DATA_GEN=false
SKIP_GA=false
SKIP_ESD=false
SKIP_EVAL=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --concept)
            CONCEPT="$2"
            shift 2
            ;;
        --num_images)
            NUM_IMAGES="$2"
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
        --esd_method)
            ESD_METHOD="$2"
            shift 2
            ;;
        --negative_guidance)
            NEGATIVE_GUIDANCE="$2"
            shift 2
            ;;
        --eval_prompts)
            EVAL_PROMPTS="$2"
            shift 2
            ;;
        --skip_data_gen)
            SKIP_DATA_GEN=true
            shift
            ;;
        --skip_ga)
            SKIP_GA=true
            shift
            ;;
        --skip_esd)
            SKIP_ESD=true
            shift
            ;;
        --skip_eval)
            SKIP_EVAL=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 --concept CONCEPT [options]"
            echo ""
            echo "Run full pipeline: Generate data -> Train GA -> Train ESD -> Evaluate"
            echo ""
            echo "Required:"
            echo "  --concept CONCEPT              Concept to erase (e.g., 'van gogh', 'picasso')"
            echo ""
            echo "Options:"
            echo "  --num_images NUM              Number of training images to generate (default: 50)"
            echo "  --num_epochs NUM              Number of training epochs (default: 10)"
            echo "  --batch_size SIZE             Batch size for training (default: 1)"
            echo "  --learning_rate LR            Learning rate (default: 5e-5)"
            echo "  --esd_method METHOD           ESD method: esd-x, esd-u, esd-all, esd-x-strict (default: esd-x)"
            echo "  --negative_guidance NG        Negative guidance for ESD (default: 2.0)"
            echo "  --eval_prompts NUM            Number of prompts for evaluation (default: 100)"
            echo ""
            echo "Skip options:"
            echo "  --skip_data_gen               Skip data generation step"
            echo "  --skip_ga                     Skip gradient ascent training"
            echo "  --skip_esd                    Skip ESD training"
            echo "  --skip_eval                   Skip evaluation"
            echo ""
            echo "Example:"
            echo "  $0 --concept 'van gogh' --num_images 100 --num_epochs 20"
            echo "  $0 --concept 'picasso' --skip_data_gen --skip_ga  # Only train ESD and evaluate"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required parameters
if [ -z "$CONCEPT" ]; then
    echo "Error: --concept is required"
    echo "Use --help for usage information"
    exit 1
fi

# Setup directories
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
PIPELINE_DIR="pipeline_results/${CONCEPT// /_}_${TIMESTAMP}"
LOG_DIR="${PIPELINE_DIR}/logs"
DATA_DIR="training_images/${CONCEPT// /_}"

mkdir -p "$LOG_DIR"

# Summary log file
SUMMARY_LOG="${PIPELINE_DIR}/pipeline_summary.log"

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$SUMMARY_LOG"
}

# Function to check if previous step succeeded
check_status() {
    if [ $1 -ne 0 ]; then
        log "ERROR: $2 failed! Exit code: $1"
        log "Pipeline aborted."
        exit 1
    else
        log "SUCCESS: $2 completed successfully"
    fi
}

# Start pipeline
log "======================================"
log "Starting Full Pipeline"
log "======================================"
log "Concept: $CONCEPT"
log "Pipeline directory: $PIPELINE_DIR"
log "======================================"

# Save configuration
cat > "${PIPELINE_DIR}/config.txt" << EOF
Pipeline Configuration
=====================
Concept: $CONCEPT
Number of training images: $NUM_IMAGES
Number of epochs: $NUM_EPOCHS
Batch size: $BATCH_SIZE
Learning rate: $LEARNING_RATE
ESD method: $ESD_METHOD
Negative guidance: $NEGATIVE_GUIDANCE
Evaluation prompts: $EVAL_PROMPTS
Timestamp: $TIMESTAMP
EOF

# Step 1: Generate training data
if [ "$SKIP_DATA_GEN" = false ]; then
    log ""
    log "STEP 1: Generating training data"
    log "================================="
    
    # Check if data already exists
    if [ -d "$DATA_DIR" ]; then
        EXISTING_COUNT=$(find "$DATA_DIR" -name "*.png" 2>/dev/null | wc -l)
        if [ $EXISTING_COUNT -ge $NUM_IMAGES ]; then
            log "Found $EXISTING_COUNT existing images in $DATA_DIR, skipping generation"
        else
            log "Found only $EXISTING_COUNT images, generating more..."
            
            ./generate_concept_training_data.sh \
                --concept "$CONCEPT" \
                --num_images $NUM_IMAGES \
                --output_dir "training_images" \
                2>&1 | tee "${LOG_DIR}/data_generation.log"
            
            check_status ${PIPESTATUS[0]} "Data generation"
        fi
    else
        log "Generating $NUM_IMAGES training images..."
        
        ./generate_concept_training_data.sh \
            --concept "$CONCEPT" \
            --num_images $NUM_IMAGES \
            --output_dir "training_images" \
            2>&1 | tee "${LOG_DIR}/data_generation.log"
        
        check_status ${PIPESTATUS[0]} "Data generation"
    fi
else
    log "Skipping data generation (--skip_data_gen)"
fi

# Verify data directory exists
if [ ! -d "$DATA_DIR" ]; then
    log "ERROR: Training data directory not found: $DATA_DIR"
    exit 1
fi

# Count available training images
TRAIN_IMAGE_COUNT=$(find "$DATA_DIR" -name "*.png" 2>/dev/null | wc -l)
log "Found $TRAIN_IMAGE_COUNT training images in $DATA_DIR"

# Step 2: Train Gradient Ascent model
GA_MODEL_DIR=""
if [ "$SKIP_GA" = false ]; then
    log ""
    log "STEP 2: Training Gradient Ascent model"
    log "======================================"
    
    python train_inpainting_erasure.py \
        --concept "$CONCEPT" \
        --data_dir "$DATA_DIR" \
        --num_epochs $NUM_EPOCHS \
        --batch_size $BATCH_SIZE \
        --learning_rate $LEARNING_RATE \
        --save_every 5 \
        2>&1 | tee "${LOG_DIR}/ga_training.log"
    
    check_status ${PIPESTATUS[0]} "Gradient Ascent training"
    
    # Extract model directory from log
    GA_MODEL_DIR=$(grep "Output directory:" "${LOG_DIR}/ga_training.log" | awk -F': ' '{print $2}')
    log "GA model saved to: $GA_MODEL_DIR"
else
    log "Skipping Gradient Ascent training (--skip_ga)"
fi

# Step 3: Train ESD model
ESD_MODEL_DIR=""
if [ "$SKIP_ESD" = false ]; then
    log ""
    log "STEP 3: Training ESD model"
    log "=========================="
    
    python train_inpainting_erasure_esd.py \
        --concept "$CONCEPT" \
        --data_dir "$DATA_DIR" \
        --num_epochs $NUM_EPOCHS \
        --batch_size $BATCH_SIZE \
        --learning_rate $LEARNING_RATE \
        --train_method "$ESD_METHOD" \
        --negative_guidance $NEGATIVE_GUIDANCE \
        --save_every 5 \
        2>&1 | tee "${LOG_DIR}/esd_training.log"
    
    check_status ${PIPESTATUS[0]} "ESD training"
    
    # Extract model directory from log
    ESD_MODEL_DIR=$(grep "Output directory:" "${LOG_DIR}/esd_training.log" | awk -F': ' '{print $2}')
    log "ESD model saved to: $ESD_MODEL_DIR"
else
    log "Skipping ESD training (--skip_esd)"
fi

# Step 4: Evaluate models
if [ "$SKIP_EVAL" = false ]; then
    log ""
    log "STEP 4: Evaluating models"
    log "========================="
    
    # Find evaluation script
    EVAL_SCRIPT=""
    if [ -f "../evaluation/evaluate_inpainting_models.py" ]; then
        EVAL_SCRIPT="../evaluation/evaluate_inpainting_models.py"
    elif [ -f "evaluate_models.py" ]; then
        EVAL_SCRIPT="evaluate_models.py"
    else
        log "WARNING: Evaluation script not found. Trying to find generate-images.py..."
        
        # Look for alternative evaluation scripts
        if [ -f "../evalscripts/generate-images.py" ]; then
            EVAL_SCRIPT="../evalscripts/generate-images.py"
        else
            log "ERROR: No evaluation script found!"
            log "Skipping evaluation step"
            SKIP_EVAL=true
        fi
    fi
    
    if [ "$SKIP_EVAL" = false ] && [ ! -z "$EVAL_SCRIPT" ]; then
        # Create evaluation output directory
        EVAL_DIR="${PIPELINE_DIR}/evaluation"
        mkdir -p "$EVAL_DIR"
        
        # Get model indices from models.json
        if [ -f "../utils/models.json" ]; then
            log "Reading model indices from models.json..."
            
            # If we have both models, evaluate both
            if [ ! -z "$GA_MODEL_DIR" ] && [ ! -z "$ESD_MODEL_DIR" ]; then
                log "Evaluating both GA and ESD models..."
                
                # Extract model indices from directory names
                GA_INDEX=$(basename "$GA_MODEL_DIR" | cut -d'_' -f1)
                ESD_INDEX=$(basename "$ESD_MODEL_DIR" | cut -d'_' -f1)
                
                log "GA model index: $GA_INDEX"
                log "ESD model index: $ESD_INDEX"
                
                # Run evaluation with appropriate parameters based on script
                if [[ "$EVAL_SCRIPT" == *"generate-images.py" ]]; then
                    # Using the generate-images.py script
                    python "$EVAL_SCRIPT" \
                        --model_indices $GA_INDEX $ESD_INDEX \
                        --num_samples $EVAL_PROMPTS \
                        --output_dir "$EVAL_DIR" \
                        2>&1 | tee "${LOG_DIR}/evaluation.log"
                else
                    # Using custom evaluation script
                    python "$EVAL_SCRIPT" \
                        --ga_model "$GA_MODEL_DIR" \
                        --esd_model "$ESD_MODEL_DIR" \
                        --concept "$CONCEPT" \
                        --num_prompts $EVAL_PROMPTS \
                        --output_dir "$EVAL_DIR" \
                        2>&1 | tee "${LOG_DIR}/evaluation.log"
                fi
                
            elif [ ! -z "$GA_MODEL_DIR" ]; then
                log "Evaluating GA model only..."
                GA_INDEX=$(basename "$GA_MODEL_DIR" | cut -d'_' -f1)
                
                python "$EVAL_SCRIPT" \
                    --model_indices $GA_INDEX \
                    --num_samples $EVAL_PROMPTS \
                    --output_dir "$EVAL_DIR" \
                    2>&1 | tee "${LOG_DIR}/evaluation.log"
                    
            elif [ ! -z "$ESD_MODEL_DIR" ]; then
                log "Evaluating ESD model only..."
                ESD_INDEX=$(basename "$ESD_MODEL_DIR" | cut -d'_' -f1)
                
                python "$EVAL_SCRIPT" \
                    --model_indices $ESD_INDEX \
                    --num_samples $EVAL_PROMPTS \
                    --output_dir "$EVAL_DIR" \
                    2>&1 | tee "${LOG_DIR}/evaluation.log"
            else
                log "WARNING: No models to evaluate!"
            fi
            
            check_status ${PIPESTATUS[0]} "Model evaluation"
            
            # Check if evaluation images were generated
            if [ -d "$EVAL_DIR" ]; then
                EVAL_IMAGE_COUNT=$(find "$EVAL_DIR" -name "*.png" -o -name "*.jpg" 2>/dev/null | wc -l)
                log "Generated $EVAL_IMAGE_COUNT evaluation images"
            fi
        else
            log "WARNING: models.json not found, skipping evaluation"
        fi
    fi
else
    log "Skipping evaluation (--skip_eval)"
fi

# Final summary
log ""
log "======================================"
log "Pipeline Complete!"
log "======================================"
log "Results directory: $PIPELINE_DIR"
log ""
log "Summary:"
log "- Training images: $TRAIN_IMAGE_COUNT"

if [ ! -z "$GA_MODEL_DIR" ]; then
    log "- GA model: $GA_MODEL_DIR"
fi

if [ ! -z "$ESD_MODEL_DIR" ]; then
    log "- ESD model: $ESD_MODEL_DIR"
fi

if [ -d "$EVAL_DIR" ] && [ ! -z "$EVAL_IMAGE_COUNT" ]; then
    log "- Evaluation images: $EVAL_IMAGE_COUNT in $EVAL_DIR"
fi

log ""
log "Logs saved in: $LOG_DIR"
log "Full summary: $SUMMARY_LOG"

# Create a convenience script to view results
cat > "${PIPELINE_DIR}/view_results.sh" << 'EOF'
#!/bin/bash
echo "Pipeline Results"
echo "================"
echo ""
echo "Configuration:"
cat config.txt
echo ""
echo "Logs:"
ls -la logs/
echo ""
if [ -d "evaluation" ]; then
    echo "Evaluation Results:"
    ls -la evaluation/ | head -20
fi
EOF

chmod +x "${PIPELINE_DIR}/view_results.sh"

log ""
log "To view results, run: ${PIPELINE_DIR}/view_results.sh"