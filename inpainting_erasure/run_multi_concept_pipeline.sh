#!/bin/bash

# Multi-concept pipeline script: Process multiple concepts through the full pipeline
# For each concept: Generate data -> Train GA model -> Train ESD model -> Evaluate

# Hardcoded concepts to process
CONCEPTS=("van gogh" "picasso" "andy warhol" "airliner" "golf ball" "french horn" "english springer spaniel" "garbage truck" "church")

# Data generation parameters
NUM_IMAGES=100
EVAL_PROMPTS=100

# Gradient Ascent hyperparameters
GA_NUM_EPOCHS=1
GA_BATCH_SIZE=1
GA_LEARNING_RATE=2e-5

# ESD hyperparameters
ESD_NUM_EPOCHS=10
ESD_BATCH_SIZE=1
ESD_LEARNING_RATE=5e-5
ESD_METHODS=("esd-x" "esd-u")  # Train both methods
ESD_NEGATIVE_GUIDANCE=2.0

# Pipeline control flags
SKIP_DATA_GEN=false
SKIP_GA=false
SKIP_ESD=false
SKIP_EVAL=false

# Setup main results directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_RESULTS_DIR="multi_concept_results_${TIMESTAMP}"
mkdir -p "$MAIN_RESULTS_DIR"

# Create main summary log
MAIN_SUMMARY="${MAIN_RESULTS_DIR}/all_concepts_summary.log"

# Function to log main summary
log_main() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$MAIN_SUMMARY"
}

# Function to process a single concept
process_concept() {
    local concept="$1"
    local concept_sanitized="${concept// /_}"
    local concept_dir="${MAIN_RESULTS_DIR}/${concept_sanitized}"
    
    log_main "Processing concept: $concept"
    
    # Create concept directory
    mkdir -p "$concept_dir"
    mkdir -p "$concept_dir/logs"
    
    # Step 1: Generate training data
    if [ "$SKIP_DATA_GEN" = false ]; then
        log_main "  Generating training data for '$concept'..."
        python ../utils/generate_diverse_training_data.py \
            --concepts "$concept" \
            --num_images $NUM_IMAGES \
            --output_dir "training_images" \
            > "$concept_dir/logs/data_generation.log" 2>&1
        
        if [ $? -ne 0 ]; then
            log_main "  ✗ Failed to generate training data for '$concept'"
            return 1
        fi
        log_main "  ✓ Generated training data"
    else
        log_main "  Skipping data generation"
    fi
    
    # Step 2: Train Gradient Ascent model
    if [ "$SKIP_GA" = false ]; then
        log_main "  Training Gradient Ascent model for '$concept'..."
        python train_inpainting_erasure.py \
            --concept "$concept" \
            --data_dir "training_images/${concept_sanitized}" \
            --num_epochs $GA_NUM_EPOCHS \
            --batch_size $GA_BATCH_SIZE \
            --learning_rate $GA_LEARNING_RATE \
            --save_every 5 \
            > "$concept_dir/logs/ga_training.log" 2>&1
        
        if [ $? -ne 0 ]; then
            log_main "  ✗ Failed to train GA model for '$concept'"
            return 1
        fi
        log_main "  ✓ Trained GA model"
    else
        log_main "  Skipping GA training"
    fi
    
    # Step 3: Train ESD models (both esd-x and esd-u)
    if [ "$SKIP_ESD" = false ]; then
        for esd_method in "${ESD_METHODS[@]}"; do
            log_main "  Training ESD model ($esd_method) for '$concept'..."
            python train_inpainting_erasure_esd.py \
                --concept "$concept" \
                --data_dir "training_images/${concept_sanitized}" \
                --num_epochs $ESD_NUM_EPOCHS \
                --batch_size $ESD_BATCH_SIZE \
                --learning_rate $ESD_LEARNING_RATE \
                --train_method "$esd_method" \
                --negative_guidance $ESD_NEGATIVE_GUIDANCE \
                --save_every 5 \
                > "$concept_dir/logs/esd_${esd_method}_training.log" 2>&1
            
            if [ $? -ne 0 ]; then
                log_main "  ✗ Failed to train ESD model ($esd_method) for '$concept'"
                return 1
            fi
            log_main "  ✓ Trained ESD model ($esd_method)"
        done
    else
        log_main "  Skipping ESD training"
    fi
    
    # Step 4: Evaluate the models
    if [ "$SKIP_EVAL" = false ]; then
        log_main "  Evaluating models for '$concept'..."
        
        # Create evaluation directory
        mkdir -p "$concept_dir/evaluation"
        
        # Run evaluation script
        python evaluate_concept_models_multi.py \
            --concept "$concept" \
            --num_prompts $EVAL_PROMPTS \
            --output_dir "$concept_dir/evaluation" \
            > "$concept_dir/logs/evaluation.log" 2>&1
        
        if [ $? -ne 0 ]; then
            log_main "  ✗ Failed to evaluate models for '$concept'"
            return 1
        fi
        log_main "  ✓ Evaluated models"
    else
        log_main "  Skipping evaluation"
    fi
    
    log_main "  ✓ Completed processing for '$concept'"
    return 0
}

# Start processing
log_main "======================================"
log_main "Multi-Concept Pipeline Started"
log_main "======================================"
log_main "Total concepts: ${#CONCEPTS[@]}"
log_main "Concepts: ${CONCEPTS[*]}"
log_main "======================================"

# Save configuration
cat > "${MAIN_RESULTS_DIR}/config.txt" << EOF
Multi-Concept Pipeline Configuration
===================================
Total concepts: ${#CONCEPTS[@]}
Concepts: ${CONCEPTS[*]}
Number of training images per concept: $NUM_IMAGES

Gradient Ascent Hyperparameters:
  Number of epochs: $GA_NUM_EPOCHS
  Batch size: $GA_BATCH_SIZE
  Learning rate: $GA_LEARNING_RATE

ESD Hyperparameters:
  Number of epochs: $ESD_NUM_EPOCHS
  Batch size: $ESD_BATCH_SIZE
  Learning rate: $ESD_LEARNING_RATE
  Methods: ${ESD_METHODS[*]}
  Negative guidance: $ESD_NEGATIVE_GUIDANCE

Evaluation prompts: $EVAL_PROMPTS
Timestamp: $TIMESTAMP
EOF

# Track results
SUCCESSFUL_CONCEPTS=()
FAILED_CONCEPTS=()

# Process each concept sequentially
CONCEPT_COUNT=0
for concept in "${CONCEPTS[@]}"; do
    CONCEPT_COUNT=$((CONCEPT_COUNT + 1))
    log_main ""
    log_main "======================================" 
    log_main "Processing concept $CONCEPT_COUNT/${#CONCEPTS[@]}: $concept"
    log_main "======================================"
    
    START_TIME=$(date +%s)
    
    if process_concept "$concept"; then
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        SUCCESSFUL_CONCEPTS+=("$concept")
        log_main "✓ Successfully processed '$concept' in ${DURATION}s"
    else
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        FAILED_CONCEPTS+=("$concept")
        log_main "✗ Failed to process '$concept' after ${DURATION}s"
    fi
done

# Generate final summary
log_main ""
log_main "======================================"
log_main "Multi-Concept Pipeline Complete!"
log_main "======================================"
log_main ""
log_main "Results Summary:"
log_main "---------------"
log_main "Total concepts: ${#CONCEPTS[@]}"
log_main "Successful: ${#SUCCESSFUL_CONCEPTS[@]}"
log_main "Failed: ${#FAILED_CONCEPTS[@]}"
log_main ""

# Detailed results
if [ ${#SUCCESSFUL_CONCEPTS[@]} -gt 0 ]; then
    log_main "Successful Concepts:"
    for concept in "${SUCCESSFUL_CONCEPTS[@]}"; do
        log_main "  ✓ $concept"
    done
fi

if [ ${#FAILED_CONCEPTS[@]} -gt 0 ]; then
    log_main ""
    log_main "Failed Concepts:"
    for concept in "${FAILED_CONCEPTS[@]}"; do
        log_main "  ✗ $concept"
    done
fi

log_main ""
log_main "Results directory: $MAIN_RESULTS_DIR"

# Create detailed results CSV
cat > "${MAIN_RESULTS_DIR}/results_summary.csv" << EOF
Concept,Status,GA Model,ESD Model,Evaluation Images
EOF

for concept in "${CONCEPTS[@]}"; do
    concept_sanitized="${concept// /_}"
    concept_dir="${MAIN_RESULTS_DIR}/${concept_sanitized}"
    
    # Determine status
    status="FAILED"
    for success in "${SUCCESSFUL_CONCEPTS[@]}"; do
        if [ "$concept" = "$success" ]; then
            status="SUCCESS"
            break
        fi
    done
    
    # Extract model info if available
    ga_model=""
    esd_model=""
    eval_count=0
    
    if [ -d "$concept_dir" ]; then
        # Try to find model info from logs
        if [ -f "$concept_dir/logs/ga_training.log" ]; then
            ga_model=$(grep "model index:" "$concept_dir/logs/ga_training.log" | tail -1 | awk -F': ' '{print $2}' | xargs || echo "N/A")
        fi
        
        if [ -f "$concept_dir/logs/esd_training.log" ]; then
            esd_model=$(grep "model index:" "$concept_dir/logs/esd_training.log" | tail -1 | awk -F': ' '{print $2}' | xargs || echo "N/A")
        fi
        
        # Count evaluation images
        if [ -d "$concept_dir/evaluation" ]; then
            eval_count=$(find "$concept_dir/evaluation" -name "*.png" -o -name "*.jpg" 2>/dev/null | wc -l)
        fi
    fi
    
    echo "$concept,$status,$ga_model,$esd_model,$eval_count" >> "${MAIN_RESULTS_DIR}/results_summary.csv"
done

# Create convenience viewer script
cat > "${MAIN_RESULTS_DIR}/view_all_results.sh" << 'EOF'
#!/bin/bash
echo "Multi-Concept Pipeline Results"
echo "=============================="
echo ""
echo "Summary:"
cat all_concepts_summary.log | tail -20
echo ""
echo "Detailed Results:"
cat results_summary.csv | column -t -s','
echo ""
echo "Individual concept results:"
for dir in */; do
    if [ -d "$dir" ] && [ "$dir" != "*/" ]; then
        echo "  - $dir"
    fi
done
echo ""
echo "To view a specific concept's results:"
echo "  cd CONCEPT_NAME"
EOF

chmod +x "${MAIN_RESULTS_DIR}/view_all_results.sh"

log_main ""
log_main "To view all results: cd $MAIN_RESULTS_DIR && ./view_all_results.sh"

# Exit with appropriate code
if [ ${#FAILED_CONCEPTS[@]} -gt 0 ]; then
    exit 1
else
    exit 0
fi