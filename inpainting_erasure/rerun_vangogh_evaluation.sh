#!/bin/bash

# Script to re-run evaluation for Van Gogh concept

CONCEPT="van gogh"
RESULTS_DIR="multi_concept_results_20250714_213643"
CONCEPT_SANITIZED=$(echo "$CONCEPT" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
CONCEPT_DIR="$RESULTS_DIR/$CONCEPT_SANITIZED"
EVAL_PROMPTS=100

echo "Re-running evaluation for Van Gogh..."
echo "Output directory: $CONCEPT_DIR/evaluation"

# Create evaluation directory
mkdir -p "$CONCEPT_DIR/evaluation"

# Run evaluation with proper error handling
python evaluate_concept_models_multi.py \
    --concept "$CONCEPT" \
    --num_prompts $EVAL_PROMPTS \
    --output_dir "$CONCEPT_DIR/evaluation" \
    2>&1 | tee "$CONCEPT_DIR/logs/evaluation_rerun.log"

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "✓ Evaluation completed successfully!"
    echo "Results saved to: $CONCEPT_DIR/evaluation"
    
    # List the generated files
    echo ""
    echo "Generated files:"
    ls -la "$CONCEPT_DIR/evaluation/"
else
    echo "✗ Evaluation failed. Check the log at: $CONCEPT_DIR/logs/evaluation_rerun.log"
    exit 1
fi