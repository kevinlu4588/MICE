#!/bin/bash

# Train inpainting erasure model for Van Gogh
echo "Training inpainting erasure model..."
python train_inpainting_erasure.py \
    --concept "van gogh" \
    --data_dir "/share/u/kevin/erasing/inpainting_erasure/van_gogh/train" \
    --model_index 5 \
    --num_epochs 2 \
    --batch_size 5 \
    --learning_rate 5e-5 \
    --save_every 10

# # Run evaluation
# echo "Running comprehensive evaluation..."
# python evaluation/comprehensive_evaluation.py \
#     --model_path "models/3_inpainting_erasure_van gogh/unet/diffusion_pytorch_model.safetensors" \
#     --prompt "a painting by Vincent van Gogh" \
#     --num_tests 5 \
#     --output_dir "evaluation_results/3_inpainting_erasure_van_gogh"

# echo "Training and evaluation complete!"