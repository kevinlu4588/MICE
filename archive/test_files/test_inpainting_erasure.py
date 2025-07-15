#!/usr/bin/env python3
"""
Quick test script for the Inpainting Erasure method.

This script demonstrates the complete workflow:
1. Train a small inpainting erasure model
2. Evaluate the results
3. Show the effectiveness of concept removal
"""

import argparse
import os
from inpainting_erasure import InpaintingEraser
from evaluate_inpainting_erasure import InpaintingErasureEvaluator

def run_quick_test(
    model_id="stabilityai/stable-diffusion-2-inpainting",
    device="cuda:0",
    num_images=10,
    iterations=50,
    num_tests=5
):
    """Run a quick test of the inpainting erasure system."""
    
    print("=" * 60)
    print("QUICK TEST: Inpainting Erasure for Van Gogh Concept")
    print("=" * 60)
    
    # Step 1: Train a small model
    print("\n1. Training small inpainting erasure model...")
    print("-" * 40)
    
    eraser = InpaintingEraser(model_id=model_id, device=device)
    
    losses = eraser.train(
        prompts_file="data/vangogh_prompts.csv",
        num_images=num_images,
        iterations=iterations,
        lr=1e-5,
        negative_guidance=2.0,
        train_method='esd-x',
        neutral_prompt="landscape painting",
        save_path="quick_test_models"
    )
    
    print(f"Training completed! Final loss: {losses[-1]:.6f}")
    
    # Step 2: Evaluate the model
    print("\n2. Evaluating trained model...")
    print("-" * 40)
    
    model_path = "quick_test_models/inpainting_esd_vangogh_esd-x.safetensors"
    
    evaluator = InpaintingErasureEvaluator(model_id=model_id, device=device)
    evaluator.load_esd_model(model_path)
    
    # Load prompts
    import pandas as pd
    prompts = pd.read_csv("data/vangogh_prompts.csv")['prompt'].tolist()
    
    # Generate comparisons
    results = evaluator.generate_comparison_images(
        prompts=prompts,
        num_tests=num_tests,
        output_dir="quick_test_results",
        seed_start=2000
    )
    
    # Create grids
    evaluator.create_comparison_grid(results, "quick_test_results")
    
    # Analyze
    analysis = evaluator.analyze_results(results, "quick_test_results")
    
    print("\n3. Test Results Summary")
    print("-" * 40)
    print(f"Total test cases: {analysis['total_tests']}")
    print(f"Mask types tested: {analysis['mask_types']}")
    print(f"Unique prompts used: {analysis['unique_prompts']}")
    
    print(f"\nResults saved to: quick_test_results/")
    print("Files to check:")
    print("- comparison_grid_*.png: Visual comparisons")
    print("- test_*_base.png: Original base images")
    print("- test_*_original_*.png: Original model inpainting")
    print("- test_*_esd_*.png: ESD model inpainting (should show less Van Gogh style)")
    
    print("\n" + "=" * 60)
    print("QUICK TEST COMPLETED!")
    print("=" * 60)
    print("\nTo run full training:")
    print("python inpainting_erasure.py --num_images 100 --iterations 500")
    print("\nTo evaluate existing model:")
    print("python evaluate_inpainting_erasure.py --esd_model_path <path_to_model>")


def main():
    parser = argparse.ArgumentParser(description="Quick test of inpainting erasure")
    parser.add_argument('--model_id', default="stabilityai/stable-diffusion-2-inpainting",
                       help='Model ID to use')
    parser.add_argument('--device', default="cuda:0",
                       help='Device to use')
    parser.add_argument('--num_images', type=int, default=10,
                       help='Number of training images (small for quick test)')
    parser.add_argument('--iterations', type=int, default=50,
                       help='Number of training iterations (small for quick test)')
    parser.add_argument('--num_tests', type=int, default=5,
                       help='Number of evaluation tests')
    
    args = parser.parse_args()
    
    run_quick_test(
        model_id=args.model_id,
        device=args.device,
        num_images=args.num_images,
        iterations=args.iterations,
        num_tests=args.num_tests
    )


if __name__ == "__main__":
    main()