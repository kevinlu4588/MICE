#!/usr/bin/env python
"""
Create comparison grid from existing generated images
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def create_grid_from_existing_images(output_dir):
    """Create grid from already generated images"""
    
    output_dir = Path(output_dir)
    
    # Define the expected directories and their display names
    model_dirs = [
        ("base_sd14", "Base SD v1.4"),
        ("ga_erased", "GA Erased"),
        ("esd_cross_attention_erased", "ESD Cross-Attention"),
        ("esd_esd-x_erased", "ESD-X"),
        ("esd_esd-u_erased", "ESD-U")
    ]
    
    # Get target concept images (first 2)
    target_images = {}
    for dir_name, display_name in model_dirs:
        dir_path = output_dir / dir_name
        if dir_path.exists():
            images = sorted(list(dir_path.glob("*.png")))[:2]
            if images:
                target_images[display_name] = images
    
    # Get unrelated concept images (all 5)
    unrelated_images = {}
    unrelated_dirs = [
        ("unrelated_base", "Base SD v1.4"),
        ("unrelated_ga", "GA Erased"),
        ("unrelated_esd_cross_attention", "ESD Cross-Attention"),
        ("unrelated_esd_esd-x", "ESD-X"),
        ("unrelated_esd_esd-u", "ESD-U")
    ]
    
    for dir_name, display_name in unrelated_dirs:
        dir_path = output_dir / dir_name
        if dir_path.exists():
            images = sorted(list(dir_path.glob("*.png")))[:5]
            if images:
                unrelated_images[display_name] = images
    
    # Calculate grid dimensions
    n_models = len(target_images)
    n_target = 2  # Number of target prompts
    n_unrelated = 5  # Number of unrelated prompts
    n_total_rows = n_target + 1 + n_unrelated  # +1 for separator
    
    if n_models == 0:
        print("No images found!")
        return
    
    # Create figure
    fig, axes = plt.subplots(n_total_rows, n_models, figsize=(5 * n_models, 5 * n_total_rows))
    
    if n_total_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_models == 1:
        axes = axes.reshape(-1, 1)
    
    # Get model names in consistent order
    model_names = list(target_images.keys())
    
    # Plot target concept images
    for row in range(n_target):
        for col, model_name in enumerate(model_names):
            if model_name in target_images and row < len(target_images[model_name]):
                img = mpimg.imread(target_images[model_name][row])
                axes[row, col].imshow(img)
                # Extract prompt from filename
                filename = target_images[model_name][row].name
                prompt = filename.split('_', 1)[1].replace('_', ' ').replace('.png', '')[:40]
                axes[row, col].set_title(f"{model_name}\n{prompt}...", fontsize=10)
                axes[row, col].axis('off')
    
    # Separator row
    separator_row = n_target
    for col in range(n_models):
        axes[separator_row, col].text(0.5, 0.5, 'UNRELATED CONCEPTS (Cars)', 
                                      ha='center', va='center', fontsize=14, weight='bold',
                                      transform=axes[separator_row, col].transAxes,
                                      bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        axes[separator_row, col].axis('off')
    
    # Plot unrelated concept images
    for row in range(n_unrelated):
        actual_row = separator_row + 1 + row
        for col, model_name in enumerate(model_names):
            if model_name in unrelated_images and row < len(unrelated_images[model_name]):
                img = mpimg.imread(unrelated_images[model_name][row])
                axes[actual_row, col].imshow(img)
                # Extract prompt from filename
                filename = unrelated_images[model_name][row].name
                prompt = filename.split('_', 1)[1].replace('_', ' ').replace('.png', '')[:40]
                axes[actual_row, col].set_title(f"{model_name}\n{prompt}...", fontsize=10)
                axes[actual_row, col].axis('off')
    
    plt.tight_layout()
    grid_path = output_dir / "van_gogh_comparison_grid.png"
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Grid saved to: {grid_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        create_grid_from_existing_images(sys.argv[1])
    else:
        # Default to the test directory
        create_grid_from_existing_images("test_eval_with_cars")