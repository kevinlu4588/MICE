#!/usr/bin/env python
"""
Template for model training scripts that follow the models.json update rules
"""

import os
import json
from datetime import datetime

def get_next_model_index():
    """Get the next model index from models.json"""
    models_json_path = "models.json"
    if os.path.exists(models_json_path):
        with open(models_json_path, 'r') as f:
            models_data = json.load(f)
        # Find the highest index and add 1
        if models_data:
            next_index = max(int(k) for k in models_data.keys()) + 1
        else:
            next_index = 0
    else:
        next_index = 0
        models_data = {}
    return next_index, models_data, models_json_path

def update_models_json(model_index, model_info, models_data, models_json_path):
    """Update models.json with new model information"""
    models_data[str(model_index)] = model_info
    with open(models_json_path, 'w') as f:
        json.dump(models_data, f, indent=2)
    print(f"Updated models.json with model index {model_index}")

# Example usage in training script:
def train_model(args):
    # Get next model index
    next_index, models_data, models_json_path = get_next_model_index()
    
    # Override with manual index if specified and it's higher
    if hasattr(args, 'model_index') and args.model_index is not None and args.model_index >= next_index:
        next_index = args.model_index
    
    # Create model name and output directory
    model_name = f"{next_index}_{args.method}_{args.concept.replace(' ', '_')}"
    output_dir = f"models/{model_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Training {args.method} model for concept: {args.concept}")
    print(f"Model index: {next_index}")
    print(f"Output directory: {output_dir}")
    
    # ... training code ...
    
    # After training, update models.json
    model_info = {
        "name": model_name,
        "method": args.method,
        "weights_trained": "cross_attention",  # or whatever weights were trained
        "concept_erased": args.concept,
        "base_model": "CompVis/stable-diffusion-v1-4",  # or args.base_model
        "description": f"{args.method} model trained to erase {args.concept} concept",
        "timestamp": datetime.now().isoformat(),
        "training_args": {
            "learning_rate": args.learning_rate,
            "num_epochs": args.num_epochs,
            # ... other relevant training arguments
        }
    }
    
    update_models_json(next_index, model_info, models_data, models_json_path)
    
    # Also save metadata.json in the model directory
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(model_info, f, indent=2)
    
    print(f"Training complete! Model saved to {output_dir}")
    
    return next_index, output_dir

if __name__ == "__main__":
    # Example argument parsing
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--concept", type=str, required=True)
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--model_index", type=int, default=None, help="Override automatic index")
    
    args = parser.parse_args()
    train_model(args)