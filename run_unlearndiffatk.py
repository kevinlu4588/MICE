#!/usr/bin/env python3
"""
Script to generate config file and run unlearndiffatk attack on a given model.
"""

import argparse
import json
import os
import subprocess
from pathlib import Path
from datetime import datetime


def generate_config(
    model_path,
    concept,
    output_dir,
    dataset_path=None,
    iterations=40,
    attack_idx=0,
    method="esdx",
    attacker="text_grad",
    k=3,
    lr=0.01,
    weight_decay=0.1
):
    """Generate a config file for unlearndiffatk attack."""
    
    # Default dataset path if not provided
    if dataset_path is None:
        dataset_path = f"/share/u/kevin/erasing/Diffusion-MU-Attack/files/dataset/{concept}"
    
    # Create timestamp for unique naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    config = {
        "overall": {
            "task": "classifier",
            "attacker": attacker,
            "logger": "json",
            "resume": None
        },
        "task": {
            "concept": concept,
            "model_name_or_path": "CompVis/stable-diffusion-v1-4",
            "target_ckpt": model_path,
            "cache_path": ".cache",
            "dataset_path": dataset_path,
            "criterion": "l2",
            "classifier_dir": "/share/u/kevin/Diffusion-MU-Attack/files/checkpoint-2800"
        },
        "attacker": {
            "insertion_location": "prefix_k",
            "k": k,
            "iteration": iterations,
            "attack_idx": attack_idx,
            "eval_seed": 0,
            "universal": False,
            "sequential": True,
            "text_grad": {
                "lr": lr,
                "weight_decay": weight_decay
            }
        },
        "logger": {
            "json": {
                "root": output_dir
            }
        }
    }
    
    # Create config directory if it doesn't exist
    # Use absolute path in Diffusion-MU-Attack directory
    attack_base_dir = Path("/share/u/kevin/erasing/Diffusion-MU-Attack")
    config_dir = attack_base_dir / "configs" / "generated"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config file
    config_filename = f"config_{method}_{concept}_{timestamp}.json"
    config_path = config_dir / config_filename
    
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    
    return str(config_path)


def run_attack(config_path, logger_name=None):
    """Run the unlearndiffatk attack with the given config."""
    
    # Change to the Diffusion-MU-Attack directory
    attack_dir = Path("/share/u/kevin/erasing/Diffusion-MU-Attack")
    if not attack_dir.exists():
        raise FileNotFoundError(f"Attack directory not found: {attack_dir}")
    
    # Build command
    cmd = [
        "python", "src/execs/attack.py",
        "--config-file", config_path
    ]
    
    if logger_name:
        cmd.extend(["--logger.name", logger_name])
    
    # Run the attack
    print(f"Running command: {' '.join(cmd)}")
    print(f"Working directory: {attack_dir}")
    
    # Run without capturing output so errors are visible
    result = subprocess.run(
        cmd,
        cwd=str(attack_dir),
        check=True
    )
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate config and run unlearndiffatk attack on a model"
    )
    
    # Required arguments
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the target model checkpoint (.safetensors file)"
    )
    parser.add_argument(
        "--concept",
        type=str,
        required=True,
        help="Concept to attack (e.g., van_gogh, english_springer_spaniel)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: ./results/<method>_<concept>_<timestamp>)"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to dataset (default: /share/u/kevin/Diffusion-MU-Attack/data/200_I2P_objects/<concept>)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=40,
        help="Number of attack iterations (default: 40)"
    )
    parser.add_argument(
        "--attack-idx",
        type=int,
        default=0,
        help="Attack index for multiple runs (default: 0)"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="esdx",
        help="Method name for logging (default: esdx)"
    )
    parser.add_argument(
        "--attacker",
        type=str,
        default="text_grad",
        choices=["text_grad", "gcg", "hard_prompt", "hard_prompt_multi", "random", "seed_search", "no_attack"],
        help="Attacker type (default: text_grad)"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Number of tokens to insert (default: 3)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Learning rate for text_grad attacker (default: 0.01)"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.1,
        help="Weight decay for text_grad attacker (default: 0.1)"
    )
    parser.add_argument(
        "--config-only",
        action="store_true",
        help="Only generate config file without running attack"
    )
    parser.add_argument(
        "--logger-name",
        type=str,
        default=None,
        help="Custom logger name (default: <method>_<concept>_attack_idx_<attack_idx>)"
    )
    
    args = parser.parse_args()
    
    # Validate model path
    if not Path(args.model_path).exists():
        print(f"Error: Model path does not exist: {args.model_path}")
        return 1
    
    # Generate output directory if not specified
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Use absolute path for results
        args.output_dir = f"/share/u/kevin/erasing/Diffusion-MU-Attack/results/{args.method}_{args.concept}_{timestamp}"
    
    # Generate logger name if not specified
    if args.logger_name is None:
        args.logger_name = f"{args.method}_{args.concept}_attack_idx_{args.attack_idx}"
    
    # Generate config file
    print(f"Generating config file for {args.concept} attack on {args.model_path}")
    config_path = generate_config(
        model_path=args.model_path,
        concept=args.concept,
        output_dir=args.output_dir,
        dataset_path=args.dataset_path,
        iterations=args.iterations,
        attack_idx=args.attack_idx,
        method=args.method,
        attacker=args.attacker,
        k=args.k,
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    print(f"Config file generated: {config_path}")
    
    if args.config_only:
        print("Config generation complete. Skipping attack execution.")
        return 0
    
    # Run the attack
    print(f"\nRunning attack with config: {config_path}")
    success = run_attack(config_path, args.logger_name)
    
    if success:
        print(f"\nAttack completed successfully!")
        print(f"Results saved to: {args.output_dir}")
        return 0
    else:
        print(f"\nAttack failed!")
        return 1


if __name__ == "__main__":
    exit(main())