#!/usr/bin/env python3
"""
Weight Change Analysis for Van Gogh Erasure Models

This script analyzes the differences in weights between original and erased UNet models
to understand what the erasure training actually modified.
"""

import torch
import numpy as np
from safetensors.torch import load_file
from diffusers import UNet2DConditionModel
import matplotlib.pyplot as plt
from pathlib import Path
import json

class WeightChangeAnalyzer:
    def __init__(self, device="cuda:0", torch_dtype=torch.bfloat16):
        self.device = device
        self.torch_dtype = torch_dtype
        
    def load_models(self, erased_model_path):
        """Load original and erased models for comparison."""
        print("Loading original SD1.4 UNet...")
        self.original_unet = UNet2DConditionModel.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            subfolder="unet"
        ).to(self.device, self.torch_dtype)
        
        print("Loading erased UNet...")
        self.erased_unet = UNet2DConditionModel.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            subfolder="unet"
        ).to(self.device, self.torch_dtype)
        
        # Load erased parameters
        esd_params = load_file(erased_model_path)
        missing_keys, unexpected_keys = self.erased_unet.load_state_dict(esd_params, strict=False)
        
        print(f"Loaded {len(esd_params)} erased parameters")
        print(f"Missing keys: {len(missing_keys)}")
        print(f"Unexpected keys: {len(unexpected_keys)}")
        
        return esd_params
    
    def analyze_weight_changes(self, esd_params, output_dir="weight_analysis"):
        """Analyze the weight changes between original and erased models."""
        Path(output_dir).mkdir(exist_ok=True)
        
        print("Analyzing weight changes...")
        
        changes = {}
        statistics = {
            "total_params_changed": 0,
            "total_params_analyzed": 0,
            "avg_absolute_change": 0.0,
            "max_absolute_change": 0.0,
            "min_absolute_change": float('inf'),
            "std_change": 0.0,
            "layer_analysis": {}
        }
        
        all_changes = []
        
        # Compare each parameter
        for name, erased_param in esd_params.items():
            if name in dict(self.original_unet.named_parameters()):
                original_param = dict(self.original_unet.named_parameters())[name]
                
                # Calculate difference
                diff = erased_param.cpu() - original_param.cpu()
                abs_diff = torch.abs(diff)
                
                # Store statistics
                changes[name] = {
                    "shape": list(diff.shape),
                    "mean_abs_change": abs_diff.mean().item(),
                    "max_abs_change": abs_diff.max().item(),
                    "min_abs_change": abs_diff.min().item(),
                    "std_change": diff.std().item(),
                    "num_params": diff.numel()
                }
                
                all_changes.extend(abs_diff.flatten().tolist())
                statistics["total_params_changed"] += diff.numel()
                
                # Layer-level analysis
                layer_type = self._get_layer_type(name)
                if layer_type not in statistics["layer_analysis"]:
                    statistics["layer_analysis"][layer_type] = {
                        "count": 0,
                        "total_params": 0,
                        "avg_change": 0.0,
                        "max_change": 0.0
                    }
                
                layer_stats = statistics["layer_analysis"][layer_type]
                layer_stats["count"] += 1
                layer_stats["total_params"] += diff.numel()
                layer_stats["avg_change"] += abs_diff.mean().item()
                layer_stats["max_change"] = max(layer_stats["max_change"], abs_diff.max().item())
        
        # Calculate overall statistics
        all_changes = torch.tensor(all_changes)
        statistics["total_params_analyzed"] = len(all_changes)
        statistics["avg_absolute_change"] = all_changes.mean().item()
        statistics["max_absolute_change"] = all_changes.max().item()
        statistics["min_absolute_change"] = all_changes.min().item()
        statistics["std_change"] = all_changes.std().item()
        
        # Finalize layer statistics
        for layer_type, layer_stats in statistics["layer_analysis"].items():
            if layer_stats["count"] > 0:
                layer_stats["avg_change"] /= layer_stats["count"]
        
        # Save detailed analysis
        with open(f"{output_dir}/weight_changes_detailed.json", "w") as f:
            json.dump(changes, f, indent=2)
        
        with open(f"{output_dir}/weight_changes_summary.json", "w") as f:
            json.dump(statistics, f, indent=2)
        
        return changes, statistics
    
    def _get_layer_type(self, param_name):
        """Categorize parameter by layer type."""
        if "attn2.to_k" in param_name:
            return "cross_attention_key"
        elif "attn2.to_v" in param_name:
            return "cross_attention_value"
        elif "attn2.to_q" in param_name:
            return "cross_attention_query"
        elif "attn2.to_out" in param_name:
            return "cross_attention_output"
        elif "attn1" in param_name:
            return "self_attention"
        elif "conv" in param_name:
            return "convolution"
        elif "linear" in param_name:
            return "linear"
        else:
            return "other"
    
    def create_visualizations(self, changes, statistics, output_dir="weight_analysis"):
        """Create visualizations of weight changes."""
        print("Creating visualizations...")
        
        # 1. Distribution of weight changes
        all_changes = []
        param_names = []
        for name, change_info in changes.items():
            all_changes.extend([change_info["mean_abs_change"]] * change_info["num_params"])
            param_names.append(name)
        
        plt.figure(figsize=(12, 8))
        
        # Histogram of changes
        plt.subplot(2, 2, 1)
        plt.hist(all_changes, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Absolute Weight Change')
        plt.ylabel('Frequency')
        plt.title('Distribution of Weight Changes')
        plt.yscale('log')
        
        # Layer-wise changes
        plt.subplot(2, 2, 2)
        layer_types = list(statistics["layer_analysis"].keys())
        layer_changes = [statistics["layer_analysis"][lt]["avg_change"] for lt in layer_types]
        
        plt.bar(range(len(layer_types)), layer_changes)
        plt.xlabel('Layer Type')
        plt.ylabel('Average Absolute Change')
        plt.title('Weight Changes by Layer Type')
        plt.xticks(range(len(layer_types)), layer_types, rotation=45)
        
        # Top changed parameters
        plt.subplot(2, 2, 3)
        sorted_changes = sorted(changes.items(), key=lambda x: x[1]["mean_abs_change"], reverse=True)
        top_10 = sorted_changes[:10]
        
        names = [name.split('.')[-2] + '.' + name.split('.')[-1] for name, _ in top_10]
        values = [change["mean_abs_change"] for _, change in top_10]
        
        plt.barh(range(len(names)), values)
        plt.yticks(range(len(names)), names)
        plt.xlabel('Mean Absolute Change')
        plt.title('Top 10 Most Changed Parameters')
        
        # Parameter count by layer
        plt.subplot(2, 2, 4)
        layer_params = [statistics["layer_analysis"][lt]["total_params"] for lt in layer_types]
        plt.pie(layer_params, labels=layer_types, autopct='%1.1f%%')
        plt.title('Parameter Distribution by Layer Type')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/weight_changes_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to {output_dir}/weight_changes_analysis.png")
    
    def print_summary(self, statistics):
        """Print a summary of the weight changes."""
        print("\n" + "="*60)
        print("WEIGHT CHANGE ANALYSIS SUMMARY")
        print("="*60)
        print(f"Total parameters analyzed: {statistics['total_params_analyzed']:,}")
        print(f"Total parameters changed: {statistics['total_params_changed']:,}")
        print(f"Average absolute change: {statistics['avg_absolute_change']:.8f}")
        print(f"Maximum absolute change: {statistics['max_absolute_change']:.8f}")
        print(f"Minimum absolute change: {statistics['min_absolute_change']:.8f}")
        print(f"Standard deviation: {statistics['std_change']:.8f}")
        
        print("\nLAYER-WISE ANALYSIS:")
        print("-" * 40)
        for layer_type, layer_stats in statistics["layer_analysis"].items():
            print(f"{layer_type}:")
            print(f"  Count: {layer_stats['count']}")
            print(f"  Total params: {layer_stats['total_params']:,}")
            print(f"  Avg change: {layer_stats['avg_change']:.8f}")
            print(f"  Max change: {layer_stats['max_change']:.8f}")
        print("="*60)
    
    def analyze_model(self, erased_model_path, output_dir="weight_analysis"):
        """Complete analysis of weight changes."""
        esd_params = self.load_models(erased_model_path)
        changes, statistics = self.analyze_weight_changes(esd_params, output_dir)
        self.create_visualizations(changes, statistics, output_dir)
        self.print_summary(statistics)
        
        return changes, statistics

def main():
    analyzer = WeightChangeAnalyzer()
    
    # Analyze the scaled grid model
    print("Analyzing weight changes for scaled grid Van Gogh erasure model...")
    changes, stats = analyzer.analyze_model(
        "scaled_grid_vangogh_model/scaled_grid_vangogh_erasure.safetensors",
        "weight_analysis_scaled_grid"
    )
    
    print("\nAnalysis complete! Check the weight_analysis_scaled_grid directory for results.")

if __name__ == "__main__":
    main()