#!/usr/bin/env python3
"""
Gradient Ascent Concept Erasure

This script implements aggressive concept erasure using gradient ascent:
- MAXIMIZE loss on Van Gogh prompts (push model away from concept)
- MINIMIZE loss on neutral prompts (preserve general capabilities)

This approach is more direct and aggressive than ESD's indirect target formulation.
"""

import os
import torch
import random
import numpy as np
from tqdm.auto import tqdm
from safetensors.torch import save_file
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DDIMScheduler, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
import json
import time
from pathlib import Path
import glob

class GradientAscentEraser:
    def __init__(self, device="cuda:0", torch_dtype=torch.bfloat16):
        self.device = device
        self.torch_dtype = torch_dtype
        
        print("Loading SD1.4 for gradient ascent concept erasure...")
        
        # Load standard SD1.4 (not inpainting)
        model_id = "CompVis/stable-diffusion-v1-4"
        
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
        self.scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
        
        # Base UNet (frozen for comparison) and trainable UNet
        self.base_unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
        self.trainable_unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
        
        # Move to device
        self.text_encoder = self.text_encoder.to(self.device, self.torch_dtype)
        self.vae = self.vae.to(self.device, self.torch_dtype)
        self.base_unet = self.base_unet.to(self.device, self.torch_dtype)
        self.trainable_unet = self.trainable_unet.to(self.device, self.torch_dtype)
        
        # Set modes
        self.text_encoder.eval()
        self.vae.eval()
        self.base_unet.eval()
        self.trainable_unet.train()
        
        # Freeze base model
        self.base_unet.requires_grad_(False)
        
        # Get trainable parameters (cross-attention layers only)
        self.trainable_params = []
        for name, module in self.trainable_unet.named_modules():
            if module.__class__.__name__ in ["Linear", "Conv2d", "LoRACompatibleLinear", "LoRACompatibleConv"]:
                if 'attn2' in name:  # Cross-attention layers
                    for n, p in module.named_parameters():
                        self.trainable_params.append(p)
        
        print(f"Found {len(self.trainable_params)} trainable parameters")
        
        # Freeze non-trainable parameters
        for param in self.trainable_unet.parameters():
            param.requires_grad = False
        
        for param in self.trainable_params:
            param.requires_grad = True
        
        self.criterion = torch.nn.MSELoss()
        
        print("Gradient Ascent Eraser initialized!")
    
    def get_van_gogh_prompts(self):
        """Comprehensive Van Gogh prompt set for aggressive erasure."""
        return [
            # Famous paintings
            "The Starry Night by Vincent van Gogh",
            "Sunflowers by Vincent van Gogh", 
            "The Potato Eaters by Vincent van Gogh",
            "Café Terrace at Night by Vincent van Gogh",
            "Irises by Vincent van Gogh",
            "The Bedroom by Vincent van Gogh",
            "Wheatfield with Crows by Vincent van Gogh",
            "Self-Portrait by Vincent van Gogh",
            "Almond Blossoms by Vincent van Gogh",
            "The Mulberry Tree by Vincent van Gogh",
            
            # Style descriptors
            "a painting by Vincent van Gogh",
            "Van Gogh style painting",
            "Van Gogh style artwork",
            "Vincent van Gogh masterpiece",
            "painting in the style of Van Gogh",
            "Van Gogh inspired art",
            "Van Gogh technique painting",
            
            # Technical style terms
            "thick impasto brushstrokes like Van Gogh",
            "swirling patterns in Van Gogh style",
            "post-impressionist painting by Van Gogh",
            "expressionist Van Gogh painting",
            "vivid colors like Van Gogh",
            "dynamic brushwork by Van Gogh",
            "textured paint like Van Gogh",
            
            # Color combinations (Van Gogh signature)
            "yellow and blue painting by Van Gogh",
            "vibrant yellows and deep blues by Van Gogh", 
            "swirling yellows and blues like Starry Night",
            "Van Gogh yellow sunflower painting",
            "Van Gogh blue and yellow night scene",
            
            # Subject matter + style
            "landscape by Vincent van Gogh",
            "portrait by Vincent van Gogh", 
            "still life by Vincent van Gogh",
            "flowers by Vincent van Gogh",
            "trees by Vincent van Gogh",
            "countryside by Vincent van Gogh",
            "night scene by Vincent van Gogh",
            "cafe scene by Vincent van Gogh",
            "garden by Vincent van Gogh",
            "wheatfield by Vincent van Gogh",
            
            # Specific elements
            "Van Gogh cypress trees",
            "Van Gogh sunflower field", 
            "Van Gogh starry night sky",
            "Van Gogh yellow house",
            "Van Gogh peasant workers",
            "Van Gogh café interior",
            "Van Gogh bedroom scene",
            "Van Gogh hospital garden",
            
            # Technique variations  
            "Van Gogh thick paint application",
            "Van Gogh visible brushstrokes",
            "Van Gogh swirling technique",
            "Van Gogh impasto method",
            "Van Gogh color theory",
            "Van Gogh emotional expression",
            "Van Gogh movement in paint",
            
            # Period references
            "Van Gogh Arles period painting",
            "Van Gogh Saint-Paul period", 
            "Van Gogh Paris period artwork",
            "Van Gogh final period painting",
            "late Van Gogh style",
            "early Van Gogh work",
            
            # Museum/famous references
            "Van Gogh Museum masterpiece",
            "famous Van Gogh painting",
            "iconic Van Gogh artwork",
            "Van Gogh exhibition piece", 
            "Van Gogh auction painting",
            "Van Gogh gallery artwork",
            
            # Emotional/expressive terms
            "passionate Van Gogh painting",
            "emotional Van Gogh artwork", 
            "intense Van Gogh colors",
            "dramatic Van Gogh scene",
            "melancholic Van Gogh style",
            "energetic Van Gogh brushwork",
            
            # Comparative terms
            "Van Gogh versus Monet style",
            "Van Gogh post-impressionism",
            "Van Gogh modern art",
            "Van Gogh Dutch art",
            "Van Gogh 19th century painting",
            
            # Additional specific works
            "The Yellow House by Van Gogh",
            "Olive Trees by Van Gogh",
            "The Night Café by Van Gogh",
            "Wheat Field series by Van Gogh",
            "Water Lilies by Van Gogh",
            "Dr. Gachet portrait by Van Gogh",
            "Postman Joseph Roulin by Van Gogh",
            
            # Abstract style references
            "Van Gogh brushstroke texture",
            "Van Gogh color harmony",
            "Van Gogh composition style",
            "Van Gogh artistic vision",
            "Van Gogh creative technique",
            "Van Gogh paint handling",
            "Van Gogh visual rhythm",
            "Van Gogh artistic expression"
        ]
    
    def get_neutral_prompts(self):
        """Neutral prompts to preserve general artistic capabilities."""
        return [
            # Generic art styles
            "a beautiful landscape painting",
            "a realistic portrait painting", 
            "an abstract modern artwork",
            "a minimalist composition",
            "a photorealistic painting",
            "a watercolor landscape",
            "an oil painting of flowers",
            "a classical still life",
            
            # Other famous artists (to preserve)
            "a painting by Claude Monet",
            "artwork by Pablo Picasso", 
            "a piece by Salvador Dali",
            "painting by Leonardo da Vinci",
            "artwork by Frida Kahlo",
            "a piece by Georgia O'Keeffe",
            "painting by Jackson Pollock",
            "artwork by Andy Warhol",
            
            # Contemporary styles
            "digital art illustration",
            "modern abstract painting",
            "contemporary artwork",
            "street art mural",
            "graphic design poster",
            "architectural photography",
            "nature photography",
            "portrait photography",
            
            # Subject matter (neutral style)
            "mountain landscape",
            "ocean sunset", 
            "forest scene",
            "urban cityscape",
            "flower garden",
            "animal portrait",
            "human figure",
            "architectural building",
            
            # Technical neutral terms
            "high resolution artwork",
            "detailed illustration",
            "professional photography",
            "studio lighting portrait",
            "macro photography",
            "wide angle landscape",
            "close-up detail shot",
            "artistic composition",
            
            # Non-art subjects
            "scientific diagram",
            "technical illustration", 
            "educational infographic",
            "medical illustration",
            "engineering blueprint",
            "map illustration",
            "data visualization",
            "product design render"
        ]
    
    def encode_text_embeddings(self, prompt):
        """Encode text to embeddings."""
        with torch.no_grad():
            text_inputs = self.tokenizer(
                prompt, return_tensors="pt", max_length=77, truncation=True, padding="max_length"
            ).input_ids.to(self.device)
            embeddings = self.text_encoder(text_inputs)[0]
        return embeddings
    
    def train_gradient_ascent(self, iterations=1000, lr=1e-4, ascent_weight=1.0, descent_weight=0.1):
        """
        Train using gradient ascent on Van Gogh prompts and descent on neutral prompts.
        
        Args:
            iterations: Number of training iterations
            lr: Learning rate 
            ascent_weight: Weight for gradient ascent loss (Van Gogh erasure)
            descent_weight: Weight for gradient descent loss (neutral preservation)
        """
        
        start_time = time.time()
        
        # Get prompt sets
        van_gogh_prompts = self.get_van_gogh_prompts()
        neutral_prompts = self.get_neutral_prompts()
        
        print(f"Starting gradient ascent training:")
        print(f"- {len(van_gogh_prompts)} Van Gogh prompts (ASCENT - maximize loss)")
        print(f"- {len(neutral_prompts)} neutral prompts (DESCENT - minimize loss)")
        print(f"- {iterations} iterations")
        print(f"- Learning rate: {lr}")
        print(f"- Ascent weight: {ascent_weight}")
        print(f"- Descent weight: {descent_weight}")
        
        # Create optimizer
        optimizer = torch.optim.AdamW(self.trainable_params, lr=lr)
        
        losses = {
            'total': [],
            'van_gogh_ascent': [],
            'neutral_descent': [],
            'van_gogh_raw': [],  # Raw loss values for monitoring
            'neutral_raw': []
        }
        
        for iteration in tqdm(range(iterations), desc="Training"):
            
            # === VAN GOGH GRADIENT ASCENT ===
            van_gogh_prompt = random.choice(van_gogh_prompts)
            
            # Random timestep and noise
            timestep = torch.randint(0, 1000, (1,), device=self.device)
            noise = torch.randn((1, 4, 64, 64), device=self.device, dtype=self.torch_dtype)
            
            # Create noisy latents 
            latents = torch.randn((1, 4, 64, 64), device=self.device, dtype=self.torch_dtype)
            noisy_latents = self.scheduler.add_noise(latents, noise, timestep)
            
            # Encode text
            van_gogh_embeddings = self.encode_text_embeddings(van_gogh_prompt)
            
            # Predict noise with trainable model
            predicted_noise = self.trainable_unet(
                noisy_latents, timestep, encoder_hidden_states=van_gogh_embeddings
            ).sample
            
            # Calculate loss (reconstruction error)
            van_gogh_loss = self.criterion(predicted_noise, noise)
            
            # GRADIENT ASCENT: Negate loss to maximize it
            van_gogh_ascent_loss = -ascent_weight * van_gogh_loss
            
            # === NEUTRAL GRADIENT DESCENT ===
            neutral_prompt = random.choice(neutral_prompts)
            
            # New random timestep and noise for neutral
            timestep_neutral = torch.randint(0, 1000, (1,), device=self.device)
            noise_neutral = torch.randn((1, 4, 64, 64), device=self.device, dtype=self.torch_dtype)
            latents_neutral = torch.randn((1, 4, 64, 64), device=self.device, dtype=self.torch_dtype)
            noisy_latents_neutral = self.scheduler.add_noise(latents_neutral, noise_neutral, timestep_neutral)
            
            # Encode neutral text
            neutral_embeddings = self.encode_text_embeddings(neutral_prompt)
            
            # Predict noise for neutral prompt
            predicted_noise_neutral = self.trainable_unet(
                noisy_latents_neutral, timestep_neutral, encoder_hidden_states=neutral_embeddings
            ).sample
            
            # Calculate neutral loss (standard descent)
            neutral_loss = self.criterion(predicted_noise_neutral, noise_neutral)
            neutral_descent_loss = descent_weight * neutral_loss
            
            # === COMBINED LOSS ===
            total_loss = van_gogh_ascent_loss + neutral_descent_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.trainable_params, max_norm=1.0)
            
            optimizer.step()
            
            # Record losses
            losses['total'].append(total_loss.item())
            losses['van_gogh_ascent'].append(van_gogh_ascent_loss.item()) 
            losses['neutral_descent'].append(neutral_descent_loss.item())
            losses['van_gogh_raw'].append(van_gogh_loss.item())  # Raw loss (should increase)
            losses['neutral_raw'].append(neutral_loss.item())    # Raw loss (should stay low)
            
            # Progress reporting
            if iteration % 50 == 0:
                elapsed_time = time.time() - start_time
                remaining_iterations = iterations - iteration
                estimated_remaining_time = (elapsed_time / (iteration + 1)) * remaining_iterations
                
                # Calculate recent averages
                recent_van_gogh_raw = sum(losses['van_gogh_raw'][-10:]) / min(10, len(losses['van_gogh_raw']))
                recent_neutral_raw = sum(losses['neutral_raw'][-10:]) / min(10, len(losses['neutral_raw']))
                recent_total = sum(losses['total'][-10:]) / min(10, len(losses['total']))
                
                print(f"\nIteration {iteration}/{iterations}")
                print(f"Training Loss: {total_loss.item():.6f} (avg: {recent_total:.6f})")
                print(f"Van Gogh RAW loss: {van_gogh_loss.item():.6f} (avg: {recent_van_gogh_raw:.6f}) [WANT HIGH]")
                print(f"Neutral RAW loss: {neutral_loss.item():.6f} (avg: {recent_neutral_raw:.6f}) [WANT LOW]")
                print(f"Ascent component: {van_gogh_ascent_loss.item():.6f}")
                print(f"Descent component: {neutral_descent_loss.item():.6f}")
                print(f"Elapsed: {elapsed_time/60:.1f}min, Est. remaining: {estimated_remaining_time/60:.1f}min")
                print(f"Van Gogh prompt: {van_gogh_prompt[:60]}...")
                print(f"Neutral prompt: {neutral_prompt[:60]}...")
        
        total_time = time.time() - start_time
        
        # Final statistics
        final_van_gogh_raw = sum(losses['van_gogh_raw'][-10:]) / min(10, len(losses['van_gogh_raw']))
        final_neutral_raw = sum(losses['neutral_raw'][-10:]) / min(10, len(losses['neutral_raw']))
        initial_van_gogh_raw = sum(losses['van_gogh_raw'][:10]) / min(10, len(losses['van_gogh_raw']))
        initial_neutral_raw = sum(losses['neutral_raw'][:10]) / min(10, len(losses['neutral_raw']))
        
        print(f"\nGradient Ascent Training Completed!")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Van Gogh loss change: {initial_van_gogh_raw:.6f} → {final_van_gogh_raw:.6f} (Δ: {final_van_gogh_raw - initial_van_gogh_raw:+.6f})")
        print(f"Neutral loss change: {initial_neutral_raw:.6f} → {final_neutral_raw:.6f} (Δ: {final_neutral_raw - initial_neutral_raw:+.6f})")
        print(f"SUCCESS METRICS:")
        print(f"  - Van Gogh loss INCREASED by {((final_van_gogh_raw/initial_van_gogh_raw - 1) * 100):+.1f}%")
        print(f"  - Neutral loss changed by {((final_neutral_raw/initial_neutral_raw - 1) * 100):+.1f}%")
        
        return losses
    
    def get_next_model_number(self, base_dir="models"):
        """Get the next available model number for versioning."""
        Path(base_dir).mkdir(exist_ok=True)
        
        # Find existing numbered model directories
        existing_models = glob.glob(os.path.join(base_dir, "*_*"))
        numbers = []
        
        for model_path in existing_models:
            model_name = os.path.basename(model_path)
            if "_" in model_name:
                try:
                    number = int(model_name.split("_")[0])
                    numbers.append(number)
                except ValueError:
                    continue
        
        # Return next available number
        return max(numbers) + 1 if numbers else 0
    
    def save_model(self, method_name="gradient_ascent", base_dir="models"):
        """Save the trained model with numbered versioning."""
        model_number = self.get_next_model_number(base_dir)
        output_dir = os.path.join(base_dir, f"{model_number}_{method_name}")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save only the trainable parameters
        state_dict = {}
        for name, param in self.trainable_unet.named_parameters():
            if param.requires_grad:
                state_dict[name] = param.data
        
        model_path = os.path.join(output_dir, "model.safetensors")
        save_file(state_dict, model_path)
        
        # Save metadata
        metadata = {
            "model_number": model_number,
            "method_name": method_name,
            "model_type": "gradient_ascent_vangogh",
            "architecture": "stable-diffusion-v1-4", 
            "method": "gradient_ascent",
            "concept": "Van Gogh",
            "trainable_params": len(state_dict),
            "training_method": "dual_objective_ascent_descent",
            "target_layers": "cross_attention"
        }
        
        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Model #{model_number} saved to {output_dir}")
        print(f"Model path: {model_path}")
        print(f"Trainable parameters: {len(state_dict)}")
        
        return output_dir

def main():
    print("="*60)
    print("GRADIENT ASCENT CONCEPT ERASURE")
    print("="*60)
    print("Aggressive Van Gogh concept removal using direct gradient ascent")
    print("- MAXIMIZE loss on Van Gogh prompts (push away from concept)")
    print("- MINIMIZE loss on neutral prompts (preserve capabilities)")
    print("="*60)
    
    # Initialize eraser
    eraser = GradientAscentEraser()
    
    print("\nStarting gradient ascent training...")
    losses = eraser.train_gradient_ascent(
        iterations=1000,
        lr=1e-4,
        ascent_weight=1.0,
        descent_weight=0.1
    )
    
    print("\nSaving model...")
    model_dir = eraser.save_model("gradient_ascent")
    
    print(f"\n" + "="*60)
    print("GRADIENT ASCENT ERASURE COMPLETED!")
    print("="*60)
    print(f"Model saved to: {model_dir}")
    print(f"Van Gogh concept should be significantly degraded")
    print(f"Neutral capabilities should be preserved")
    print("="*60)

if __name__ == "__main__":
    main()