#!/usr/bin/env python3
"""
Quick SD1.4 Van Gogh Erasure Training

This script trains a Van Gogh concept erasure model directly on SD1.4 architecture
for compatibility with the simple evaluation script.
"""

import os
import torch
from tqdm.auto import tqdm
from safetensors.torch import save_file
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from PIL import Image
import json

class SD14VanGoghEraser:
    def __init__(self, device="cuda:0", torch_dtype=torch.bfloat16):
        self.device = device
        self.torch_dtype = torch_dtype
        
        print("Loading SD1.4 for Van Gogh erasure training...")
        
        # Load base SD1.4 pipeline
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            torch_dtype=self.torch_dtype,
            use_safetensors=True
        ).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        
        # ESD UNet for training
        self.esd_unet = UNet2DConditionModel.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            subfolder="unet"
        ).to(self.device, self.torch_dtype)
        
        # Get trainable parameters (cross-attention layers only)
        self.trainable_params = []
        for name, module in self.esd_unet.named_modules():
            if module.__class__.__name__ in ["Linear", "Conv2d", "LoRACompatibleLinear", "LoRACompatibleConv"]:
                if 'attn2' in name:
                    for n, p in module.named_parameters():
                        self.trainable_params.append(p)
        
        print(f"Found {len(self.trainable_params)} trainable parameters")
        
        # Freeze non-trainable parameters
        for param in self.esd_unet.parameters():
            param.requires_grad = False
        
        for param in self.trainable_params:
            param.requires_grad = True
        
        self.criterion = torch.nn.MSELoss()
        
    def train_erasure(self, iterations=20, lr=1e-5, negative_guidance=2.0):
        """Train Van Gogh concept erasure."""
        
        # Test prompts
        vangogh_prompts = [
            "a painting by Vincent van Gogh",
            "The Starry Night by Vincent van Gogh",
            "sunflowers by Vincent van Gogh",
            "a landscape by Vincent van Gogh",
            "Van Gogh style painting"
        ]
        
        neutral_prompt = "a landscape painting"
        
        # Create optimizer
        optimizer = torch.optim.AdamW(self.trainable_params, lr=lr)
        
        print(f"Starting training: {iterations} iterations, lr={lr}")
        
        losses = []
        
        for iteration in tqdm(range(iterations)):
            # Random prompt
            prompt = vangogh_prompts[iteration % len(vangogh_prompts)]
            
            # Encode prompts
            with torch.no_grad():
                # Van Gogh prompt
                tokens = self.pipe.tokenizer(
                    prompt, return_tensors="pt", max_length=77, truncation=True, padding="max_length"
                ).input_ids.to(self.device)
                vangogh_embeddings = self.pipe.text_encoder(tokens)[0]
                
                # Neutral prompt
                tokens = self.pipe.tokenizer(
                    neutral_prompt, return_tensors="pt", max_length=77, truncation=True, padding="max_length"
                ).input_ids.to(self.device)
                neutral_embeddings = self.pipe.text_encoder(tokens)[0]
                
                # Empty prompt
                tokens = self.pipe.tokenizer(
                    "", return_tensors="pt", max_length=77, truncation=True, padding="max_length"
                ).input_ids.to(self.device)
                null_embeddings = self.pipe.text_encoder(tokens)[0]
                
                # Random noise
                noise = torch.randn(1, 4, 64, 64, device=self.device, dtype=self.torch_dtype)
                timesteps = torch.randint(0, 1000, (1,), device=self.device)
                
                # Base model predictions
                base_noise_pred_vangogh = self.pipe.unet(
                    noise, timesteps, encoder_hidden_states=vangogh_embeddings
                ).sample
                
                base_noise_pred_neutral = self.pipe.unet(
                    noise, timesteps, encoder_hidden_states=neutral_embeddings
                ).sample
                
                base_noise_pred_null = self.pipe.unet(
                    noise, timesteps, encoder_hidden_states=null_embeddings
                ).sample
            
            # ESD model predictions
            esd_noise_pred_vangogh = self.esd_unet(
                noise, timesteps, encoder_hidden_states=vangogh_embeddings
            ).sample
            
            esd_noise_pred_null = self.esd_unet(
                noise, timesteps, encoder_hidden_states=null_embeddings
            ).sample
            
            # ESD loss: target = neutral - negative_guidance * (vangogh - null)
            target_noise = base_noise_pred_neutral - negative_guidance * (base_noise_pred_vangogh - base_noise_pred_null)
            
            # Loss
            loss = self.criterion(esd_noise_pred_vangogh, target_noise)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            if iteration % 5 == 0:
                print(f"Iteration {iteration}, Loss: {loss.item():.6f}")
        
        print(f"Training completed! Final loss: {losses[-1]:.6f}")
        return losses
    
    def save_model(self, output_dir="sd14_vangogh_erasure_model"):
        """Save the trained model."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save only the trainable parameters
        state_dict = {}
        for name, param in self.esd_unet.named_parameters():
            if param.requires_grad:
                state_dict[name] = param.data
        
        model_path = os.path.join(output_dir, "sd14_vangogh_erasure.safetensors")
        save_file(state_dict, model_path)
        
        # Save metadata
        metadata = {
            "model_type": "sd14_vangogh_erasure",
            "architecture": "stable-diffusion-v1-4",
            "method": "esd-x",
            "concept": "Van Gogh",
            "trainable_params": len(state_dict)
        }
        
        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Model saved to {model_path}")
        print(f"Trainable parameters: {len(state_dict)}")
        
        return model_path

def main():
    # Quick training
    eraser = SD14VanGoghEraser()
    
    print("Training Van Gogh erasure model...")
    losses = eraser.train_erasure(iterations=10, lr=1e-5, negative_guidance=2.0)
    
    print("Saving model...")
    model_path = eraser.save_model()
    
    print(f"Training completed! Model saved to: {model_path}")

if __name__ == "__main__":
    main()