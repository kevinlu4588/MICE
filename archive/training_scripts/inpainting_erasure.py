#!/usr/bin/env python3
"""
Baseline Inpainting Erasure Method for Concept Removal

This script integrates concept erasure with inpainting by training a UNet to avoid 
generating specific concept (e.g., "Van Gogh") content in masked regions while 
preserving the original content in non-masked areas.

Key innovations:
1. Multi-mask training strategy for robustness
2. Inpainting-aware ESD loss with neutral target guidance
3. Masked loss application for localized concept removal
4. Context preservation outside masked regions
"""

import os
import torch
import pandas as pd
import random
import numpy as np
from tqdm.auto import tqdm
from safetensors.torch import save_file
from diffusers import StableDiffusionInpaintPipeline, StableDiffusionPipeline, UNet2DConditionModel, DDIMScheduler
from diffusers.image_processor import VaeImageProcessor
from PIL import Image, ImageDraw
import argparse
import json
from pathlib import Path

class InpaintingEraser:
    def __init__(
        self, 
        model_id="stabilityai/stable-diffusion-2-inpainting",
        device="cuda:0",
        torch_dtype=torch.bfloat16
    ):
        """Initialize the inpainting erasure system."""
        self.model_id = model_id
        self.device = device
        self.torch_dtype = torch_dtype
        
        print(f"Loading models from: {model_id}")
        print(f"Device: {device}, dtype: {torch_dtype}")
        
        # Load dual model setup
        self.load_models()
        
        # Initialize image processors
        self.vae_scale_factor = 2 ** (len(self.pipe.vae.config.block_out_channels) - 1)
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor,
            do_normalize=False,
            do_binarize=True, 
            do_convert_grayscale=True
        )
        
        # Loss criterion
        self.criterion = torch.nn.MSELoss()
        
        print("Inpainting Eraser initialized successfully!")
    
    def load_models(self):
        """Load base and ESD models."""
        # Base UNet (frozen, for reference predictions)
        self.base_unet = UNet2DConditionModel.from_pretrained(
            self.model_id, subfolder="unet"
        ).to(self.device, self.torch_dtype)
        self.base_unet.requires_grad_(False)
        
        # ESD UNet (trainable, for concept erasure)
        self.esd_unet = UNet2DConditionModel.from_pretrained(
            self.model_id, subfolder="unet"
        ).to(self.device, self.torch_dtype)
        
        # Pipeline for inpainting
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            self.model_id,
            unet=self.base_unet,
            torch_dtype=self.torch_dtype,
            use_safetensors=True
        ).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        
        # Regular text-to-image pipeline for generating training images
        self.text2img_pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",  # Use SD1.4 for generating base images
            torch_dtype=self.torch_dtype,
            use_safetensors=True
        ).to(self.device)
        self.text2img_pipe.set_progress_bar_config(disable=True)
    
    def get_trainable_parameters(self, train_method='esd-x'):
        """Get trainable parameters from ESD UNet based on training method."""
        esd_params = []
        esd_param_names = []
        
        for name, module in self.esd_unet.named_modules():
            if module.__class__.__name__ in ["Linear", "Conv2d", "LoRACompatibleLinear", "LoRACompatibleConv"]:
                if train_method == 'esd-x' and 'attn2' in name:
                    for n, p in module.named_parameters():
                        esd_param_names.append(name + '.' + n)
                        esd_params.append(p)
                elif train_method == 'esd-u' and ('attn2' not in name):
                    for n, p in module.named_parameters():
                        esd_param_names.append(name + '.' + n)
                        esd_params.append(p)
                elif train_method == 'esd-all':
                    for n, p in module.named_parameters():
                        esd_param_names.append(name + '.' + n)
                        esd_params.append(p)
                elif train_method == 'esd-x-strict' and ('attn2.to_k' in name or 'attn2.to_v' in name):
                    for n, p in module.named_parameters():
                        esd_param_names.append(name + '.' + n)
                        esd_params.append(p)
        
        return esd_param_names, esd_params
    
    def create_mask(self, image_size, mask_type="center_square"):
        """Create different types of masks for training diversity."""
        width, height = image_size
        mask = Image.new("RGB", (width, height), "black")
        draw = ImageDraw.Draw(mask)
        
        if mask_type == "center_square":
            # Central square covering 33% of image
            square_size = min(width, height) // 3
            x1 = (width - square_size) // 2
            y1 = (height - square_size) // 2
            x2 = x1 + square_size
            y2 = y1 + square_size
            draw.rectangle([x1, y1, x2, y2], fill="white")
        
        elif mask_type == "circle":
            # Central circle
            radius = min(width, height) // 4
            center_x, center_y = width // 2, height // 2
            draw.ellipse([center_x - radius, center_y - radius, 
                         center_x + radius, center_y + radius], fill="white")
        
        elif mask_type == "car_area":
            # Lower-center rectangular region
            car_width = int(width * 0.4)
            car_height = int(height * 0.3)
            x1 = (width - car_width) // 2
            y1 = (height - car_height) // 2
            x2 = x1 + car_width
            y2 = y1 + car_height
            draw.rectangle([x1, y1, x2, y2], fill="white")
        
        return mask
    
    def load_prompts(self, prompt_file):
        """Load Van Gogh prompts from CSV file."""
        df = pd.read_csv(prompt_file)
        return df['prompt'].tolist()
    
    def generate_training_images(self, prompts, num_images=100, output_dir="training_images"):
        """Generate base images for training using Van Gogh prompts."""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Generating {num_images} training images...")
        
        generated_images = []
        used_prompts = []
        
        for i in tqdm(range(num_images), desc="Generating images"):
            # Select random prompt
            prompt = random.choice(prompts)
            
            # Generate image using text-to-image pipeline
            with torch.no_grad():
                result = self.text2img_pipe(
                    prompt=prompt,
                    num_inference_steps=50,
                    guidance_scale=7.5,
                    height=512,
                    width=512,
                    generator=torch.Generator().manual_seed(i)
                )
                image = result.images[0]
            
            # Save image and metadata
            image_path = os.path.join(output_dir, f"image_{i:03d}.png")
            image.save(image_path)
            
            generated_images.append({
                'image_path': image_path,
                'prompt': prompt,
                'seed': i
            })
            used_prompts.append(prompt)
        
        # Save metadata
        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(generated_images, f, indent=2)
        
        print(f"Generated {len(generated_images)} images saved to {output_dir}")
        return generated_images
    
    def prepare_training_data(self, generated_images, mask_types=["center_square", "circle", "car_area"]):
        """Prepare training data with multiple masks per image."""
        training_samples = []
        
        for img_data in generated_images:
            image = Image.open(img_data['image_path']).convert("RGB")
            
            for mask_type in mask_types:
                mask = self.create_mask(image.size, mask_type)
                
                training_samples.append({
                    'image': image,
                    'mask': mask,
                    'prompt': img_data['prompt'],
                    'mask_type': mask_type,
                    'original_seed': img_data['seed']
                })
        
        print(f"Prepared {len(training_samples)} training samples from {len(generated_images)} base images")
        return training_samples
    
    def get_noise_predictions(self, image, mask, prompt, neutral_prompt, timestep):
        """Get noise predictions for ESD loss computation."""
        # Process image and mask
        init_image = self.pipe.image_processor.preprocess(image, height=512, width=512)
        init_image = init_image.to(dtype=self.torch_dtype, device=self.device)
        
        mask_condition = self.mask_processor.preprocess(mask, height=512, width=512)
        mask_condition = mask_condition.to(dtype=self.torch_dtype, device=self.device)
        
        # Create masked image
        masked_image = init_image * (mask_condition < 0.5)
        
        # Encode to latents
        with torch.no_grad():
            image_latents = self.pipe.vae.encode(init_image).latent_dist.sample()
            image_latents = image_latents * self.pipe.vae.config.scaling_factor
            
            masked_image_latents = self.pipe.vae.encode(masked_image).latent_dist.sample()
            masked_image_latents = masked_image_latents * self.pipe.vae.config.scaling_factor
        
        # Prepare mask for latent space
        mask_latents = torch.nn.functional.interpolate(
            mask_condition, size=(image_latents.shape[2], image_latents.shape[3])
        )
        
        # Add noise to get noisy latents at specified timestep
        noise = torch.randn_like(image_latents)
        noisy_latents = self.pipe.scheduler.add_noise(image_latents, noise, timestep.to(self.device))
        
        # Get text embeddings
        with torch.no_grad():
            # Van Gogh prompt embeddings
            erase_embeds, null_embeds = self.pipe.encode_prompt(
                prompt=prompt,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=''
            )
            
            # Neutral prompt embeddings
            neutral_embeds, _ = self.pipe.encode_prompt(
                prompt=neutral_prompt,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
                negative_prompt=''
            )
        
        # Prepare UNet input for inpainting
        latent_model_input = torch.cat([noisy_latents, mask_latents, masked_image_latents], dim=1)
        
        # Get noise predictions from base model
        with torch.no_grad():
            # Van Gogh style prediction
            noise_pred_erase = self.base_unet(
                latent_model_input,
                timestep.to(self.device),
                encoder_hidden_states=erase_embeds.to(self.device),
                return_dict=False
            )[0]
            
            # Null prediction
            noise_pred_null = self.base_unet(
                latent_model_input,
                timestep.to(self.device),
                encoder_hidden_states=null_embeds.to(self.device),
                return_dict=False
            )[0]
            
            # Neutral prediction
            noise_pred_neutral = self.base_unet(
                latent_model_input,
                timestep.to(self.device),
                encoder_hidden_states=neutral_embeds.to(self.device),
                return_dict=False
            )[0]
        
        return {
            'latent_input': latent_model_input,
            'timestep': timestep,
            'erase_embeds': erase_embeds,
            'neutral_embeds': neutral_embeds,
            'noise_pred_erase': noise_pred_erase,
            'noise_pred_null': noise_pred_null,
            'noise_pred_neutral': noise_pred_neutral,
            'mask_latents': mask_latents
        }
    
    def compute_inpainting_esd_loss(self, predictions, negative_guidance=2.0):
        """Compute inpainting-aware ESD loss."""
        # Extract predictions
        noise_pred_erase = predictions['noise_pred_erase']
        noise_pred_null = predictions['noise_pred_null']
        noise_pred_neutral = predictions['noise_pred_neutral']
        mask_latents = predictions['mask_latents']
        
        # Get ESD model prediction
        noise_pred_esd = self.esd_unet(
            predictions['latent_input'],
            predictions['timestep'].to(self.device),
            encoder_hidden_states=predictions['erase_embeds'].to(self.device),
            return_dict=False
        )[0]
        
        # Compute target: pull toward neutral, push away from Van Gogh
        target_noise = noise_pred_neutral - (negative_guidance * (noise_pred_erase - noise_pred_null))
        
        # Apply loss only to masked regions
        mask_weight = mask_latents.expand_as(noise_pred_esd)
        
        # Masked loss computation
        loss_masked = self.criterion(noise_pred_esd * mask_weight, target_noise * mask_weight)
        
        # Optional: add small preservation loss for non-masked regions
        preservation_weight = 0.1
        loss_preserve = self.criterion(
            noise_pred_esd * (1 - mask_weight), 
            noise_pred_neutral * (1 - mask_weight)
        )
        
        total_loss = loss_masked + preservation_weight * loss_preserve
        
        return total_loss, loss_masked.item(), loss_preserve.item()
    
    def train(
        self,
        prompts_file="data/vangogh_prompts.csv",
        num_images=100,
        iterations=500,
        lr=1e-5,
        negative_guidance=2.0,
        train_method='esd-x',
        neutral_prompt="landscape painting",
        save_path="inpainting_esd_models",
        num_inference_steps=50
    ):
        """Train the inpainting erasure model."""
        
        print("Starting Inpainting Erasure Training")
        print("=" * 50)
        
        # Load prompts
        prompts = self.load_prompts(prompts_file)
        print(f"Loaded {len(prompts)} Van Gogh prompts")
        
        # Generate training images
        generated_images = self.generate_training_images(prompts, num_images)
        
        # Prepare training data with multiple masks
        training_samples = self.prepare_training_data(generated_images)
        
        # Setup training
        esd_param_names, esd_params = self.get_trainable_parameters(train_method)
        optimizer = torch.optim.Adam(esd_params, lr=lr)
        
        print(f"Training {len(esd_params)} parameters using method: {train_method}")
        
        # Setup scheduler
        self.pipe.scheduler.set_timesteps(num_inference_steps)
        
        # Training loop
        losses = []
        pbar = tqdm(range(iterations), desc='Training Inpainting ESD')
        
        for iteration in pbar:
            optimizer.zero_grad()
            
            # Select random training sample
            sample = random.choice(training_samples)
            
            # Select random timestep
            timestep_idx = random.randint(0, num_inference_steps - 1)
            timestep = self.pipe.scheduler.timesteps[timestep_idx]
            
            # Get noise predictions
            predictions = self.get_noise_predictions(
                sample['image'],
                sample['mask'], 
                sample['prompt'],
                neutral_prompt,
                timestep.unsqueeze(0).to(self.device)
            )
            
            # Compute loss
            total_loss, masked_loss, preserve_loss = self.compute_inpainting_esd_loss(
                predictions, negative_guidance
            )
            
            # Backpropagate
            total_loss.backward()
            optimizer.step()
            
            losses.append(total_loss.item())
            
            # Update progress bar
            pbar.set_postfix({
                'total_loss': total_loss.item(),
                'masked_loss': masked_loss,
                'preserve_loss': preserve_loss,
                'timestep': timestep_idx,
                'mask_type': sample['mask_type']
            })
        
        # Save model
        os.makedirs(save_path, exist_ok=True)
        
        esd_param_dict = {}
        for name, param in zip(esd_param_names, esd_params):
            esd_param_dict[name] = param
        
        model_path = os.path.join(save_path, f"inpainting_esd_vangogh_{train_method}.safetensors")
        save_file(esd_param_dict, model_path)
        
        # Save training metadata
        metadata = {
            'iterations': iterations,
            'lr': lr,
            'negative_guidance': negative_guidance,
            'train_method': train_method,
            'neutral_prompt': neutral_prompt,
            'num_training_images': num_images,
            'num_training_samples': len(training_samples),
            'final_loss': losses[-1] if losses else 0,
            'avg_loss': np.mean(losses) if losses else 0
        }
        
        metadata_path = os.path.join(save_path, "training_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nTraining completed!")
        print(f"Model saved to: {model_path}")
        print(f"Metadata saved to: {metadata_path}")
        print(f"Final loss: {losses[-1]:.6f}")
        print(f"Average loss: {np.mean(losses):.6f}")
        
        return losses


def main():
    parser = argparse.ArgumentParser(description="Inpainting Erasure for Concept Removal")
    parser.add_argument('--model_id', default="stabilityai/stable-diffusion-2-inpainting",
                       help='Model ID for inpainting')
    parser.add_argument('--prompts_file', default="data/vangogh_prompts.csv",
                       help='CSV file with Van Gogh prompts')
    parser.add_argument('--num_images', type=int, default=100,
                       help='Number of training images to generate')
    parser.add_argument('--iterations', type=int, default=500,
                       help='Number of training iterations')
    parser.add_argument('--lr', type=float, default=1e-5,
                       help='Learning rate')
    parser.add_argument('--negative_guidance', type=float, default=2.0,
                       help='Negative guidance strength')
    parser.add_argument('--train_method', default='esd-x',
                       choices=['esd-x', 'esd-u', 'esd-all', 'esd-x-strict'],
                       help='Training method')
    parser.add_argument('--neutral_prompt', default="landscape painting",
                       help='Neutral prompt to guide toward')
    parser.add_argument('--save_path', default="inpainting_esd_models",
                       help='Directory to save model')
    parser.add_argument('--device', default="cuda:0",
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Initialize eraser
    eraser = InpaintingEraser(
        model_id=args.model_id,
        device=args.device
    )
    
    # Train model
    losses = eraser.train(
        prompts_file=args.prompts_file,
        num_images=args.num_images,
        iterations=args.iterations,
        lr=args.lr,
        negative_guidance=args.negative_guidance,
        train_method=args.train_method,
        neutral_prompt=args.neutral_prompt,
        save_path=args.save_path
    )
    
    print("Inpainting erasure training completed successfully!")


if __name__ == "__main__":
    main()