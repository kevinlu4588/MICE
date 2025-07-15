#!/usr/bin/env python
"""
Inpainting Erasure Training Script with ESD Method
Uses masking and inpainting task with ESD (Erased Stable Diffusion) to erase concepts
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline, UNet2DConditionModel, DDPMScheduler
from diffusers.optimization import get_scheduler
from PIL import Image
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
import argparse
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL
from PIL import Image
from torchvision.transforms.functional import to_pil_image
import random

class InpaintingDataset(Dataset):
    """Dataset for inpainting erasure training"""
    def __init__(self, image_dir, caption="a painting by van gogh", image_size=512):
        self.image_paths = list(Path(image_dir).glob("*.png")) + list(Path(image_dir).glob("*.jpg"))
        if len(self.image_paths) == 0:
            print(f"Warning: No images found in {image_dir}")
            # Check for images with different extensions
            self.image_paths = list(Path(image_dir).glob("*"))
            self.image_paths = [p for p in self.image_paths if p.is_file() and p.suffix.lower() in ['.png', '.jpg', '.jpeg']]
        print(f"Found {len(self.image_paths)} images in dataset")
        self.caption = caption
        self.image_size = image_size
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
        
        # Convert to tensor
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        image = 2.0 * image - 1.0  # Normalize to [-1, 1]
        
        # Create random mask (we'll mask random regions to force inpainting)
        mask = torch.zeros((1, self.image_size, self.image_size))
        
        # Random rectangular mask
        h_start = torch.randint(0, self.image_size // 2, (1,)).item()
        w_start = torch.randint(0, self.image_size // 2, (1,)).item()
        h_size = torch.randint(self.image_size // 4, self.image_size // 2, (1,)).item()
        w_size = torch.randint(self.image_size // 4, self.image_size // 2, (1,)).item()
        
        mask[:, h_start:h_start+h_size, w_start:w_start+w_size] = 1.0
        
        # Create masked image (masked regions set to gray)
        masked_image = image.clone()
        masked_image[:, mask[0] > 0.5] = 0.0
        
        return {
            "image": image,
            "masked_image": masked_image,
            "mask": mask,
            "caption": self.caption
        }

def get_esd_trainable_parameters(esd_unet, train_method='esd-x'):
    """Get trainable parameters based on ESD training method"""
    esd_params = []
    esd_param_names = []
    for name, module in esd_unet.named_modules():
        if module.__class__.__name__ in ["Linear", "Conv2d", "LoRACompatibleLinear", "LoRACompatibleConv"]:
            if train_method == 'esd-x' and 'attn2' in name:
                for n, p in module.named_parameters():
                    esd_param_names.append(name+'.'+n)
                    esd_params.append(p)
                    
            if train_method == 'esd-u' and ('attn2' not in name):
                for n, p in module.named_parameters():
                    esd_param_names.append(name+'.'+n)
                    esd_params.append(p)
                    
            if train_method == 'esd-all' :
                for n, p in module.named_parameters():
                    esd_param_names.append(name+'.'+n)
                    esd_params.append(p)
                    
            if train_method == 'esd-x-strict' and ('attn2.to_k' in name or 'attn2.to_v' in name):
                for n, p in module.named_parameters():
                    esd_param_names.append(name+'.'+n)
                    esd_params.append(p)

    return esd_param_names, esd_params

def train_inpainting_erasure_esd(args):
    """Main training function for inpainting erasure with ESD"""
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load models.json to determine next model index
    models_json_path = "../utils/models.json"
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
    
    # Override with manual index if specified and it's higher
    if args.model_index is not None and args.model_index >= next_index:
        next_index = args.model_index
    
    # Create output directory
    output_dir = f"../models/{next_index}_inpainting_erasure_esd_{args.concept.replace(' ', '_')}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Training inpainting erasure model with ESD for concept: {args.concept}")
    print(f"Output directory: {output_dir}")
    print(f"Training method: {args.train_method}")
    
    # Load models
    print("Loading models...")
    model_id = "CompVis/stable-diffusion-v1-4"
    
    # Load scheduler and models separately
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    
    # Load two UNets for ESD - base (frozen) and trainable
    base_unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    esd_unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    
    # Move to device and set precision
    text_encoder = text_encoder.to(device, dtype=torch.float16)
    vae = vae.to(device, dtype=torch.float16)
    base_unet = base_unet.to(device, dtype=torch.float32)
    esd_unet = esd_unet.to(device, dtype=torch.float32)
    
    # Freeze text encoder, VAE, and base UNet
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    base_unet.requires_grad_(False)
    
    # Get trainable parameters based on ESD method
    esd_param_names, esd_params = get_esd_trainable_parameters(esd_unet, train_method=args.train_method)
    print(f"Number of trainable parameters: {len(esd_params)}")
    for name in esd_param_names[:5]:  # Print first 5 parameter names
        print(f"Training parameter: {name}")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(esd_params, lr=args.learning_rate)
    
    # Setup dataset and dataloader
    dataset = InpaintingDataset(args.data_dir, caption=f"a painting by {args.concept}")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Training settings
    num_epochs = args.num_epochs
    num_training_steps = num_epochs * len(dataloader)
    
    # Learning rate scheduler
    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    
    # Create pipeline for encoding prompts
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        unet=base_unet,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    
    # Get prompt embeddings for ESD
    with torch.no_grad():
        # Erase concept embeddings
        erase_embeds, null_embeds = pipe.encode_prompt(
            prompt=args.concept,
            device=device,
            num_images_per_prompt=args.batch_size,
            do_classifier_free_guidance=True,
            negative_prompt=''
        )
        erase_embeds = erase_embeds.to(device)
        null_embeds = null_embeds.to(device)
    
    # MSE loss for ESD
    criteria = torch.nn.MSELoss()
    
    # Training loop
    print(f"Starting training for {num_epochs} epochs...")
    global_step = 0
    
    for epoch in range(num_epochs):
        esd_unet.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            # Move batch to device
            images = batch["image"].to(device, dtype=torch.float16)
            masked_images = batch["masked_image"].to(device, dtype=torch.float16)
            masks = batch["mask"].to(device, dtype=torch.float16)
            
            # Encode images to latent space
            with torch.no_grad():
                latents = vae.encode(images).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                
                masked_latents = vae.encode(masked_images).latent_dist.sample()
                masked_latents = masked_latents * vae.config.scaling_factor
                
                # Resize mask to match latent dimensions
                mask_latent = F.interpolate(masks, size=(latents.shape[2], latents.shape[3]))
                
                # Get text embeddings for the batch
                text_inputs = tokenizer(
                    batch["caption"],
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                )
                text_embeddings = text_encoder(text_inputs.input_ids.to(device))[0]
            
            # Sample random timestep for each sample
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (args.batch_size,), device=device)
            timesteps = timesteps.long()
            
            # Sample noise
            noise = torch.randn_like(latents)
            
            # Add noise to latents
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # Blend noisy latents with masked latents for inpainting
            blended_latents = noisy_latents * (1 - mask_latent) + masked_latents * mask_latent
            
            # Get noise predictions from base UNet
            with torch.no_grad():
                # Noise prediction with erase concept
                noise_pred_erase = base_unet(
                    blended_latents.to(torch.float32),
                    timesteps,
                    encoder_hidden_states=erase_embeds.to(torch.float32),
                    return_dict=False
                )[0]
                
                # Noise prediction with null prompt
                noise_pred_null = base_unet(
                    blended_latents.to(torch.float32),
                    timesteps,
                    encoder_hidden_states=null_embeds.to(torch.float32),
                    return_dict=False
                )[0]
                
                # For inpainting, we use the text embeddings as the "from" concept
                noise_pred_from = base_unet(
                    blended_latents.to(torch.float32),
                    timesteps,
                    encoder_hidden_states=text_embeddings.to(torch.float32),
                    return_dict=False
                )[0]
            
            # Get noise prediction from ESD UNet
            noise_pred_esd = esd_unet(
                blended_latents.to(torch.float32),
                timesteps,
                encoder_hidden_states=text_embeddings.to(torch.float32),
                return_dict=False
            )[0]
            
            # ESD loss calculation
            # We want the ESD model to predict noise that moves away from the erase concept
            target = noise_pred_from - (args.negative_guidance * (noise_pred_erase - noise_pred_null))
            loss = criteria(noise_pred_esd, target)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            # Update progress
            epoch_loss += loss.item()
            global_step += 1
            progress_bar.set_postfix({"loss": loss.item()})
        
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} average loss: {avg_epoch_loss:.6f}")
        
        # Save checkpoint every few epochs
        if (epoch + 1) % args.save_every == 0:
            checkpoint_dir = os.path.join(output_dir, f"checkpoint-{epoch+1}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Save only the ESD UNet weights
            torch.save({
                'epoch': epoch,
                'model_state_dict': esd_unet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
                'model_index': next_index,
                'esd_param_names': esd_param_names
            }, os.path.join(checkpoint_dir, "checkpoint.pt"))
            
            print(f"Saved checkpoint at epoch {epoch+1}")
    
    # Save final model
    print("Saving final model...")
    
    # Create pipeline with modified UNet
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        unet=esd_unet,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False
    )
    
    # Save the full pipeline
    pipe.save_pretrained(output_dir)
    
    # Save just the ESD-trained weights
    esd_weights = {}
    for name, param in zip(esd_param_names, esd_params):
        esd_weights[name] = param.data.cpu()
    
    torch.save(esd_weights, os.path.join(output_dir, "inpainting_erasure_esd_weights.pt"))
    
    # Save metadata
    model_name = f"{next_index}_inpainting_erasure_esd_{args.concept.replace(' ', '_')}"
    metadata = {
        "model_name": model_name,
        "method": f"Inpainting Erasure with ESD ({args.train_method})",
        "concept": args.concept,
        "base_model": model_id,
        "num_epochs": num_epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "train_method": args.train_method,
        "negative_guidance": args.negative_guidance,
        "timestamp": datetime.now().isoformat(),
        "description": f"Inpainting erasure model trained to erase {args.concept} concept using masked inpainting and ESD method"
    }
    
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Update models.json
    models_data[str(next_index)] = {
        "name": model_name,
        "method": f"Inpainting Erasure with ESD ({args.train_method})",
        "weights_trained": args.train_method,
        "concept_erased": args.concept,
        "base_model": model_id,
        "description": f"Inpainting erasure model using masked inpainting and ESD method to erase {args.concept} concept"
    }
    
    with open(models_json_path, 'w') as f:
        json.dump(models_data, f, indent=2)
    
    print(f"Training complete! Model saved to {output_dir}")
    print(f"Updated models.json with model index {next_index}")

def main():
    parser = argparse.ArgumentParser(description="Train inpainting erasure model with ESD")
    parser.add_argument("--concept", type=str, default="van gogh", help="Concept to erase")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing training images")
    parser.add_argument("--model_index", type=int, default=None, help="Model index for naming")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--save_every", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument("--train_method", type=str, default="esd-x", 
                        choices=["esd-x", "esd-u", "esd-all", "esd-x-strict"],
                        help="ESD training method")
    parser.add_argument("--negative_guidance", type=float, default=2.0, 
                        help="Negative guidance value for ESD")
    
    args = parser.parse_args()
    train_inpainting_erasure_esd(args)

if __name__ == "__main__":
    main()