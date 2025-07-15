#!/usr/bin/env python
"""
Inpainting Erasure Training Script
Uses masking and inpainting task with gradient ascent to erase concepts
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

def train_inpainting_erasure(args):
    """Main training function for inpainting erasure"""
    
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
    output_dir = f"../models/{next_index}_inpainting_erasure_{args.concept.replace(' ', '_')}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Training inpainting erasure model for concept: {args.concept}")
    print(f"Output directory: {output_dir}")
    
    # Load models
    print("Loading models...")
    model_id = "CompVis/stable-diffusion-v1-4"
    
    # Load scheduler and models separately
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    
    # Move to device and set precision
    text_encoder = text_encoder.to(device, dtype=torch.float16)
    vae = vae.to(device, dtype=torch.float16)
    unet = unet.to(device, dtype=torch.float32)  # Keep UNet in float32 for training
    
    # Freeze text encoder and VAE
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    
    # Set up training for cross-attention layers only
    unet.requires_grad_(False)
    for name, param in unet.named_parameters():
        if 'attn2' in name:  # Cross-attention layers
            param.requires_grad = True
            print(f"Training parameter: {name}")
    
    # Setup optimizer
    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)
    
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
    
    # Training loop
    print(f"Starting training for {num_epochs} epochs...")
    global_step = 0
    
    for epoch in range(num_epochs):
        unet.train()
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
                
                # Get text embeddings
                text_inputs = tokenizer(
                    batch["caption"],
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                )
                text_embeddings = text_encoder(text_inputs.input_ids.to(device))[0]
            
            # Sample noise
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (args.batch_size,), device=device)
            timesteps = timesteps.long()
            
            # Add noise to latents
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # For standard UNet, we'll use the noisy latents directly
            # and incorporate the mask information differently
            # We'll blend the noisy latents with masked latents based on the mask
            blended_latents = noisy_latents * (1 - mask_latent) + masked_latents * mask_latent
            # VAE decode utility
            def decode_and_save(latents, filename_prefix, vae):
                # Scale down the latents before decoding
                scaled = latents / vae.config.scaling_factor
                with torch.no_grad():
                    imgs = vae.decode(scaled).sample  # Shape: (B, 3, 512, 512)

                for i, img in enumerate(imgs):
                    img = img.clamp(0, 1).cpu()
                    pil = to_pil_image(img)
                    pil.save(os.path.join(output_dir, f"{filename_prefix}_{i}.png"))
                    print("saved to", os.path.join(output_dir, f"{filename_prefix}_{i}.png"))

            # 1. Save decoded noisy latents
            decode_and_save(noisy_latents, "noisy_latents", vae)

            # 2. Save decoded blended latents
            decode_and_save(blended_latents, "blended_latents", vae)

            # 3. Save decoded masked latents
            decode_and_save(masked_latents, "masked_latents", vae)

            # 4. Save mask latent as grayscale images
            mask_dir = os.path.join(output_dir, "mask_latents")
            os.makedirs(mask_dir, exist_ok=True)

            for i, mask in enumerate(mask_latent):
                # If mask is (1, 64, 64), squeeze channel dim
                if mask.shape[0] == 1:
                    mask_img = mask.squeeze(0)
                else:
                    # Optionally average across channels to get grayscale
                    mask_img = mask.mean(dim=0)

                # Normalize for visualization
                mask_img = mask_img.clamp(0, 1).cpu()
                pil = to_pil_image(mask_img)
                pil.save(os.path.join(mask_dir, f"mask_latent_{i}.png"))

            # quit("saved latents")

            # Predict noise residual
            noise_pred = unet(
                blended_latents.to(torch.float32),
                timesteps,
                encoder_hidden_states=text_embeddings.to(torch.float32),
                return_dict=False
            )[0]
            
            # Calculate loss (MSE between predicted and actual noise)
            loss = F.mse_loss(noise_pred, noise.to(torch.float32))
            
            # GRADIENT ASCENT: Negate the loss to maximize it
            loss = -loss
            
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
            
            # Save only the modified UNet weights
            torch.save({
                'epoch': epoch,
                'model_state_dict': unet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
                'model_index': next_index
            }, os.path.join(checkpoint_dir, "checkpoint.pt"))
            
            print(f"Saved checkpoint at epoch {epoch+1}")
    
    # Save final model
    print("Saving final model...")
    
    # Create pipeline with modified UNet
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        unet=unet,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False
    )
    
    # Save the full pipeline
    pipe.save_pretrained(output_dir)
    
    # Save just the UNet weights separately
    unet_weights = {}
    for name, param in unet.named_parameters():
        if param.requires_grad:
            unet_weights[name] = param.data.cpu()
    
    torch.save(unet_weights, os.path.join(output_dir, "inpainting_erasure_weights.pt"))
    
    # Save metadata
    model_name = f"{next_index}_inpainting_erasure_{args.concept.replace(' ', '_')}"
    metadata = {
        "model_name": model_name,
        "method": "Inpainting Erasure with Gradient Ascent",
        "concept": args.concept,
        "base_model": model_id,
        "num_epochs": num_epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "timestamp": datetime.now().isoformat(),
        "description": f"Inpainting erasure model trained to erase {args.concept} concept using masked inpainting and gradient ascent"
    }
    
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Update models.json
    models_data[str(next_index)] = {
        "name": model_name,
        "method": "Inpainting Erasure with Gradient Ascent",
        "weights_trained": "cross_attention",
        "concept_erased": args.concept,
        "base_model": model_id,
        "description": f"Inpainting erasure model using masked inpainting and gradient ascent to erase {args.concept} concept"
    }
    
    with open(models_json_path, 'w') as f:
        json.dump(models_data, f, indent=2)
    
    print(f"Training complete! Model saved to {output_dir}")
    print(f"Updated models.json with model index {next_index}")

def main():
    parser = argparse.ArgumentParser(description="Train inpainting erasure model")
    parser.add_argument("--concept", type=str, default="van gogh", help="Concept to erase")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing training images")
    parser.add_argument("--model_index", type=int, default=3, help="Model index for naming")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--save_every", type=int, default=5, help="Save checkpoint every N epochs")
    
    args = parser.parse_args()
    train_inpainting_erasure(args)

if __name__ == "__main__":
    main()