#!/usr/bin/env python3
"""
Custom inpainting script using UNet and scheduler directly without HuggingFace pipeline.
This script manually handles the diffusion process for inpainting.
"""

import torch
import numpy as np
from PIL import Image, ImageDraw
from diffusers import UNet2DConditionModel, DDIMScheduler, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
import os
import argparse
from pathlib import Path


class CustomInpainter:
    def __init__(self, model_id="stabilityai/stable-diffusion-2-inpainting", device=None):
        """Initialize the custom inpainter with individual components."""
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading model components from: {model_id}")
        print(f"Using device: {self.device}")
        
        # Load individual components
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
        self.scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
        
        # Move to device
        self.text_encoder = self.text_encoder.to(self.device)
        self.vae = self.vae.to(self.device)
        self.unet = self.unet.to(self.device)
        
        # Set to eval mode
        self.text_encoder.eval()
        self.vae.eval()
        self.unet.eval()
        
        print("All components loaded successfully!")
    
    def encode_text(self, prompt, negative_prompt=""):
        """Encode text prompts to embeddings."""
        # Tokenize prompts
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        # Get text embeddings
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_inputs.input_ids.to(self.device))[0]
        
        # Get unconditional embeddings for classifier-free guidance
        if negative_prompt:
            uncond_input = self.tokenizer(
                negative_prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
        else:
            uncond_input = self.tokenizer(
                "",
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            )
        
        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
        
        # Concatenate for classifier-free guidance
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        return text_embeddings
    
    def prepare_image(self, image, mask):
        """Prepare image and mask for inpainting."""
        # Convert PIL to tensor
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        mask = np.array(mask.convert("L")).astype(np.float32) / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Encode image to latent space
        with torch.no_grad():
            image_latents = self.vae.encode(image * 2.0 - 1.0).latent_dist.sample()
            image_latents = image_latents * self.vae.config.scaling_factor
        
        # Prepare mask for latent space (downsample)
        mask_latents = torch.nn.functional.interpolate(
            mask, size=(image_latents.shape[2], image_latents.shape[3])
        )
        
        # Invert mask (1 for areas to inpaint, 0 for areas to keep)
        mask_latents = 1 - mask_latents
        
        return image_latents, mask_latents
    
    def inpaint(
        self,
        prompt,
        image,
        mask,
        negative_prompt="",
        num_inference_steps=50,
        guidance_scale=7.5,
        strength=1.0,
        generator=None,
        height=512,
        width=512
    ):
        """Perform inpainting using the custom diffusion loop."""
        
        print(f"Starting inpainting process...")
        print(f"Prompt: {prompt}")
        print(f"Steps: {num_inference_steps}, Guidance: {guidance_scale}")
        
        # Encode text
        text_embeddings = self.encode_text(prompt, negative_prompt)
        
        # Prepare image and mask
        image_latents, mask_latents = self.prepare_image(image, mask)
        
        # Set scheduler timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        
        # Calculate start timestep for partial denoising
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start:]
        
        # Add noise to image latents
        if generator is not None:
            noise = torch.randn(image_latents.shape, generator=generator, device=self.device, dtype=image_latents.dtype)
        else:
            noise = torch.randn_like(image_latents)
        if t_start > 0:
            image_latents = self.scheduler.add_noise(image_latents, noise, timesteps[0:1])
        
        # Prepare latents
        latents = image_latents.clone()
        
        # Denoising loop
        for i, t in enumerate(timesteps):
            # Expand latents for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2)
            
            # Prepare masked latents and mask for inpainting UNet
            masked_image_latents = image_latents * mask_latents
            masked_image_latents = torch.cat([masked_image_latents] * 2)
            mask_input = torch.cat([mask_latents] * 2)
            
            # Concatenate for inpainting UNet (latents + masked_latents + mask)
            latent_model_input = torch.cat([latent_model_input, masked_image_latents, mask_input], dim=1)
            
            # Predict noise
            with torch.no_grad():
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings,
                    return_dict=False,
                )[0]
            
            # Perform classifier-free guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Compute the previous noisy sample
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            
            # Apply mask to preserve non-inpainted areas
            if i < len(timesteps) - 1:
                noise_timestep = timesteps[i + 1] if i + 1 < len(timesteps) else torch.tensor(0)
                init_latents_proper = self.scheduler.add_noise(image_latents, noise, noise_timestep.unsqueeze(0))
                latents = (init_latents_proper * mask_latents) + (latents * (1 - mask_latents))
            
            if i % 10 == 0:
                print(f"Step {i+1}/{len(timesteps)}")
        
        # Decode latents to image
        with torch.no_grad():
            latents = latents / self.vae.config.scaling_factor
            image = self.vae.decode(latents).sample
        
        # Convert to PIL
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = (image * 255).astype(np.uint8)
        
        return Image.fromarray(image[0])


def create_mask(image_size, mask_type="center_square"):
    """Create a mask for inpainting."""
    width, height = image_size
    mask = Image.new("RGB", (width, height), "black")
    draw = ImageDraw.Draw(mask)
    
    if mask_type == "center_square":
        # Create a white square in the center
        square_size = min(width, height) // 3
        x1 = (width - square_size) // 2
        y1 = (height - square_size) // 2
        x2 = x1 + square_size
        y2 = y1 + square_size
        draw.rectangle([x1, y1, x2, y2], fill="white")
    elif mask_type == "circle":
        # Create a white circle in the center
        radius = min(width, height) // 4
        center_x, center_y = width // 2, height // 2
        draw.ellipse([center_x - radius, center_y - radius, 
                     center_x + radius, center_y + radius], fill="white")
    elif mask_type == "car_area":
        # Create a mask for the car area (centered)
        car_width = int(width * 0.4)  # 40% of image width
        car_height = int(height * 0.3)  # 30% of image height
        x1 = (width - car_width) // 2
        y1 = (height - car_height) // 2
        x2 = x1 + car_width
        y2 = y1 + car_height
        draw.rectangle([x1, y1, x2, y2], fill="white")
    
    return mask


def main():
    parser = argparse.ArgumentParser(description="Custom inpainting using UNet and scheduler")
    parser.add_argument("--model", default="stabilityai/stable-diffusion-2-inpainting",
                       help="Model ID to use for inpainting")
    parser.add_argument("--prompt", default="a beautiful landscape with mountains",
                       help="Prompt for inpainting")
    parser.add_argument("--negative-prompt", default="",
                       help="Negative prompt for inpainting")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--mask-type", choices=["center_square", "circle", "car_area"], 
                       default="center_square", help="Type of mask to create")
    parser.add_argument("--output-dir", default="./custom_inpainting_results",
                       help="Output directory for results")
    parser.add_argument("--steps", type=int, default=50,
                       help="Number of inference steps")
    parser.add_argument("--guidance-scale", type=float, default=7.5,
                       help="Guidance scale for generation")
    parser.add_argument("--strength", type=float, default=1.0,
                       help="Strength of the inpainting (0.0 to 1.0)")
    parser.add_argument("--device", help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Custom Inpainting Script")
    print("=" * 60)
    
    # Load image
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        return
    
    image = Image.open(args.image).convert("RGB")
    image = image.resize((512, 512))
    
    # Create mask
    mask = create_mask(image.size, args.mask_type)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize custom inpainter
    inpainter = CustomInpainter(args.model, args.device)
    
    # Perform inpainting
    result = inpainter.inpaint(
        prompt=args.prompt,
        image=image,
        mask=mask,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        strength=args.strength,
        height=512,
        width=512
    )
    
    # Save results
    image.save(os.path.join(args.output_dir, "original_image.png"))
    mask.save(os.path.join(args.output_dir, "mask.png"))
    result.save(os.path.join(args.output_dir, "custom_inpainted_result.png"))
    
    print(f"\nCustom inpainting completed!")
    print(f"Results saved to: {args.output_dir}")
    print(f"Files created:")
    print(f"  - original_image.png")
    print(f"  - mask.png")
    print(f"  - custom_inpainted_result.png")


if __name__ == "__main__":
    main()