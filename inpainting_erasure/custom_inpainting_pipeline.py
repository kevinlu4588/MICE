#!/usr/bin/env python3
"""
Fixed custom inpainting script using UNet and scheduler directly without HuggingFace pipeline.
This script manually handles the diffusion process for inpainting with proper preprocessing.
"""

import torch
import numpy as np
from PIL import Image, ImageDraw
from diffusers import UNet2DConditionModel, DDIMScheduler, AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
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
        
        # Initialize image processors (matching HF implementation)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, 
            do_normalize=False, 
            do_binarize=True, 
            do_convert_grayscale=True
        )
        
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

    def prepare_image_and_mask(self, image, mask, height, width, batch_size, do_classifier_free_guidance):
        """Prepare image and mask for inpainting using proper processors."""
        # Process image using VaeImageProcessor (matching HF implementation)
        init_image = self.image_processor.preprocess(image, height=height, width=width)
        init_image = init_image.to(dtype=torch.float32, device=self.device)
        
        # Process mask using mask processor  
        mask_condition = self.mask_processor.preprocess(mask, height=height, width=width)
        mask_condition = mask_condition.to(device=self.device)
        
        # Create masked image (areas to be inpainted are set to 0)
        masked_image = init_image * (mask_condition < 0.5)
        
        # Encode original image to latents
        with torch.no_grad():
            image_latents = self.vae.encode(init_image).latent_dist.sample()
            image_latents = image_latents * self.vae.config.scaling_factor
        
        # Encode masked image to latents
        with torch.no_grad():
            masked_image_latents = self.vae.encode(masked_image).latent_dist.sample()
            masked_image_latents = masked_image_latents * self.vae.config.scaling_factor
        
        # Prepare mask for latent space
        mask = torch.nn.functional.interpolate(
            mask_condition, size=(image_latents.shape[2], image_latents.shape[3])
        )
        
        # Duplicate for batch size
        if mask.shape[0] < batch_size:
            mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)
        if masked_image_latents.shape[0] < batch_size:
            masked_image_latents = masked_image_latents.repeat(batch_size // masked_image_latents.shape[0], 1, 1, 1)
        if image_latents.shape[0] < batch_size:
            image_latents = image_latents.repeat(batch_size // image_latents.shape[0], 1, 1, 1)
        
        # Handle classifier-free guidance
        if do_classifier_free_guidance:
            mask = torch.cat([mask] * 2)
            masked_image_latents = torch.cat([masked_image_latents] * 2)
        
        return image_latents, mask, masked_image_latents

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
        
        # Determine batch size and guidance
        batch_size = 1
        do_classifier_free_guidance = guidance_scale > 1.0
        
        # Encode text
        text_embeddings = self.encode_text(prompt, negative_prompt)
        
        # Prepare image and mask
        image_latents, mask, masked_image_latents = self.prepare_image_and_mask(
            image, mask, height, width, batch_size, do_classifier_free_guidance
        )
        
        # Set scheduler timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        
        # Get timesteps for strength
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start:]
        
        # Prepare initial latents
        latents_shape = (batch_size, 4, height // self.vae_scale_factor, width // self.vae_scale_factor)
        
        if generator is not None:
            noise = torch.randn(latents_shape, generator=generator, device=self.device, dtype=text_embeddings.dtype)
        else:
            noise = torch.randn(latents_shape, device=self.device, dtype=text_embeddings.dtype)
        
        # Initialize latents
        if strength == 1.0:
            latents = noise * self.scheduler.init_noise_sigma
        else:
            latents = self.scheduler.add_noise(image_latents, noise, timesteps[0:1])

        # Denoising loop
        for i, t in enumerate(timesteps):
            # Expand latents for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            
            # Scale model input (important for proper denoising)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
            # For 9-channel UNet (inpainting), concatenate mask and masked image
            if self.unet.config.in_channels == 9:
                latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)
            
            # Predict noise
            with torch.no_grad():
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings,
                    return_dict=False,
                )[0]
            
            # Perform classifier-free guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Compute the previous noisy sample
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            
            # For 4-channel UNet, apply masking manually (key difference from 9-channel)
            if self.unet.config.in_channels == 4:
                init_latents_proper = image_latents
                if do_classifier_free_guidance:
                    init_mask, _ = mask.chunk(2)
                else:
                    init_mask = mask
                
                if i < len(timesteps) - 1:
                    noise_timestep = timesteps[i + 1]
                    init_latents_proper = self.scheduler.add_noise(
                        init_latents_proper, noise, torch.tensor([noise_timestep], device=self.device)
                    )
                
                latents = (1 - init_mask) * init_latents_proper + init_mask * latents
            
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
    parser = argparse.ArgumentParser(description="Fixed custom inpainting using UNet and scheduler")
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
    print("Fixed Custom Inpainting Script")
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
    
    print(f"\nFixed custom inpainting completed!")
    print(f"Results saved to: {args.output_dir}")
    print(f"Files created:")
    print(f"  - original_image.png")
    print(f"  - mask.png")
    print(f"  - custom_inpainted_result.png")


if __name__ == "__main__":
    main()