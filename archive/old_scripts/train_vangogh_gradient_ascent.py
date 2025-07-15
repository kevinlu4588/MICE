#!/usr/bin/env python
# coding=utf-8
"""
Gradient Ascent training script for erasing Van Gogh concept from Stable Diffusion
Based on train_text_to_image.py but modified for concept erasure
"""

import argparse
import logging
import math
import os
import random
import shutil
from pathlib import Path
import json
from datetime import datetime

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_xformers_available
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

# Will error if the minimal version of diffusers is not installed
check_min_version("0.31.0.dev0")

logger = get_logger(__name__, log_level="INFO")

def parse_args():
    parser = argparse.ArgumentParser(description="Gradient ascent training for Van Gogh concept erasure")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        required=True,
        help="A folder containing the training data (Van Gogh images)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/2_gradient_ascent_vangogh",
        help="The output directory where the model will be saved.",
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="The resolution for input images",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size for training."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=500,
        help="Total number of training steps.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for gradient ascent",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help="The scheduler type to use.",
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=0, help="Number of steps for lr warmup."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for Adam.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for Adam.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for Adam")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="Log directory",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision.",
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether to use xformers."
    )
    parser.add_argument(
        "--train_only_xattn",
        action="store_true",
        default=True,
        help="Train only cross-attention layers",
    )
    parser.add_argument(
        "--validation_prompts",
        type=str,
        nargs="+",
        default=["a painting in the style of van gogh", "starry night", "sunflowers painting"],
        help="Prompts for validation",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help="Run validation every X steps",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_config=accelerator_project_config,
    )
    
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    
    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
    
    # Handle the output directory creation
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Load scheduler, tokenizer and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    
    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    # Set up trainable parameters
    if args.train_only_xattn:
        # Freeze all parameters first
        unet.requires_grad_(False)
        # Then unfreeze only cross-attention parameters
        for name, param in unet.named_parameters():
            if 'attn2' in name:
                param.requires_grad = True
                logger.info(f"Training parameter: {name}")
    else:
        unet.requires_grad_(True)
    
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available")
    
    # Enable TF32 for faster training on Ampere GPUs
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
    
    # Initialize the optimizer - only on parameters that require grad
    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    logger.info(f"Number of trainable parameters: {sum(p.numel() for p in trainable_params)}")
    
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    # Load the dataset
    data_files = {"train": os.path.join(args.train_data_dir, "**")}
    dataset = load_dataset("imagefolder", data_files=data_files, cache_dir=None)
    
    # Preprocessing the datasets
    train_transforms = transforms.Compose([
        transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(args.resolution),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    
    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples["image"]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        # Use the caption if available, otherwise use a default Van Gogh prompt
        captions = []
        for i in range(len(images)):
            if "text" in examples and examples["text"][i]:
                captions.append(examples["text"][i])
            else:
                captions.append("a painting by van gogh")
        
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, 
            padding="max_length", truncation=True, return_tensors="pt"
        )
        examples["input_ids"] = inputs.input_ids
        return examples
    
    train_dataset = dataset["train"].with_transform(preprocess_train)
    
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids}
    
    # DataLoader creation
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=0,
    )
    
    # Scheduler and math around the number of training steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )
    
    # Prepare everything with accelerator
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    
    # Move other models to device and cast to weight_dtype
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    
    # Initialize trackers
    if accelerator.is_main_process:
        accelerator.init_trackers("vangogh_gradient_ascent", config=vars(args))
    
    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    
    logger.info("***** Running gradient ascent training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    
    global_step = 0
    first_epoch = 0
    
    progress_bar = tqdm(
        range(args.max_train_steps),
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                
                # Sample noise
                noise = torch.randn_like(latents)
                
                # Sample timesteps
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()
                
                # Add noise to the latents
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Get text embeddings
                encoder_hidden_states = text_encoder(batch["input_ids"], return_dict=False)[0]
                
                # Predict the noise residual
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]
                
                # Get the target
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                
                # Compute loss
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                
                # GRADIENT ASCENT: Flip the sign of the loss
                loss = -loss
                
                # Gather the losses across all processes
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                
                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Checks if the accelerator has performed an optimization step
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0
                
                # Validation
                if global_step % args.validation_steps == 0:
                    logger.info(f"Running validation at step {global_step}...")
                    if accelerator.is_main_process:
                        pipeline = StableDiffusionPipeline.from_pretrained(
                            args.pretrained_model_name_or_path,
                            text_encoder=text_encoder,
                            vae=vae,
                            unet=accelerator.unwrap_model(unet),
                            safety_checker=None,
                            torch_dtype=weight_dtype,
                        )
                        pipeline = pipeline.to(accelerator.device)
                        pipeline.set_progress_bar_config(disable=True)
                        
                        if args.enable_xformers_memory_efficient_attention:
                            pipeline.enable_xformers_memory_efficient_attention()
                        
                        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
                        
                        validation_dir = os.path.join(args.output_dir, "validation", f"step_{global_step}")
                        os.makedirs(validation_dir, exist_ok=True)
                        
                        for i, prompt in enumerate(args.validation_prompts):
                            with torch.autocast("cuda"):
                                image = pipeline(prompt, num_inference_steps=50, generator=generator).images[0]
                            image.save(os.path.join(validation_dir, f"{i}_{prompt.replace(' ', '_')}.png"))
                        
                        del pipeline
                        torch.cuda.empty_cache()
            
            if global_step >= args.max_train_steps:
                break
    
    # Save the final model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        
        # Save just the trained weights
        save_path = args.output_dir
        os.makedirs(save_path, exist_ok=True)
        
        # Save model weights
        if args.train_only_xattn:
            # Save only the cross-attention weights
            xattn_state_dict = {}
            for name, param in unet.named_parameters():
                if 'attn2' in name and param.requires_grad:
                    xattn_state_dict[name] = param.data
            torch.save(xattn_state_dict, os.path.join(save_path, "pytorch_model.bin"))
        else:
            unet.save_pretrained(save_path)
        
        # Save the full pipeline
        pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            safety_checker=None,
        )
        pipeline.save_pretrained(save_path)
        
        # Save metadata
        metadata = {
            "model_name": "2_gradient_ascent_vangogh",
            "method": "Gradient Ascent",
            "erase_concept": "van gogh",
            "train_only_xattn": args.train_only_xattn,
            "max_train_steps": args.max_train_steps,
            "learning_rate": args.learning_rate,
            "timestamp": datetime.now().isoformat(),
            "base_model": args.pretrained_model_name_or_path,
            "description": "Gradient ascent model trained to erase Van Gogh concept from cross-attention layers"
        }
        
        with open(os.path.join(save_path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {save_path}")
    
    accelerator.end_training()

if __name__ == "__main__":
    main()