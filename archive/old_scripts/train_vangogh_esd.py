import os
import torch
import sys
import random
from tqdm.auto import tqdm
from safetensors.torch import save_file
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
import argparse
import json
from datetime import datetime

sys.path.append('./original_esd')
from utils.sd_utils import esd_sd_call
StableDiffusionPipeline.__call__ = esd_sd_call

def load_sd_models(basemodel_id="CompVis/stable-diffusion-v1-4", torch_dtype=torch.bfloat16, device='cuda:0'):
    
    base_unet = UNet2DConditionModel.from_pretrained(basemodel_id, subfolder="unet").to(device, torch_dtype)
    base_unet.requires_grad_(False)
    
    esd_unet = UNet2DConditionModel.from_pretrained(basemodel_id, subfolder="unet").to(device, torch_dtype)
    pipe = StableDiffusionPipeline.from_pretrained(basemodel_id, unet=base_unet, torch_dtype=torch_dtype, use_safetensors=True).to(device)
    
    return pipe, base_unet, esd_unet

def get_esd_trainable_parameters(esd_unet, train_method='esd-x'):
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

if __name__ == '__main__':
    # Load models.json to determine next model index
    models_json_path = "models.json"
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
    
    # Fixed parameters for Van Gogh erasure
    erase_concept = "van gogh"
    erase_concept_from = None
    num_inference_steps = 50
    guidance_scale = 3.0
    negative_guidance = 2.0
    train_method = "esd-x"  # Only cross-attention
    iterations = 200
    batchsize = 1
    height = width = 512
    lr = 1e-4
    save_path = f"models/{next_index}_esd_{erase_concept.replace(' ', '_')}_xattn"
    os.makedirs(save_path, exist_ok=True)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch_dtype = torch.bfloat16
    
    print(f"Training ESD model to erase '{erase_concept}' using method '{train_method}'")
    print(f"Model will be saved to: {save_path}")
    
    criteria = torch.nn.MSELoss()

    pipe, base_unet, esd_unet = load_sd_models(basemodel_id="CompVis/stable-diffusion-v1-4", torch_dtype=torch_dtype, device=device)
    pipe.set_progress_bar_config(disable=True)
    pipe.scheduler.set_timesteps(num_inference_steps)

    esd_param_names, esd_params = get_esd_trainable_parameters(esd_unet, train_method=train_method)
    optimizer = torch.optim.Adam(esd_params, lr=lr)

    with torch.no_grad():
        # get prompt embeds
        erase_embeds, null_embeds = pipe.encode_prompt(prompt=erase_concept,
                                                       device=device,
                                                       num_images_per_prompt=batchsize,
                                                       do_classifier_free_guidance=True,
                                                       negative_prompt='')
                                                 
        erase_embeds = erase_embeds.to(device)
        null_embeds = null_embeds.to(device)
        
        timestep_cond = None
        if pipe.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(guidance_scale - 1).repeat(batchsize)
            timestep_cond = pipe.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=pipe.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=torch_dtype)
        
        if erase_concept_from is not None:
            erase_from_embeds, _ = pipe.encode_prompt(prompt=erase_concept_from,
                                                                device=device,
                                                                num_images_per_prompt=batchsize,
                                                                do_classifier_free_guidance=False,
                                                                negative_prompt="",
                                                                )
            erase_from_embeds = erase_from_embeds.to(device)
    
    
    pbar = tqdm(range(iterations), desc='Training ESD')
    losses = []
    for iteration in pbar:
        optimizer.zero_grad()
        # get the noise predictions for erase concept
        pipe.unet = base_unet
        run_till_timestep = random.randint(0, num_inference_steps-1)
        run_till_timestep_scheduler = pipe.scheduler.timesteps[run_till_timestep]
        seed = random.randint(0, 2**15)
        with torch.no_grad():
            xt = pipe(erase_concept if erase_concept_from is None else erase_concept_from,
                  num_images_per_prompt=batchsize,
                  num_inference_steps=num_inference_steps,
                  guidance_scale=guidance_scale,
                  run_till_timestep = run_till_timestep,
                  generator=torch.Generator().manual_seed(seed),
                  output_type='latent',
                  height=height,
                  width=width,
                 ).images
    
            noise_pred_erase = pipe.unet(
                xt,
                run_till_timestep_scheduler,
                encoder_hidden_states=erase_embeds,
                timestep_cond=timestep_cond,
                cross_attention_kwargs=None,
                added_cond_kwargs=None,
                return_dict=False,
            )[0]
            
            # get the noise predictions for null embeds
            noise_pred_null = pipe.unet(
                xt,
                run_till_timestep_scheduler,
                encoder_hidden_states=null_embeds,
                timestep_cond=timestep_cond,
                cross_attention_kwargs=None,
                added_cond_kwargs=None,
                return_dict=False,
            )[0]
            
            # get the noise predictions for erase concept from embeds
            if erase_concept_from is not None:
                noise_pred_erase_from = pipe.unet(
                    xt,
                    run_till_timestep_scheduler,
                    encoder_hidden_states=erase_from_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=None,
                    added_cond_kwargs=None,
                    return_dict=False,
                )[0]
            else:
                noise_pred_erase_from = noise_pred_erase
        
        
        pipe.unet = esd_unet
        noise_pred_esd_model = pipe.unet(
            xt,
            run_till_timestep_scheduler,
            encoder_hidden_states=erase_embeds if erase_concept_from is None else erase_from_embeds,
            timestep_cond=timestep_cond,
            cross_attention_kwargs=None,
            added_cond_kwargs=None,
            return_dict=False,
        )[0]
        
        
        loss = criteria(noise_pred_esd_model, noise_pred_erase_from - (negative_guidance*(noise_pred_erase - noise_pred_null))) 
        loss.backward()
        losses.append(loss.item())
        pbar.set_postfix(esd_loss=loss.item(),
                         timestep=run_till_timestep,)
        optimizer.step()
    
    # Save the model weights
    esd_param_dict = {}
    for name, param in zip(esd_param_names, esd_params):
        esd_param_dict[name] = param
    
    save_file(esd_param_dict, f"{save_path}/model.safetensors")
    
    # Save metadata
    model_name = f"{next_index}_esd_{erase_concept.replace(' ', '_')}_xattn"
    metadata = {
        "model_name": model_name,
        "method": "ESD (Erased Stable Diffusion)",
        "erase_concept": erase_concept,
        "train_method": train_method,
        "iterations": iterations,
        "learning_rate": lr,
        "negative_guidance": negative_guidance,
        "guidance_scale": guidance_scale,
        "timestamp": datetime.now().isoformat(),
        "base_model": "CompVis/stable-diffusion-v1-4",
        "description": "ESD model trained to erase Van Gogh concept from cross-attention layers only"
    }
    
    with open(f"{save_path}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Update models.json
    models_data[str(next_index)] = {
        "name": model_name,
        "method": "ESD (Erased Stable Diffusion)",
        "weights_trained": "cross_attention",
        "concept_erased": erase_concept,
        "base_model": "CompVis/stable-diffusion-v1-4",
        "description": "ESD model trained to erase Van Gogh concept from cross-attention layers only"
    }
    
    with open(models_json_path, 'w') as f:
        json.dump(models_data, f, indent=2)
    
    print(f"\nTraining complete! Model saved to {save_path}")
    print(f"Updated models.json with model index {next_index}")
    print(f"Final loss: {losses[-1]:.6f}")