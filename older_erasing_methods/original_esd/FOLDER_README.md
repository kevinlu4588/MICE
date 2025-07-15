# Original ESD (Erasing Stable Diffusion) Code

This folder contains the original ESD implementation and supporting files.

## Core Files

### Training Scripts
- `esd_sd.py` - Main ESD training script for Stable Diffusion v1.x
- `esd_sdxl.py` - ESD training script for SDXL models
- `alt_esd_diffusers.py` - Alternative ESD implementation using diffusers

### Notebooks
- `notebooks/esd_inference_sd.ipynb` - SD inference examples
- `notebooks/esd_inference_sdxl.ipynb` - SDXL inference examples  
- `notebooks/alt_esd_inference.ipynb` - Alternative inference methods

### Evaluation Scripts
- `evalscripts/generate-images.py` - Image generation for evaluation
- `evalscripts/imageclassify.py` - Image classification evaluation
- `evalscripts/lpips_eval.py` - LPIPS perceptual similarity evaluation
- `evalscripts/nudenet-classes.py` - NSFW content detection
- `evalscripts/sld-generate-images.py` - SLD image generation
- `evalscripts/styleloss.py` - Style transfer loss evaluation

### Utilities
- `utils/sd_utils.py` - Stable Diffusion utilities
- `utils/sdxl_utils.py` - SDXL specific utilities
- `utils/utils.py` - General utility functions

### Data
- `data/` - Contains various prompt datasets for training and evaluation
  - `art_prompts.csv` - Art-related prompts
  - `coco_30k.csv` - COCO dataset prompts
  - `imagenet_prompts.csv` - ImageNet class prompts
  - `unsafe-prompts4703.csv` - NSFW prompts for safety evaluation
  - And more specialized prompt sets

### Documentation
- `images/` - Documentation images and figures
- `requirements.txt` - Python dependencies
- `LICENSE` - Project license
- `README.md` - Original project documentation

## Usage

Refer to the main README.md for detailed usage instructions for the original ESD implementation.