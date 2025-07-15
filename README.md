# Concept Erasure Project

This project implements various methods for erasing concepts from Stable Diffusion models, with a focus on inpainting erasure techniques.

## Project Structure

```
.
├── models/                      # Trained models (indexed by number)
├── evaluation_results/          # Results from model evaluations
├── training_images/             # Generated training data for concepts
├── datasets/                    # Dataset storage
├── archive/                     # Old/unused files
│
├── evaluation/
│   ├── evaluate_erased_model.py # Evaluation script for erased models
│   ├── evaluate_latest.sh       # Evaluate the most recent model
│   └── evaluate_dynamic.sh      # Evaluate with custom parameters
│
├── inpainting_erasure/
│   ├── train_inpainting_erasure.py  # Main training script
│   └── run_inpainting_erasure.sh    # Run training script
│
├── utils/
│   ├── models.json              # Registry of all trained models
│   ├── generate_training_data.py    # Generate training images
│   └── generate_all_training_data.sh# Generate all training data
│
├── test_model.ipynb            # Notebook for testing models
├── README.md                   # This file
└── CLAUDE.md                   # Project-specific instructions
```

## Quick Start

### 1. Generate Training Data
```bash
cd utils/
./generate_all_training_data.sh
```

### 2. Train a Model
```bash
cd inpainting_erasure/
./run_inpainting_erasure.sh
```

### 3. Evaluate a Model
```bash
cd evaluation/

# Evaluate the latest model
./evaluate_latest.sh

# Or evaluate a specific model
./evaluate_dynamic.sh ../models/3_inpainting_erasure_van_gogh "van gogh"
```

### 4. Test Models Interactively
Open `test_model.ipynb` in Jupyter to interactively test models.

## Concepts Available
- picasso
- andy warhol  
- english springer spaniel
- airliner
- chainsaw
- golf ball

## Model Registry
All trained models are tracked in `models.json` with incremental indices.

## Key Directories

### `original_esd/`
Original ESD (Erasing Stable Diffusion) implementation files.

### `inpainting_erasure/`
Core inpainting erasure method implementations.

### `gradient_ascent_erasure/`
Gradient ascent based erasure implementations.

### `evaluation/`
Comprehensive evaluation tools for analyzing erasure effectiveness.

## Architecture
Built on Stable Diffusion v1.4 with cross-attention layer modifications using gradient ascent and inpainting techniques.