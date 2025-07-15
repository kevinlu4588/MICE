# Claude Prompt Generation for Training Data

This directory contains scripts for generating diverse prompts using Claude API and creating training images from those prompts.

## Scripts

1. **generate_prompts_claude.py** - Connects to Claude API to generate diverse prompts for concepts
2. **generate_training_data_from_csv.py** - Generates training images using prompts from CSV files
3. **claude_prompt_pipeline_example.sh** - Example script showing the full pipeline

## Setup

1. Install required Python packages:
```bash
pip install anthropic python-dotenv
```

2. Get your Anthropic API key from: https://console.anthropic.com/settings/keys

3. Add your API key to the `.env` file in the root directory:
```bash
# In /share/u/kevin/erasing/.env
ANTHROPIC_API_KEY="your-api-key-here"
```

The script will automatically load the API key from the `.env` file. Alternatively, you can:
- Set it as an environment variable: `export ANTHROPIC_API_KEY="your-api-key-here"`
- Pass it directly: `--api_key "your-api-key-here"`

## Usage

### Step 1: Generate Prompts with Claude

```bash
python generate_prompts_claude.py \
    --concepts "van gogh" "picasso" "french horn" \
    --num_prompts 100
```

This will:
- Connect to Claude API
- Generate 100 diverse prompts for each concept
- Save prompts to CSV files in `../datasets/prompt_csvs/` directory

### Step 2: Generate Training Images

```bash
python generate_training_data_from_csv.py \
    --num_images 100
```

This will:
- Load prompts from CSV files in `../datasets/prompt_csvs/`
- Generate images using Stable Diffusion
- Save images to `../datasets/training_images/` directory

### Custom Directories

You can override the default directories:

```bash
# Custom directories
python generate_prompts_claude.py \
    --concepts "van gogh" \
    --output_dir /path/to/my/prompts

python generate_training_data_from_csv.py \
    --csv_dir /path/to/my/prompts \
    --output_dir /path/to/my/images
```

### Using with the Pipeline

To integrate with the multi-concept pipeline, update `run_multi_concept_pipeline.sh`:

```bash
# Replace the data generation step with:
python ../utils/generate_prompts_claude.py \
    --concepts "$concept" \
    --num_prompts $NUM_IMAGES \
    --output_dir "../datasets/prompt_csvs"

python ../utils/generate_training_data_from_csv.py \
    --concepts "$concept" \
    --csv_dir "../datasets/prompt_csvs" \
    --output_dir "training_images" \
    --num_images $NUM_IMAGES
```

## CSV Format

The generated CSV files have the following format:
```csv
index,prompt
0,"a vibrant painting of sunflowers by van gogh"
1,"van gogh style portrait of a farmer"
2,"swirling night sky painted by van gogh"
...
```

## Benefits

- **Diversity**: Claude generates highly varied prompts avoiding repetition
- **Quality**: Prompts are crafted specifically for each concept type
- **Reproducibility**: CSV files can be saved and reused
- **Flexibility**: Easy to regenerate or modify prompts

## Tips

- For artists, Claude generates prompts with varied subjects, styles, and compositions
- For objects/animals, prompts include different settings, angles, and contexts
- You can edit the CSV files manually to add/remove/modify prompts
- Consider generating more prompts than needed and selecting the best ones