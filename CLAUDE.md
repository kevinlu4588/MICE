# Claude Context

## Model Naming Rules

Each new trained model must follow an incremental index naming convention:
- Format: `{index}_{model_description}`
- Examples:
  - `0_gradient_ascent`
  - `1_feature_extraction`
  - `2_adversarial_training`

The index should increment with each new model to maintain a clear training history and ensure unique model identification.

## Evaluation Rules

When evaluating, only run the scripts that will generate image grids. Do not waste tokens analyzing the images themselves.

## Model Training Rules

For each model trained, the code should first check the models.json file to see what the next model index should be and then after training update the code should update the models.json file with the new model.