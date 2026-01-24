# SGLD-Adam Two-Stage Training Experiment

This experiment implements a two-stage training approach: first training with SGLD (Stochastic Gradient Langevin Dynamics) for a fixed number of epochs, then switching to Adam optimization for additional epochs.

## Features

1. **Checkpoint Resumption**:
   - Resume training from a checkpoint of another experiment
   - Automatic architecture validation (checks hidden_size and input dimension d)
   - Clear error messages if architecture doesn't match
   - Set `resume_from_checkpoint` in config to path of `.pth` file

2. **Two-Stage Training**:
   - **Stage 1 (SGLD)**: Train for `sgld_epochs` epochs with fixed learning rate and temperature
   - **Stage 2 (Adam)**: Train for `adam_epochs` epochs with Adam optimizer
   - Both stages use the same loss function and share hyperparameters (batch size, regularization, etc.) by default

2. **Encoder Coordinate Tracking**: 
   - Tracks the count of weight pairs where encoder coordinate > 0.5 at each epoch
   - Stored in `encoder_above_0_5_count` in both train and test metrics
   - Saved to `results.json` in the training history

3. **Colored Marker Points**:
   - Randomly selects `colored_points_count` (default: 20) weight pairs at experiment start
   - These pairs are highlighted with distinct colors in visualization plots
   - Colored points do NOT affect axis scaling or trajectory tracking
   - Used as an "eyeball metric" for visual inspection

## Usage

### Basic Usage

```bash
python experiments_sgld_adam/train_sgld_adam.py experiments_sgld_adam/config_sgld_adam.json
```

### Configuration Parameters

The experiment uses the standard `ExperimentConfig` with additional two-stage specific parameters:

#### Two-Stage Training Parameters

- `sgld_epochs` (int, optional): Number of epochs for SGLD stage (t_1). Defaults to `epochs` if not specified.
- `adam_epochs` (int, optional): Number of epochs for Adam stage (t_2). Defaults to `epochs` if not specified.
- `sgld_learning_rate` (float, optional): Learning rate for SGLD stage. Defaults to `learning_rate` if not specified.
- `sgld_temperature` (float, optional): Temperature parameter for SGLD noise. Defaults to `temperature` if not specified.
- `adam_learning_rate` (float, optional): Learning rate for Adam stage. Defaults to `sgld_learning_rate` or `learning_rate` if not specified.
- `colored_points_count` (int): Number of distinct colored points to track (C). Default: 20.
- `resume_from_checkpoint` (str, optional): Path to checkpoint file (`.pth`) to resume training from. Architecture must match (hidden_size and d). If `null`, training starts from scratch.

#### Shared Hyperparameters

The following hyperparameters are shared between both stages:
- `batch_size`: Batch size for training
- `encoder_regularization`: L2 regularization for encoder weights
- `decoder_regularization`: L2 regularization for decoder weights
- Loss function configuration (from `loss` section)

### Example Configuration

See `config_sgld_adam.json` for a complete example configuration file.

### Resuming from Checkpoint

To resume training from a checkpoint:

```json
{
  "training": {
    "resume_from_checkpoint": "experiments/previous_experiment_20260121_120000/model.pth",
    "sgld_epochs": 50,
    "adam_epochs": 50,
    ...
  }
}
```

**Important**: The checkpoint architecture must match your current config:
- `hidden_size` must match
- `d` (input dimension) must match

If there's a mismatch, you'll get a clear error message showing what doesn't match.

## Output

The experiment generates the following outputs in the experiment directory:

1. **Training History** (`results.json`):
   - Per-epoch metrics including `encoder_above_0_5_count`
   - Training and test losses
   - Colored pairs information

2. **Visualizations**:
   - `encoder_decoder_pairs_epoch_XXXX.png`: Scatter plots with colored marker points
   - `encoder_weights_histogram.png`: Histogram of encoder weights
   - `network_outputs_histogram.png`: Histogram of network outputs

3. **Model Checkpoint** (`model.pth`): Final trained model state

## Implementation Details

### Training Flow

1. **Initialization**: Model and data loaders are created
2. **Stage 1 (SGLD)**:
   - Train for `sgld_epochs` epochs with SGLD optimizer
   - Fixed learning rate `sgld_lr` and temperature `sgld_temp`
   - Log metrics and save plots at each epoch
3. **Stage 2 (Adam)**:
   - Switch to Adam optimizer
   - Train for `adam_epochs` epochs with learning rate `adam_lr`
   - Optional learning rate warmup and cosine annealing
   - Log metrics and save plots at each epoch

### Colored Points

- Selected randomly at experiment start using the data seed
- Stored in `tracker.colored_pairs` as list of `(i, j)` tuples
- Visualized in plots with distinct colors from `tab20` colormap
- Do not affect axis limits (computed from all pairs first)
- No trajectory tracking (static markers only)

### Encoder Coordinate Tracking

- Computed at each epoch: `count = sum(encoder_weights > 0.5)`
- Added to both `train_metrics` and `test_metrics` as `encoder_above_0_5_count`
- Saved in training history for analysis

## Files

- `train_sgld_adam.py`: Main experiment script
- `config_sgld_adam.json`: Example configuration file
- `README.md`: This documentation

## Dependencies

Uses the same dependencies as the main project:
- PyTorch
- NumPy
- Matplotlib
- Standard `mft_denoising` modules

## Notes

- The experiment reuses existing training infrastructure from `main.py`
- Colored points are fixed at start and don't change during training
- Temperature parameter controls the noise level in SGLD (higher = more noise)
- Both stages can use different learning rates for fine-tuning
