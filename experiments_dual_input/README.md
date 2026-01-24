# Dual Input Type Experiment

This experiment extends the standard denoising setup to support two input types with separate dimensions and lambda values.

## Features

1. **Dual Input Dimensions**:
   - `d_1`: Dimension of input type 1 (coordinates 0..d_1-1)
   - `d_2`: Dimension of input type 2 (coordinates d_1..d_1+d_2-1)
   - Total input dimension: `d_total = d_1 + d_2`

2. **Dual Lambda Loss**:
   - `lambda_1`: Weight for active position loss in type 1
   - `lambda_2`: Weight for active position loss in type 2
   - Loss applies different weights to each input type's coordinate range

3. **Sparse Signal Generation**:
   - Sparse signal is chosen randomly across the full `d_1 + d_2` dimensional space
   - No restriction on which type gets active coordinates

4. **Three-Plot Visualization**:
   - **Blue plot**: Encoder-decoder pairs for type 1 (coordinates 0..d_1-1)
   - **Red plot**: Encoder-decoder pairs for type 2 (coordinates d_1..d_1+d_2-1)
   - **Overlay plot**: Both types on same axes with shared limits

## Usage

### Basic Usage

```bash
python experiments_dual_input/train_dual_input.py experiments_dual_input/config_dual_input.json
```

### Configuration Parameters

#### Data Configuration

- `d_1` (int): Dimension of input type 1
- `d_2` (int): Dimension of input type 2
- `sparsity` (int): Total number of active coordinates across the full `d_1 + d_2` space
- `noise_variance` (float): Variance of Gaussian noise
- `n_train` (int): Number of training samples
- `n_val` (int): Number of validation samples
- `seed` (int): Random seed
- `device` (str): Device to use ("cuda" or "cpu")

#### Loss Configuration

- `loss_type`: Must be `"dual_lambda_scaled_mse"`
- `lambda_1` (float): Weight for active position loss in type 1 (coordinates 0..d_1-1)
- `lambda_2` (float): Weight for active position loss in type 2 (coordinates d_1..d_1+d_2-1)

#### Model Configuration

- `hidden_size` (int): Hidden layer dimension
- `encoder_initialization_scale` (float): Scale factor for encoder weight initialization
- `decoder_initialization_scale` (float): Scale factor for decoder weight initialization

## Architecture

- **Input**: `(B, d_1 + d_2)` - Combined input from both types
- **Encoder**: `fc1` maps `(d_1 + d_2) -> hidden_size`
- **Decoder**: `fc2` maps `hidden_size -> (d_1 + d_2)`
- **Output**: `(B, d_1 + d_2)` - Reconstructed signal

## Loss Function

The dual lambda scaled MSE loss computes:

```
loss = lambda_1 * loss_on_type1 + lambda_2 * loss_on_type2 + loss_off
```

Where:
- `loss_on_type1`: MSE on active coordinates in range [0, d_1-1]
- `loss_on_type2`: MSE on active coordinates in range [d_1, d_1+d_2-1]
- `loss_off`: MSE on inactive coordinates (both types)

## Visualization

The experiment generates three plots at each epoch:

1. **Type 1 (Blue)**: Shows encoder-decoder weight pairs for coordinates 0..d_1-1
2. **Type 2 (Red)**: Shows encoder-decoder weight pairs for coordinates d_1..d_1+d_2-1
3. **Overlay**: Both types on the same axes with distinct colors

All plots use the same axis limits computed from both types for easy comparison.

## Example Configuration

See `config_dual_input.json` for a complete example configuration.

## Files

- `train_dual_input.py`: Main training script
- `config_dual_input.json`: Example configuration file
- `README.md`: This documentation

## Dependencies

Uses the same dependencies as the main project:
- PyTorch
- NumPy
- Matplotlib
- Standard `mft_denoising` modules

## Notes

- The sparse signal generation randomly selects `sparsity` coordinates from the full `d_1 + d_2` space
- There's no guarantee about how many active coordinates fall in each type's range
- The model learns to handle both input types simultaneously with different loss weights
- Visualization helps understand how the model represents each input type differently
