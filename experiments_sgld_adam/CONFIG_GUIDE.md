# Configuration Parameter Dependencies Guide

This guide explains the parameter dependencies in the two-stage SGLD-Adam training configuration.

## Parameter Hierarchy

### Epoch Parameters

1. **`sgld_epochs`** (Stage 1 epochs)
   - **Primary parameter** for Stage 1 training
   - If `null` or not set → falls back to `epochs`
   - **Recommended**: Set explicitly

2. **`adam_epochs`** (Stage 2 epochs)
   - **Primary parameter** for Stage 2 training
   - If `null` or not set → falls back to `epochs`
   - **Recommended**: Set explicitly

3. **`epochs`** (Legacy/fallback)
   - Used only as fallback if `sgld_epochs` or `adam_epochs` are `null`
   - **Not directly used** in two-stage training
   - Kept for backward compatibility

### Learning Rate Parameters

1. **`sgld_learning_rate`** (Stage 1 learning rate)
   - **Primary parameter** for Stage 1 (SGLD)
   - If `null` or not set → falls back to `learning_rate`
   - **Recommended**: Set explicitly

2. **`adam_learning_rate`** (Stage 2 learning rate)
   - **Primary parameter** for Stage 2 (Adam)
   - If `null` or not set → falls back to `sgld_learning_rate`, then `learning_rate`
   - **Recommended**: Set explicitly

3. **`learning_rate`** (Legacy/fallback)
   - Used only as fallback if stage-specific rates are `null`
   - **Not directly used** in two-stage training
   - Kept for backward compatibility

### Temperature Parameters

1. **`sgld_temperature`** (Stage 1 temperature)
   - **Primary parameter** for Stage 1 (SGLD) noise
   - If `null` or not set → falls back to `temperature`
   - **Recommended**: Set explicitly

2. **`temperature`** (Legacy/fallback)
   - Used only as fallback if `sgld_temperature` is `null`
   - **Not directly used** in two-stage training
   - Kept for backward compatibility

## Shared Parameters

These parameters are used by **both stages** and cannot be stage-specific:

- `batch_size`: Batch size for training
- `encoder_regularization`: L2 regularization for encoder weights
- `decoder_regularization`: L2 regularization for decoder weights
- `enable_warmup`: Enable learning rate warmup (only applies to Stage 2/Adam)
- `warmup_fraction`: Fraction of steps for warmup (only applies to Stage 2/Adam)

## Recommended Configuration Pattern

For clarity and to avoid confusion, **always set stage-specific parameters explicitly**:

```json
{
  "training": {
    "sgld_epochs": 50,           // Explicitly set
    "adam_epochs": 50,           // Explicitly set
    "sgld_learning_rate": 0.0001, // Explicitly set
    "sgld_temperature": 0.1,     // Explicitly set
    "adam_learning_rate": 0.0001, // Explicitly set
    
    // Legacy parameters (can be ignored or set to defaults)
    "epochs": 1,
    "learning_rate": 0.0001,
    "temperature": 0.0,
    
    // Shared parameters
    "batch_size": 128,
    "encoder_regularization": 0.0,
    "decoder_regularization": 0.0
  }
}
```

## Fallback Chain Summary

- **Stage 1 epochs**: `sgld_epochs` → `epochs`
- **Stage 1 learning rate**: `sgld_learning_rate` → `learning_rate`
- **Stage 1 temperature**: `sgld_temperature` → `temperature`
- **Stage 2 epochs**: `adam_epochs` → `epochs`
- **Stage 2 learning rate**: `adam_learning_rate` → `sgld_learning_rate` → `learning_rate`

## Example: Minimal Configuration

If you only set stage-specific parameters:

```json
{
  "training": {
    "sgld_epochs": 50,
    "adam_epochs": 50,
    "sgld_learning_rate": 0.0001,
    "sgld_temperature": 0.1,
    "adam_learning_rate": 0.0001,
    "batch_size": 128
  }
}
```

The legacy parameters (`epochs`, `learning_rate`, `temperature`) will use their default values from `TrainingConfig`, but they won't affect training since stage-specific parameters are set.
