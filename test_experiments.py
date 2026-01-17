"""
Test file for experiments.

Includes:
1. Small experiment to verify the system works
2. Hand-designed diagonal model test with specific datapoint logging
"""

import torch
import numpy as np
from pathlib import Path

from mft_denoising.data import create_dataloaders, DataConfig
from mft_denoising.nn import TwoLayerNet
from mft_denoising.config import (
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
    LossConfig,
)
from mft_denoising.losses import create_loss_function
from mft_denoising.optimizers import create_optimizer
from mft_denoising.experiment import ExperimentTracker
from main import train, evaluate


def create_diagonal_model(input_size: int, hidden_size: int) -> TwoLayerNet:
    """
    Create a model with diagonal encoder and decoder matrices.
    
    Args:
        input_size: Input dimension d
        hidden_size: Hidden layer dimension
    
    Returns:
        TwoLayerNet with:
        - Encoder diagonal: entries = 3, size = min(d, hidden_size)
        - Decoder diagonal: entries = 1, size = min(d, hidden_size)
    """
    model = TwoLayerNet(input_size=input_size, hidden_size=hidden_size)
    
    # Number of diagonal entries = min(d, hidden_size)
    n_diag = min(input_size, hidden_size)
    
    # Initialize encoder (fc1) to diagonal matrix
    with torch.no_grad():
        model.fc1.weight.zero_()
        model.fc1.bias.zero_()
        # Set first n_diag diagonal entries to 3
        for i in range(n_diag):
            model.fc1.weight[i, i] = 3.0
        
        # Initialize decoder (fc2) to diagonal matrix
        model.fc2.weight.zero_()
        model.fc2.bias.zero_()
        # Set first n_diag diagonal entries to 1
        for i in range(n_diag):
            model.fc2.weight[i, i] = 1.0
    
    return model


def sample_datapoints_with_first_coord_target(n_samples: int, data_dim: int, sparsity: int, 
                                               noise_variance: float, target_first_coord: int,
                                               device: str = "cpu") -> tuple:
    """
    Sample datapoints where the target x_star has a specific value at the first coordinate.
    
    Args:
        n_samples: Number of samples to generate
        data_dim: Dimension d
        sparsity: Number of active coordinates
        noise_variance: Variance of noise
        target_first_coord: Target value for first coordinate (0 or 1)
        device: Device to generate on
    
    Returns:
        (x, x_star) tuple where x_star[:, 0] == target_first_coord (approximately)
    """
    import math
    
    x_star = torch.zeros(n_samples, data_dim, device=device)
    
    for i in range(n_samples):
        # Ensure first coordinate is set to target_first_coord
        x_star[i, 0] = float(target_first_coord)
        
        # Fill remaining sparsity - 1 positions (if needed)
        if target_first_coord == 0:
            # Need sparsity more positions
            remaining_positions = sparsity
        else:
            # Already have first coordinate, need sparsity - 1 more
            remaining_positions = sparsity - 1
        
        # Choose remaining positions from indices 1 onwards
        if remaining_positions > 0:
            available_indices = list(range(1, data_dim))
            selected = torch.tensor(
                np.random.choice(available_indices, size=min(remaining_positions, len(available_indices)), replace=False),
                device=device
            )
            x_star[i, selected] = 1.0
    
    # Add noise
    noise = torch.randn(n_samples, data_dim, device=device) * math.sqrt(noise_variance)
    x = x_star + noise
    
    return x, x_star


def test_small_experiment():
    """Run a small experiment to verify the system works."""
    print("=" * 80)
    print("TEST 1: Small Experiment")
    print("=" * 80)
    
    # Create small experiment configuration
    config = ExperimentConfig(
        model=ModelConfig(
            hidden_size=16,
            encoder_initialization_scale=1.0,
            decoder_initialization_scale=1.0,
        ),
        training=TrainingConfig(
            optimizer_type="sgld",
            learning_rate=1e-3,
            temperature=0.0,
            epochs=5,
            batch_size=32,
            encoder_regularization=0.0,
            decoder_regularization=0.0,
        ),
        loss=LossConfig(
            loss_type="scaled_mse",
            lambda_on=10.0,
        ),
        data=DataConfig(
            d=32,
            sparsity=2,
            noise_variance=0.1,
            n_train=1000,  # Small training set
            n_val=100,
            seed=42,
        ),
        experiment_name="test_small",
        output_dir="experiments/test_small",
        save_model=False,
        save_plots=False,
    )
    
    # Create dataloaders
    train_loader, test_loader = create_dataloaders(config.data, batch_size=config.training.batch_size)
    
    # Create model
    model = TwoLayerNet(
        input_size=config.data.d,
        hidden_size=config.model.hidden_size,
        encoder_initialization_scale=config.model.encoder_initialization_scale,
        decoder_initialization_scale=config.model.decoder_initialization_scale,
    )
    
    # Create tracker
    tracker = ExperimentTracker(config)
    
    print(f"Training with {config.training.optimizer_type.upper()}...")
    print(f"Training samples: {config.data.n_train}, Epochs: {config.training.epochs}")
    
    # Train
    model = train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        config=config,
        tracker=tracker,
    )
    
    # Save results
    tracker.save_results()
    
    print("✓ Small experiment completed successfully!")
    print()


def test_diagonal_model():
    """Test hand-designed diagonal model on large sample with loss diagnostics."""
    print("=" * 80)
    print("TEST 2: Diagonal Model Test")
    print("=" * 80)
    
    # Configuration
    data_dim = 32
    hidden_size = 16
    sparsity = 2
    noise_variance = 0.1
    lambda_on = 10.0
    encoder_reg = 0.0
    decoder_reg = 0.0
    n_samples = 100000
    batch_size = 1000  # Process in batches for efficiency
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create diagonal model
    model = create_diagonal_model(data_dim, hidden_size)
    model = model.to(device)
    model.eval()
    
    print(f"Model created: d={data_dim}, hidden_size={hidden_size}")
    n_diag = min(data_dim, hidden_size)
    print(f"Diagonal size: {n_diag} (encoder=3, decoder=1)")
    print()
    
    # Create loss function
    loss_config = LossConfig(loss_type="scaled_mse", lambda_on=lambda_on)
    loss_fn = create_loss_function(loss_config)
    
    # Regularization terms (constant across all samples)
    encoder_l2 = torch.sum(model.fc1.weight ** 2)
    decoder_l2 = torch.sum(model.fc2.weight ** 2)
    reg_loss = encoder_reg * encoder_l2 + decoder_reg * decoder_l2
    
    # Generate a single random test input-output pair for demonstration
    print("Example Input-Output Pair:")
    print("-" * 80)
    import math
    # Sample a single random datapoint
    idx = torch.randperm(data_dim, device=device)[:sparsity]
    x_star_example = torch.zeros(1, data_dim, device=device)
    x_star_example[0, idx] = 1.0
    noise_example = torch.randn(1, data_dim, device=device) * math.sqrt(noise_variance)
    x_example = x_star_example + noise_example
    
    with torch.no_grad():
        output_example = model(x_example)
    
    print(f"Input x (noisy):     {x_example[0].cpu().numpy()}")
    print(f"Target x_star:       {x_star_example[0].cpu().numpy()}")
    print(f"Model output:        {output_example[0].cpu().numpy()}")
    print(f"Absolute difference: {torch.abs(output_example[0] - x_star_example[0]).cpu().numpy()}")
    print()
    
    # Generate samples using TwoHotStream for efficiency
    from mft_denoising.data import TwoHotStream
    data_cfg = DataConfig(
        d=data_dim,
        sparsity=sparsity,
        noise_variance=noise_variance,
        n_train=n_samples,
        n_val=0,
        seed=42,
        device=device,
    )
    stream = TwoHotStream(data_cfg)
    
    print(f"Generating {n_samples} samples...")
    
    # Accumulate losses
    total_data_loss = 0.0
    total_loss_on = 0.0
    total_loss_off = 0.0
    total_loss_all = 0.0
    n_processed = 0
    
    # Process in batches
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for batch_idx in range(n_batches):
            current_batch_size = min(batch_size, n_samples - n_processed)
            x, x_star = stream.sample_batch(current_batch_size)
            
            # Forward pass
            output = model(x)
            
            # Compute losses for the batch
            data_loss, loss_on, loss_off = loss_fn(output, x_star)
            
            # Total loss (data loss + regularization)
            total_loss = data_loss + reg_loss
            
            # Accumulate
            total_data_loss += data_loss.item() * current_batch_size
            total_loss_on += loss_on.item() * current_batch_size
            total_loss_off += loss_off.item() * current_batch_size
            total_loss_all += total_loss.item() * current_batch_size
            
            n_processed += current_batch_size
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {n_processed}/{n_samples} samples...")
    
    # Compute averages
    avg_data_loss = total_data_loss / n_samples
    avg_loss_on = total_loss_on / n_samples
    avg_loss_off = total_loss_off / n_samples
    avg_total_loss = total_loss_all / n_samples
    
    print()
    print("Loss Diagnostics (averaged over {} samples):".format(n_samples))
    print("=" * 80)
    print(f"Data loss (scaled MSE): {avg_data_loss:.6f}")
    print(f"  - Loss on (weighted): {avg_loss_on:.6f}")
    print(f"  - Loss off: {avg_loss_off:.6f}")
    print(f"Regularization:")
    print(f"  - Encoder L2: {encoder_reg * encoder_l2.item():.6f}")
    print(f"  - Decoder L2: {decoder_reg * decoder_l2.item():.6f}")
    print(f"  - Total regularization: {reg_loss.item():.6f}")
    print(f"Total loss (data + regularization): {avg_total_loss:.6f}")
    print()
    
    print("✓ Diagonal model test completed successfully!")
    print()


if __name__ == "__main__":
    # Run both tests
    test_small_experiment()
    test_diagonal_model()
    
    print("=" * 80)
    print("All tests completed!")
    print("=" * 80)
