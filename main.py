import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import sys
from pathlib import Path

from mft_denoising.data import create_dataloaders
from mft_denoising.nn import TwoLayerNet
from mft_denoising.config import (
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
    LossConfig,
)
from mft_denoising.data import DataConfig
from mft_denoising.losses import create_loss_function
from mft_denoising.optimizers import create_optimizer
from mft_denoising.experiment import ExperimentTracker



def train(
    model,
    train_loader,
    test_loader,
    config: ExperimentConfig,
    tracker: Optional[ExperimentTracker] = None,
):
    """
    Train model using configured optimizer (SGLD or ADAM) with configured loss function.
    
    Args:
        model: TwoLayerNet model
        train_loader: Training data loader
        test_loader: Test data loader
        config: Experiment configuration
        tracker: Optional experiment tracker for logging
    
    Returns:
        Trained model
    """
    # Validate device - fall back to CPU if CUDA is requested but not available
    device_str = config.data.device
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print(f"Warning: CUDA requested but not available. Falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)
    model = model.to(device)
    
    # Create optimizer based on config
    optimizer = create_optimizer(model, config.training)
    
    # Create loss function based on config
    loss_fn = create_loss_function(config.loss)
    
    # Start tracking if provided
    if tracker is not None:
        tracker.start()
    
    for epoch in range(config.training.epochs):
        model.train()
        total_train_loss = 0
        total_train_scaled_loss = 0
        total_train_loss_on_w = 0
        total_train_loss_off = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            
            # Compute loss using configured loss function
            total_scaled_loss, loss_on_w, loss_off = loss_fn(
                output=output,
                label=target
            )
            
            # Add L2 regularization for encoder and decoder
            encoder_l2 = torch.sum(model.fc1.weight ** 2)
            decoder_l2 = torch.sum(model.fc2.weight ** 2)
            
            # Total loss = scaled loss + regularization
            loss = (total_scaled_loss 
                   + config.training.encoder_regularization * encoder_l2 
                   + config.training.decoder_regularization * decoder_l2)
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            total_train_scaled_loss += total_scaled_loss.item()
            total_train_loss_on_w += loss_on_w.item()
            total_train_loss_off += loss_off.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch+1}, Batch: {batch_idx}, '
                      f'Loss: {loss.item():.6f}, On-Loss: {loss_on_w.item():.6f}, Off-Loss: {loss_off.item():.6f}')
              
        avg_loss = total_train_loss / len(train_loader)
        avg_scaled_loss = total_train_scaled_loss / len(train_loader)   
        avg_loss_on_w = total_train_loss_on_w / len(train_loader)
        avg_loss_off = total_train_loss_off / len(train_loader)
        
        # Evaluate
        test_metrics = evaluate(model, test_loader, loss_fn, device)
        test_scaled_loss = test_metrics["scaled_loss"]
        
        print(f'Epoch {epoch+1}/{config.training.epochs}: Train Loss: {avg_loss:.6f}, Train Scaled Loss: {avg_scaled_loss:.6f}, Train On-Loss: {avg_loss_on_w:.6f}, Train Off-Loss: {avg_loss_off:.6f}, Test Scaled Loss: {test_scaled_loss:.6f}')
        
        # Log epoch to tracker
        if tracker is not None:
            train_metrics = {
                "loss": avg_loss,
                "scaled_loss": avg_scaled_loss,
                "loss_on": avg_loss_on_w,
                "loss_off": avg_loss_off,
            }
            tracker.log_epoch(epoch + 1, train_metrics, test_metrics)
    
    return model


def evaluate(model, test_loader, loss_fn, device):
    """Evaluate model and return metrics"""
    model.eval()
    test_total_scaled_loss = 0
    test_loss_on = 0
    test_loss_off = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_scaled_loss, loss_on_w, loss_off = loss_fn(
                output=output,
                label=target
            )
            test_total_scaled_loss += total_scaled_loss.item()
            test_loss_on += loss_on_w.item()
            test_loss_off += loss_off.item()
    
    n_batches = len(test_loader)
    return {
        "scaled_loss": test_total_scaled_loss / n_batches,
        "loss_on": test_loss_on / n_batches,
        "loss_off": test_loss_off / n_batches,
    }


def plot_encoder_weights_histogram(model, bins=50, figsize=(10, 6), save_path=None):
    """
    Plot histogram of encoder (first layer) weights.
    
    Args:
        model: TwoLayerNet instance
        bins: Number of histogram bins
        figsize: Figure size (width, height)
        save_path: Optional path to save the figure
    
    Returns:
        fig, ax: matplotlib figure and axis objects
    """
    # Extract encoder weights (fc1.weight has shape: hidden_size x input_size)
    encoder_weights = model.fc1.weight.data.cpu().numpy().flatten()
    
    # Compute statistics
    mean = encoder_weights.mean()
    std = encoder_weights.std()
    median = np.median(encoder_weights)
    min_val = encoder_weights.min()
    max_val = encoder_weights.max()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot histogram
    n, bins_edges, patches = ax.hist(encoder_weights, bins=bins, density=True, 
                                      alpha=0.7, color='steelblue', edgecolor='black')
    
    # Check if there are weights with absolute value > 0.25 and adjust y-axis if needed
    threshold = 0.25
    large_weights_mask = np.abs(encoder_weights) > threshold
    ylim_adjusted = False
    
    if np.any(large_weights_mask):
        # Find bin centers
        bin_centers = (bins_edges[:-1] + bins_edges[1:]) / 2
        
        # Find bins corresponding to large weights
        large_weight_bins_mask = np.abs(bin_centers) > threshold
        
        if np.any(large_weight_bins_mask):
            # Find maximum density among large weight bins
            max_density_large = np.max(n[large_weight_bins_mask])
            
            # Set y-axis limit to focus on larger weights (with 10% margin)
            if max_density_large > 0:
                ax.set_ylim(0, max_density_large * 1.1)
                ylim_adjusted = True
    
    # Add vertical lines for mean and median
    ax.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean:.4f}')
    ax.axvline(median, color='green', linestyle='--', linewidth=2, label=f'Median: {median:.4f}')
    
    # Add labels and title
    ax.set_xlabel('Weight Value', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    title = 'Distribution of Encoder Weights (fc1)'
    if ylim_adjusted:
        title += '\n(Weights near 0 truncated for clarity)'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add interval count labels below x-axis
    interval_size = 0.2
    min_val_range = np.floor(min_val / interval_size) * interval_size
    max_val_range = np.ceil(max_val / interval_size) * interval_size
    intervals = np.arange(min_val_range, max_val_range + interval_size, interval_size)
    
    # Helper function to format count with 1-2 significant figures
    def format_count(n):
        if n < 100:
            return str(n)
        elif n < 1000:
            return str(round(n, -1))  # Round to nearest 10
        elif n < 10000:
            return f"{n/1000:.1f}k".rstrip('0').rstrip('.')
        else:
            return f"{n/1000:.0f}k"
    
    for i in range(len(intervals) - 1):
        interval_start = intervals[i]
        interval_end = intervals[i + 1]
        # Count values in this interval
        count = np.sum((encoder_weights >= interval_start) & (encoder_weights < interval_end))
        # Handle the last interval to include the endpoint
        if i == len(intervals) - 2:
            count = np.sum((encoder_weights >= interval_start) & (encoder_weights <= interval_end))
        
        if count > 0:
            interval_center = (interval_start + interval_end) / 2
            # Position label below x-axis (at y = -3% of y-range)
            y_pos = ax.get_ylim()[0] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.03
            ax.text(interval_center, y_pos, format_count(count), 
                   ha='center', va='top', fontsize=8, 
                   color='gray', rotation=0)
    
    # Add statistics text box
    stats_text = f'Statistics:\n'
    stats_text += f'Mean: {mean:.4f}\n'
    stats_text += f'Std: {std:.4f}\n'
    stats_text += f'Median: {median:.4f}\n'
    stats_text += f'Min: {min_val:.4f}\n'
    stats_text += f'Max: {max_val:.4f}\n'
    stats_text += f'Total weights: {len(encoder_weights):,}'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=9, family='monospace')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig, ax

def plot_network_outputs_histogram(model, data_loader=None, n_samples=1, bins=50, 
                                   figsize=(10, 6), save_path=None):
    """
    Plot histogram of neural network output values.
    
    Args:
        model: TwoLayerNet instance
        data_loader: DataLoader to sample from (if None, uses random inputs)
        n_samples: Number of samples to generate
        bins: Number of histogram bins
        figsize: Figure size (width, height)
        save_path: Optional path to save the figure
    
    Returns:
        fig, ax: matplotlib figure and axis objects
    """
    model.eval()
    device = next(model.parameters()).device
    
    outputs_list = []
    
    with torch.no_grad():
        if data_loader is not None:
            # Use data from data_loader
            samples_collected = 0
            for data, _ in data_loader:
                data = data.to(device)
                output = model(data)
                outputs_list.append(output.cpu().numpy())
                samples_collected += data.size(0)
                if samples_collected >= n_samples:
                    break
        else:
            # Generate random inputs
            input_size = model.fc1.in_features
            # Generate in batches for efficiency
            batch_size = 128
            n_batches = (n_samples + batch_size - 1) // batch_size
            
            for _ in range(n_batches):
                current_batch_size = min(batch_size, n_samples - len(outputs_list) * batch_size)
                random_input = torch.randn(current_batch_size, input_size, device=device)
                output = model(random_input)
                outputs_list.append(output.cpu().numpy())
    
    # Concatenate all outputs and flatten
    all_outputs = np.concatenate(outputs_list, axis=0).flatten()
    
    # Limit to n_samples if we collected more
    if len(all_outputs) > n_samples * model.fc2.out_features:
        all_outputs = all_outputs[:n_samples * model.fc2.out_features]
    
    # Compute statistics
    mean = all_outputs.mean()
    std = all_outputs.std()
    median = np.median(all_outputs)
    min_val = all_outputs.min()
    max_val = all_outputs.max()
    
    # Count values in different ranges (for sparse prediction analysis)
    near_zero = np.sum((all_outputs >= -0.1) & (all_outputs <= 0.1))
    near_one = np.sum((all_outputs >= 0.9) & (all_outputs <= 1.1))
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot histogram
    n, bins_edges, patches = ax.hist(all_outputs, bins=bins, density=True, 
                                      alpha=0.7, color='coral', edgecolor='black')
    
    # Check if there are outputs with absolute value > 0.25 and adjust y-axis if needed
    threshold = 0.25
    large_outputs_mask = np.abs(all_outputs) > threshold
    ylim_adjusted = False
    
    if np.any(large_outputs_mask):
        # Find bin centers
        bin_centers = (bins_edges[:-1] + bins_edges[1:]) / 2
        
        # Find bins corresponding to large outputs
        large_output_bins_mask = np.abs(bin_centers) > threshold
        
        if np.any(large_output_bins_mask):
            # Find maximum density among large output bins
            max_density_large = np.max(n[large_output_bins_mask])
            
            # Set y-axis limit to focus on larger outputs (with 10% margin)
            if max_density_large > 0:
                ax.set_ylim(0, max_density_large * 1.1)
                ylim_adjusted = True
    
    # Add vertical lines for mean, median, 0, and 1
    ax.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean:.4f}')
    ax.axvline(median, color='green', linestyle='--', linewidth=2, label=f'Median: {median:.4f}')
    ax.axvline(0, color='blue', linestyle=':', linewidth=2, alpha=0.5, label='Target: 0')
    ax.axvline(1, color='purple', linestyle=':', linewidth=2, alpha=0.5, label='Target: 1')
    
    # Add labels and title
    ax.set_xlabel('Output Value', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    title = 'Distribution of Neural Network Outputs'
    if ylim_adjusted:
        title += '\n(Outputs near 0 truncated for clarity)'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Add interval count labels below x-axis
    interval_size = 0.2
    min_val_range = np.floor(min_val / interval_size) * interval_size
    max_val_range = np.ceil(max_val / interval_size) * interval_size
    intervals = np.arange(min_val_range, max_val_range + interval_size, interval_size)
    
    # Helper function to format count with 1-2 significant figures
    def format_count(n):
        if n < 100:
            return str(n)
        elif n < 1000:
            return str(round(n, -1))  # Round to nearest 10
        elif n < 10000:
            return f"{n/1000:.1f}k".rstrip('0').rstrip('.')
        else:
            return f"{n/1000:.0f}k"
    
    for i in range(len(intervals) - 1):
        interval_start = intervals[i]
        interval_end = intervals[i + 1]
        # Count values in this interval
        count = np.sum((all_outputs >= interval_start) & (all_outputs < interval_end))
        # Handle the last interval to include the endpoint
        if i == len(intervals) - 2:
            count = np.sum((all_outputs >= interval_start) & (all_outputs <= interval_end))
        
        if count > 0:
            interval_center = (interval_start + interval_end) / 2
            # Position label below x-axis (at y = -3% of y-range)
            y_pos = ax.get_ylim()[0] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.03
            ax.text(interval_center, y_pos, format_count(count), 
                   ha='center', va='top', fontsize=8, 
                   color='gray', rotation=0)
    
    # Add statistics text box
    stats_text = f'Statistics:\n'
    stats_text += f'Mean: {mean:.4f}\n'
    stats_text += f'Std: {std:.4f}\n'
    stats_text += f'Median: {median:.4f}\n'
    stats_text += f'Min: {min_val:.4f}\n'
    stats_text += f'Max: {max_val:.4f}\n'
    stats_text += f'Total values: {len(all_outputs):,}\n'
    stats_text += f'Near 0 (±0.1): {100*near_zero/len(all_outputs):.1f}%\n'
    stats_text += f'Near 1 (±0.1): {100*near_one/len(all_outputs):.1f}%'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
            fontsize=9, family='monospace')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig, ax

def main():
    # Load config from JSON file if provided, otherwise use defaults
    if len(sys.argv) > 1:
        config_path = Path(sys.argv[1])
        if not config_path.exists():
            print(f"Error: Config file not found: {config_path}")
            sys.exit(1)
        config = ExperimentConfig.load_json(config_path)
        print(f"Loaded config from: {config_path}")
    else:
        # Default configuration (can be used if no config file is provided)
        config = ExperimentConfig(
            model=ModelConfig(
                hidden_size=128,
                encoder_initialization_scale=1.0,
                decoder_initialization_scale=1.0,
            ),
            training=TrainingConfig(
                optimizer_type="sgld",  # or "adam"
                learning_rate=1e-4,
                temperature=0.0,  # Only used for SGLD
                epochs=1,
                batch_size=128,
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
                n_train=1000000,
                n_val=1000,
                seed=42,
            ),
            experiment_name="experiment",
            output_dir=None,  # Auto-generate
            save_model=True,
            save_plots=True,
        )
        print("Using default configuration (provide a JSON config file as argument to use custom config)")
    
    # Create dataloaders
    train_loader, test_loader = create_dataloaders(config.data, batch_size=config.training.batch_size)
    
    # Create model
    model = TwoLayerNet(
        input_size=config.data.d,
        hidden_size=config.model.hidden_size,
        encoder_initialization_scale=config.model.encoder_initialization_scale,
        decoder_initialization_scale=config.model.decoder_initialization_scale,
    )
    
    # Validate device before training
    device_str = config.data.device
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print(f"Warning: CUDA requested but not available. Falling back to CPU.")
        device_str = "cpu"
        config.data.device = "cpu"  # Update config for consistency
    
    # Create experiment tracker
    tracker = ExperimentTracker(config)
    
    print(f"Training with {config.training.optimizer_type.upper()} on device: {device_str}")
    print(f"Learning rate: {config.training.learning_rate}, "
          f"Temperature: {config.training.temperature}, "
          f"Encoder regularization: {config.training.encoder_regularization}, "
          f"Decoder regularization: {config.training.decoder_regularization}")
    
    # Train
    model = train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        config=config,
        tracker=tracker,
    )
    
    print("\nTraining complete!")
    
    # Save plots
    if config.save_plots:
        encoder_plot_path = tracker.get_plot_path('encoder_weights_histogram.png')
        output_plot_path = tracker.get_plot_path('network_outputs_histogram.png')
        plot_encoder_weights_histogram(model=model, save_path=encoder_plot_path)
        plot_network_outputs_histogram(model=model, data_loader=test_loader, save_path=output_plot_path)
    
    # Save final results
    final_metrics = {}
    model_state = model.state_dict() if config.save_model else None
    tracker.save_results(final_metrics=final_metrics, model_state=model_state)

if __name__ == "__main__":
    main()