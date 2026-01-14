import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


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
    
    # Add vertical lines for mean and median
    ax.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean:.4f}')
    ax.axvline(median, color='green', linestyle='--', linewidth=2, label=f'Median: {median:.4f}')
    
    # Add labels and title
    ax.set_xlabel('Weight Value', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Distribution of Encoder Weights (fc1)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
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
    
    # Add vertical lines for mean, median, 0, and 1
    ax.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean:.4f}')
    ax.axvline(median, color='green', linestyle='--', linewidth=2, label=f'Median: {median:.4f}')
    ax.axvline(0, color='blue', linestyle=':', linewidth=2, alpha=0.5, label='Target: 0')
    ax.axvline(1, color='purple', linestyle=':', linewidth=2, alpha=0.5, label='Target: 1')
    
    # Add labels and title
    ax.set_xlabel('Output Value', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Distribution of Neural Network Outputs', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
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

