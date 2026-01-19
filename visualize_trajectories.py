"""
Visualize encoder-decoder pair trajectories across epochs.

Tracks specific weight pairs through training and plots them over scatter plots.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict
import json
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from mft_denoising.nn import TwoLayerNet
from mft_denoising.config import ExperimentConfig


def load_checkpoint(checkpoint_path: Path, config: ExperimentConfig) -> torch.nn.Module:
    """Load model from checkpoint."""
    model = TwoLayerNet(
        input_size=config.data.d,
        hidden_size=config.model.hidden_size,
        encoder_initialization_scale=1.0,  # Doesn't matter, we're loading weights
        decoder_initialization_scale=1.0,
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    return model


def sample_random_pairs(hidden_size: int, input_size: int, n_pairs: int = 6, seed: int = 42) -> List[Tuple[int, int]]:
    """
    Sample random encoder-decoder pairs.
    
    Returns list of (encoder_i, input_j) tuples such that we track:
    encoder[i, j] and decoder[j, i]
    """
    np.random.seed(seed)
    pairs = []
    for _ in range(n_pairs):
        i = np.random.randint(0, hidden_size)
        j = np.random.randint(0, input_size)
        pairs.append((i, j))
    return pairs


def extract_pair_values(model: torch.nn.Module, pairs: List[Tuple[int, int]]) -> List[Tuple[float, float]]:
    """Extract encoder and decoder values for given pairs."""
    encoder_weights = model.fc1.weight.data.cpu().numpy()  # (hidden_size, input_size)
    decoder_weights = model.fc2.weight.data.cpu().numpy()  # (input_size, hidden_size)
    
    values = []
    for i, j in pairs:
        enc_val = encoder_weights[i, j]
        dec_val = decoder_weights[j, i]
        values.append((enc_val, dec_val))
    return values


def plot_scatter_with_trajectories(
    experiment_dir: Path,
    config: ExperimentConfig,
    n_pairs: int = 6,
    seed: int = 42,
    save_path: Path = None
):
    """
    Load checkpoints and plot scatter plots with tracked trajectories.
    
    Args:
        experiment_dir: Directory containing checkpoints
        config: Experiment configuration
        n_pairs: Number of random pairs to track
        seed: Random seed for pair selection
        save_path: Path to save the figure
    """
    # Find all checkpoints
    checkpoint_files = sorted(experiment_dir.glob("checkpoint_epoch_*.pth"), 
                              key=lambda x: int(x.stem.split('_')[-1]))
    
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {experiment_dir}")
    
    print(f"Found {len(checkpoint_files)} checkpoints")
    
    # Sample random pairs (using first model to get dimensions)
    first_model = load_checkpoint(checkpoint_files[0], config)
    hidden_size = first_model.fc1.weight.shape[0]
    input_size = first_model.fc1.weight.shape[1]
    pairs = sample_random_pairs(hidden_size, input_size, n_pairs, seed)
    
    print(f"Tracking {n_pairs} pairs: {pairs}")
    
    # Collect trajectories across epochs
    trajectories: List[List[Tuple[float, float]]] = [[] for _ in range(n_pairs)]
    
    for checkpoint_path in checkpoint_files:
        model = load_checkpoint(checkpoint_path, config)
        pair_values = extract_pair_values(model, pairs)
        
        for pair_idx, (enc_val, dec_val) in enumerate(pair_values):
            trajectories[pair_idx].append((enc_val, dec_val))
    
    # Create figure with subplots for every epoch
    n_epochs = len(checkpoint_files)
    n_cols = 5
    n_rows = (n_epochs + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    if n_rows == 1:
        axes = axes if isinstance(axes, np.ndarray) else [axes]
    else:
        axes = axes.flatten()
    
    # Colors for each tracked pair
    colors = plt.cm.tab10(np.linspace(0, 1, n_pairs))
    
    # Determine axis limits from first epoch
    first_model = load_checkpoint(checkpoint_files[0], config)
    encoder_weights = first_model.fc1.weight.data.cpu().numpy()
    decoder_weights = first_model.fc2.weight.data.cpu().numpy()
    
    encoder_all = []
    decoder_all = []
    for i in range(hidden_size):
        for j in range(input_size):
            encoder_all.append(encoder_weights[i, j])
            decoder_all.append(decoder_weights[j, i])
    
    xlim = (min(encoder_all), max(encoder_all))
    ylim = (min(decoder_all), max(decoder_all))
    
    for epoch_idx, checkpoint_path in enumerate(checkpoint_files):
        ax = axes[epoch_idx]
        
        # Load model for this epoch
        model = load_checkpoint(checkpoint_path, config)
        
        # Extract all encoder-decoder pairs for scatter plot
        encoder_weights = model.fc1.weight.data.cpu().numpy()
        decoder_weights = model.fc2.weight.data.cpu().numpy()
        
        encoder_pairs = []
        decoder_pairs = []
        for i in range(hidden_size):
            for j in range(input_size):
                encoder_pairs.append(encoder_weights[i, j])
                decoder_pairs.append(decoder_weights[j, i])
        
        encoder_pairs = np.array(encoder_pairs)
        decoder_pairs = np.array(decoder_pairs)
        
        # Plot all pairs as background scatter
        ax.scatter(encoder_pairs, decoder_pairs, s=0.5, alpha=0.2, 
                  color='gray', rasterized=True)
        
        # Plot trajectories up to this epoch
        for pair_idx, color in enumerate(colors):
            traj = trajectories[pair_idx]
            if epoch_idx < len(traj):
                # Plot path up to current epoch
                if epoch_idx > 0:
                    traj_enc = [t[0] for t in traj[:epoch_idx + 1]]
                    traj_dec = [t[1] for t in traj[:epoch_idx + 1]]
                    ax.plot(traj_enc, traj_dec, color=color, alpha=0.6, linewidth=1.5, linestyle='--')
                
                # Highlight current position
                current_enc, current_dec = traj[epoch_idx]
                ax.scatter([current_enc], [current_dec], color=color, s=50, 
                          edgecolors='black', linewidths=1, zorder=10, alpha=0.8)
        
        # Extract epoch number from filename
        epoch_num = int(checkpoint_path.stem.split('_')[-1])
        ax.set_title(f'Epoch {epoch_num}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Encoder Weight', fontsize=8)
        ax.set_ylabel('Decoder Weight', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Set consistent axis limits
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    
    # Hide unused subplots
    for idx in range(len(checkpoint_files), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig, axes


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_trajectories.py <experiment_dir> [config_path]")
        print("Example: python visualize_trajectories.py experiments/my_experiment_20260117_123456 configs/my_config.json")
        sys.exit(1)
    
    experiment_dir = Path(sys.argv[1])
    if not experiment_dir.exists():
        print(f"Error: Experiment directory not found: {experiment_dir}")
        sys.exit(1)
    
    # Load config (try from experiment dir first, then from provided path)
    if len(sys.argv) >= 3:
        config_path = Path(sys.argv[2])
    else:
        config_path = experiment_dir / "config.json"
    
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    config = ExperimentConfig.load_json(config_path)
    
    # Create output path
    output_path = experiment_dir / "encoder_decoder_trajectories.png"
    
    print(f"Loading experiment from: {experiment_dir}")
    print(f"Config from: {config_path}")
    print(f"Output will be saved to: {output_path}")
    
    plot_scatter_with_trajectories(
        experiment_dir=experiment_dir,
        config=config,
        n_pairs=6,
        seed=42,
        save_path=output_path
    )
