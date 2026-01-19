"""
Visualize encoder-decoder pair trajectories across epochs as a video.

Tracks specific weight pairs through training and creates an animated video.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple
import sys
import imageio

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


def create_frame(
    model: torch.nn.Module,
    trajectories: List[List[Tuple[float, float]]],
    pairs: List[Tuple[int, int]],
    epoch_idx: int,
    epoch_num: int,
    hidden_size: int,
    input_size: int,
    colors: np.ndarray,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    figsize: Tuple[int, int] = (10, 8)
) -> np.ndarray:
    """Create a single frame for the video."""
    fig, ax = plt.subplots(figsize=figsize)
    
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
    
    ax.set_title(f'Epoch {epoch_num}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Encoder Weight', fontsize=12)
    ax.set_ylabel('Decoder Weight', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    plt.tight_layout()
    
    # Convert figure to numpy array
    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    
    return frame


def plot_scatter_with_trajectories_video(
    experiment_dir: Path,
    config: ExperimentConfig,
    n_pairs: int = 6,
    seed: int = 42,
    save_path: Path = None,
    fps: int = 10
):
    """
    Load checkpoints and create video animation of scatter plots with tracked trajectories.
    
    Args:
        experiment_dir: Directory containing checkpoints
        config: Experiment configuration
        n_pairs: Number of random pairs to track
        seed: Random seed for pair selection
        save_path: Path to save the video file
        fps: Frames per second for the video
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
    
    print("Loading all checkpoints and extracting trajectories...")
    for checkpoint_path in checkpoint_files:
        model = load_checkpoint(checkpoint_path, config)
        pair_values = extract_pair_values(model, pairs)
        
        for pair_idx, (enc_val, dec_val) in enumerate(pair_values):
            trajectories[pair_idx].append((enc_val, dec_val))
    
    # Determine axis limits from all epochs to ensure everything is visible
    print("Determining axis limits...")
    encoder_all = []
    decoder_all = []
    for checkpoint_path in checkpoint_files:
        model = load_checkpoint(checkpoint_path, config)
        encoder_weights = model.fc1.weight.data.cpu().numpy()
        decoder_weights = model.fc2.weight.data.cpu().numpy()
        for i in range(hidden_size):
            for j in range(input_size):
                encoder_all.append(encoder_weights[i, j])
                decoder_all.append(decoder_weights[j, i])
    
    xlim = (min(encoder_all), max(encoder_all))
    ylim = (min(decoder_all), max(decoder_all))
    
    # Colors for each tracked pair
    colors = plt.cm.tab10(np.linspace(0, 1, n_pairs))
    
    # Create frames
    print(f"Creating {len(checkpoint_files)} frames...")
    frames = []
    for epoch_idx, checkpoint_path in enumerate(checkpoint_files):
        if (epoch_idx + 1) % 10 == 0:
            print(f"  Processing epoch {epoch_idx + 1}/{len(checkpoint_files)}")
        
        model = load_checkpoint(checkpoint_path, config)
        epoch_num = int(checkpoint_path.stem.split('_')[-1])
        
        frame = create_frame(
            model=model,
            trajectories=trajectories,
            pairs=pairs,
            epoch_idx=epoch_idx,
            epoch_num=epoch_num,
            hidden_size=hidden_size,
            input_size=input_size,
            colors=colors,
            xlim=xlim,
            ylim=ylim
        )
        frames.append(frame)
    
    # Save video
    if save_path is None:
        save_path = experiment_dir / "encoder_decoder_trajectories.mp4"
    
    print(f"Saving video to: {save_path}")
    imageio.mimwrite(save_path, frames, fps=fps, codec='libx264', quality=8)
    print(f"Video saved! ({len(frames)} frames at {fps} fps)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_trajectories_video.py <experiment_dir> [config_path] [fps]")
        print("Example: python visualize_trajectories_video.py experiments/my_experiment_20260117_123456")
        print("         python visualize_trajectories_video.py experiments/my_experiment_20260117_123456 configs/my_config.json 15")
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
    
    # FPS (frames per second) - optional argument
    fps = int(sys.argv[3]) if len(sys.argv) >= 4 else 10
    
    print(f"Loading experiment from: {experiment_dir}")
    print(f"Config from: {config_path}")
    print(f"Video FPS: {fps}")
    
    plot_scatter_with_trajectories_video(
        experiment_dir=experiment_dir,
        config=config,
        n_pairs=6,
        seed=42,
        save_path=None,  # Will auto-generate path
        fps=fps
    )
