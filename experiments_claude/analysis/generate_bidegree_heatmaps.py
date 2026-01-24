"""
Generate hidden_size bidegree heatmaps for all checkpoints in an experiment.

Loads encoder weights from .npz checkpoint files and generates only the
hidden_size node bidegree heatmap for each checkpoint.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mft_denoising.diagnostics import analyze_encoder_graph


def load_encoder_weights_from_npz(npz_path: Path) -> np.ndarray:
    """
    Load encoder weights from .npz checkpoint file.
    
    The .npz file contains flattened encoder_pairs, which we reshape back
    to the original (hidden_size, input_size) shape.
    
    Args:
        npz_path: Path to .npz file
        
    Returns:
        encoder_weights: numpy array of shape (hidden_size, input_size)
    """
    data = np.load(npz_path)
    encoder_pairs = data['encoder_pairs']  # Flattened array
    hidden_size = int(data['hidden_size'])
    input_size = int(data['input_size'])
    
    # Reshape to original encoder weight matrix
    # encoder_pairs is stored in row-major order: encoder[i,j] for i in range(hidden_size), j in range(input_size)
    encoder_weights = encoder_pairs.reshape(hidden_size, input_size)
    
    return encoder_weights


def generate_hidden_size_bidegree_heatmap(
    encoder_weights: np.ndarray,
    threshold: float = 0.5,
    save_path: Optional[Path] = None
) -> None:
    """
    Generate only the hidden_size bidegree heatmap from encoder weights.
    
    Args:
        encoder_weights: numpy array of shape (hidden_size, d)
        threshold: Threshold for rounding weights
        save_path: Path to save the heatmap
    """
    # Run graph analysis
    results = analyze_encoder_graph(encoder_weights, threshold=threshold)
    
    # Extract hidden_size bidegree heatmap
    right_bidegree_heatmap = results["right_bidegree_heatmap"]
    d, hidden_size = results["G_dir"].shape
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(right_bidegree_heatmap, cmap='plasma', aspect='auto', origin='lower')
    ax.set_xlabel('Out-Degree', fontsize=12)
    ax.set_ylabel('In-Degree', fontsize=12)
    ax.set_title(f'hidden_size Node Bidegree Heatmap (directed, hidden_size={hidden_size}, threshold={threshold})',
                 fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Count')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()


def main():
    """Process all checkpoints in the experiment directory."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate hidden_size bidegree heatmaps for all checkpoints"
    )
    parser.add_argument(
        "experiment_dir",
        type=str,
        help="Path to experiment directory containing encoder_decoder_pairs_epoch_*.npz files"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for rounding weights (default: 0.5)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for heatmaps (default: experiment_dir/bidegree_heatmaps)"
    )
    
    args = parser.parse_args()
    
    experiment_dir = Path(args.experiment_dir)
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = experiment_dir / "bidegree_heatmaps"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all checkpoint files
    pattern = "encoder_decoder_pairs_epoch_*.npz"
    checkpoint_files = sorted(experiment_dir.glob(pattern))
    
    if not checkpoint_files:
        print(f"No {pattern} files found in {experiment_dir}")
        return
    
    print(f"Found {len(checkpoint_files)} checkpoint files")
    print(f"Output directory: {output_dir}")
    print(f"Threshold: {args.threshold}\n")
    
    # Process each checkpoint
    for checkpoint_file in checkpoint_files:
        # Extract epoch number from filename
        epoch_str = checkpoint_file.stem.replace("encoder_decoder_pairs_epoch_", "")
        epoch_num = int(epoch_str)
        
        print(f"Processing epoch {epoch_num}...", end=" ")
        
        try:
            # Load encoder weights
            encoder_weights = load_encoder_weights_from_npz(checkpoint_file)
            
            # Generate heatmap
            output_path = output_dir / f"hidden_size_bidegree_epoch_{epoch_num:04d}.png"
            generate_hidden_size_bidegree_heatmap(
                encoder_weights,
                threshold=args.threshold,
                save_path=output_path
            )
            
        except Exception as e:
            print(f"ERROR: {e}")
            continue
    
    print(f"\nCompleted! Generated {len(checkpoint_files)} heatmaps in {output_dir}")


if __name__ == "__main__":
    main()
