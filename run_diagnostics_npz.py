#!/usr/bin/env python3
"""
Run MFT denoising diagnostics on a saved encoder_decoder_pairs.npz file.

This script:
1. Loads encoder-decoder weight pairs from a .npz file
2. Reconstructs the full weight matrices
3. Runs blob formation diagnostics
4. Runs graph analysis diagnostics
5. Saves histograms

Usage:
    python run_diagnostics_npz.py <path_to_encoder_decoder_pairs.npz> [--threshold THRESHOLD]
"""

import argparse
import numpy as np
from pathlib import Path
import sys
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from mft_denoising.diagnostics import (
    compute_lightweight_blob_metrics,
    analyze_encoder_graph,
    save_graph_histograms,
    get_all_weight_pairs,
    compute_weight_statistics,
    compute_lightweight_clustering
)
import torch
from mft_denoising.nn import TwoLayerNet


def load_weights_from_npz(npz_path: Path):
    """
    Load encoder and decoder weights from a .npz file.
    
    The .npz file contains:
    - encoder_pairs: flattened array of encoder weights
    - decoder_pairs: flattened array of decoder weights
    - hidden_size: hidden layer size
    - input_size: input dimension (d)
    
    Returns:
        (encoder_weights, decoder_weights) as numpy arrays
        encoder_weights: shape (hidden_size, input_size)
        decoder_weights: shape (input_size, hidden_size)
    """
    data = np.load(npz_path)
    
    encoder_pairs = data['encoder_pairs']
    decoder_pairs = data['decoder_pairs']
    hidden_size = int(data['hidden_size'])
    input_size = int(data['input_size'])
    
    # Reconstruct full weight matrices
    # encoder_pairs is stored row-major: [enc[0,0], enc[0,1], ..., enc[0,d-1], enc[1,0], ...]
    encoder_weights = encoder_pairs.reshape(hidden_size, input_size)
    
    # decoder_pairs is stored as: [dec[0,0], dec[1,0], ..., dec[d-1,0], dec[0,1], ...]
    # This is column-major order, so we reshape to (input_size, hidden_size) and transpose
    # Actually, looking at save_encoder_decoder_pairs, it does:
    # for i in range(hidden_size):
    #     for j in range(input_size):
    #         encoder_pairs.append(encoder_weights[i, j])
    #         decoder_pairs.append(decoder_weights[j, i])
    # So decoder_pairs is: [dec[0,0], dec[1,0], ..., dec[d-1,0], dec[0,1], dec[1,1], ...]
    # This is column-major, so reshape to (hidden_size, input_size) then transpose
    decoder_weights_reshaped = decoder_pairs.reshape(hidden_size, input_size)
    decoder_weights = decoder_weights_reshaped.T  # Now (input_size, hidden_size)
    
    return encoder_weights, decoder_weights


def create_dummy_model(encoder_weights: np.ndarray, decoder_weights: np.ndarray) -> torch.nn.Module:
    """
    Create a dummy TwoLayerNet model with the given weights.
    
    This is needed because compute_lightweight_blob_metrics expects a model object.
    """
    hidden_size, input_size = encoder_weights.shape
    
    model = TwoLayerNet(
        input_size=input_size,
        hidden_size=hidden_size,
        encoder_initialization_scale=1.0,
        decoder_initialization_scale=1.0,
    )
    
    # Set the weights
    model.fc1.weight.data = torch.from_numpy(encoder_weights)
    model.fc2.weight.data = torch.from_numpy(decoder_weights)
    
    return model


def run_diagnostics(npz_path: Path, threshold: float = 0.5, output_dir: Path = None):
    """
    Run full diagnostics on encoder-decoder pairs from .npz file.
    
    Args:
        npz_path: Path to .npz file
        threshold: Threshold for graph analysis
        output_dir: Directory to save results (defaults to same directory as .npz file)
    
    Returns:
        Dictionary with all diagnostic results
    """
    print(f"Loading weights from: {npz_path}")
    
    # Load weights
    encoder_weights, decoder_weights = load_weights_from_npz(npz_path)
    
    hidden_size, input_size = encoder_weights.shape
    print(f"  Encoder shape: {encoder_weights.shape}")
    print(f"  Decoder shape: {decoder_weights.shape}")
    print(f"  Total weight pairs: {hidden_size * input_size:,}")
    
    # Determine output directory
    if output_dir is None:
        output_dir = npz_path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Run blob formation diagnostics
    print("\n" + "=" * 80)
    print("BLOB FORMATION DIAGNOSTICS")
    print("=" * 80)
    
    # Create dummy model for blob metrics
    model = create_dummy_model(encoder_weights, decoder_weights)
    
    # Run blob metrics with sampling (to avoid memory issues)
    # For large models, use sampling instead of all pairs
    n_samples = min(50000, hidden_size * input_size)  # Sample up to 50K pairs
    print(f"Computing blob metrics (sampling {n_samples:,} weight pairs)...")
    blob_metrics = compute_lightweight_blob_metrics(
        model,
        compute_full_stats=False,  # Use sampling to avoid OOM
        n_samples=n_samples,
        eps=0.1,
        min_samples=50
    )
    
    print(f"  Weight correlation: {blob_metrics['weight_correlation']:.4f}")
    print(f"  Number of clusters: {blob_metrics['n_clusters_dbscan']}")
    print(f"  Silhouette score: {blob_metrics['silhouette_score']:.4f}" if blob_metrics['silhouette_score'] is not None else "  Silhouette score: None")
    print(f"  Encoder mean: {blob_metrics['encoder_mean']:.4f}, std: {blob_metrics['encoder_std']:.4f}")
    print(f"  Decoder mean: {blob_metrics['decoder_mean']:.4f}, std: {blob_metrics['decoder_std']:.4f}")
    print(f"  Computation time: {blob_metrics['computation_time_ms']:.2f} ms")
    
    # 2. Run graph analysis diagnostics
    print("\n" + "=" * 80)
    print("GRAPH ANALYSIS DIAGNOSTICS")
    print("=" * 80)
    
    print(f"Analyzing encoder graph (threshold={threshold})...")
    graph_results = analyze_encoder_graph(encoder_weights, threshold=threshold)
    
    print(f"  4-cycle count: {graph_results['four_cycle_count']:.2f}")
    
    # Extract degree statistics
    left_counts, left_bins = graph_results['left_degree_hist']
    right_counts, right_bins = graph_results['right_degree_hist']
    
    left_nonzero = left_counts > 0
    right_nonzero = right_counts > 0
    
    if left_nonzero.any():
        left_deg_range = (left_bins[:-1][left_nonzero].min(), left_bins[:-1][left_nonzero].max())
        print(f"  Left node (d) degree range: {left_deg_range[0]:.0f} - {left_deg_range[1]:.0f}")
    
    if right_nonzero.any():
        right_deg_range = (right_bins[:-1][right_nonzero].min(), right_bins[:-1][right_nonzero].max())
        print(f"  Right node (hidden_size) degree range: {right_deg_range[0]:.0f} - {right_deg_range[1]:.0f}")
    
    # Extract overlap statistics
    left_overlap_counts, left_overlap_bins = graph_results['left_overlap_hist']
    right_overlap_counts, right_overlap_bins = graph_results['right_overlap_hist']
    
    left_overlap_nonzero = left_overlap_counts > 0
    right_overlap_nonzero = right_overlap_counts > 0
    
    if left_overlap_nonzero.any():
        left_overlap_range = (left_overlap_bins[:-1][left_overlap_nonzero].min(), 
                             left_overlap_bins[:-1][left_overlap_nonzero].max())
        print(f"  Left overlap range: {left_overlap_range[0]:.2f} - {left_overlap_range[1]:.2f}")
    
    if right_overlap_nonzero.any():
        right_overlap_range = (right_overlap_bins[:-1][right_overlap_nonzero].min(),
                              right_overlap_bins[:-1][right_overlap_nonzero].max())
        print(f"  Right overlap range: {right_overlap_range[0]:.2f} - {right_overlap_range[1]:.2f}")
    
    # 3. Save histograms
    print("\n" + "=" * 80)
    print("SAVING HISTOGRAMS")
    print("=" * 80)
    
    histograms_dir = save_graph_histograms(graph_results, output_dir, threshold=threshold)
    print(f"Histograms saved to: {histograms_dir}")
    print(f"  - d_degree_histogram.png")
    print(f"  - hidden_size_degree_histogram.png")
    print(f"  - d_bidegree_heatmap.png")
    print(f"  - hidden_size_bidegree_heatmap.png")
    print(f"  - d_overlap_histogram.png")
    print(f"  - hidden_size_overlap_histogram.png")
    
    # 4. Save results to JSON
    results_json = {
        "source_file": str(npz_path),
        "threshold": threshold,
        "model_shape": {
            "input_size": int(input_size),
            "hidden_size": int(hidden_size)
        },
        "blob_metrics": {
            "weight_correlation": float(blob_metrics['weight_correlation']),
            "encoder_mean": float(blob_metrics['encoder_mean']),
            "encoder_std": float(blob_metrics['encoder_std']),
            "decoder_mean": float(blob_metrics['decoder_mean']),
            "decoder_std": float(blob_metrics['decoder_std']),
            "n_clusters_dbscan": int(blob_metrics['n_clusters_dbscan']),
            "silhouette_score": float(blob_metrics['silhouette_score']) if blob_metrics['silhouette_score'] is not None else None,
            "cluster_centers": blob_metrics['cluster_centers'],
            "n_noise_points": int(blob_metrics['n_noise_points']),
            "n_pairs_analyzed": int(blob_metrics['n_pairs_analyzed']),
            "computation_time_ms": float(blob_metrics['computation_time_ms'])
        },
        "graph_metrics": {
            "four_cycle_count": float(graph_results['four_cycle_count']),
            "left_degree_hist": {
                "counts": graph_results['left_degree_hist'][0].tolist(),
                "bins": graph_results['left_degree_hist'][1].tolist()
            },
            "right_degree_hist": {
                "counts": graph_results['right_degree_hist'][0].tolist(),
                "bins": graph_results['right_degree_hist'][1].tolist()
            },
            "left_overlap_hist": {
                "counts": graph_results['left_overlap_hist'][0].tolist(),
                "bins": graph_results['left_overlap_hist'][1].tolist()
            },
            "right_overlap_hist": {
                "counts": graph_results['right_overlap_hist'][0].tolist(),
                "bins": graph_results['right_overlap_hist'][1].tolist()
            },
            "left_bidegree_heatmap_shape": list(graph_results['left_bidegree_heatmap'].shape),
            "right_bidegree_heatmap_shape": list(graph_results['right_bidegree_heatmap'].shape)
        }
    }
    
    results_json_path = output_dir / "diagnostics_results.json"
    with open(results_json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\nResults saved to: {results_json_path}")
    
    return results_json


def main():
    parser = argparse.ArgumentParser(
        description="Run MFT denoising diagnostics on encoder_decoder_pairs.npz file"
    )
    parser.add_argument(
        "npz_path",
        type=str,
        help="Path to encoder_decoder_pairs.npz file"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for graph analysis (default: 0.5)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save results (default: same directory as .npz file)"
    )
    
    args = parser.parse_args()
    
    npz_path = Path(args.npz_path)
    if not npz_path.exists():
        print(f"Error: File not found: {npz_path}")
        return 1
    
    if not npz_path.suffix == '.npz':
        print(f"Warning: File does not have .npz extension: {npz_path}")
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    try:
        results = run_diagnostics(npz_path, threshold=args.threshold, output_dir=output_dir)
        print("\n" + "=" * 80)
        print("DIAGNOSTICS COMPLETE")
        print("=" * 80)
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
