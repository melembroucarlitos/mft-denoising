"""
Lightweight real-time diagnostics for monitoring blob formation during training.

Provides efficient per-epoch metrics with <5% overhead by sampling weight pairs.
Reuses core logic from experiments_claude/analysis/measure_decoupling.py.
"""

import time
import numpy as np
import torch
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving files
import matplotlib.pyplot as plt

try:
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import silhouette_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def sample_weight_pairs(
    encoder_weights: np.ndarray,
    decoder_weights: np.ndarray,
    n_samples: int
) -> np.ndarray:
    """
    Efficiently sample random weight pairs without replacement.

    Pairs: encoder[i,j] with decoder[j,i] (path: input j -> hidden i -> output j)

    Args:
        encoder_weights: (hidden_size, d) numpy array
        decoder_weights: (d, hidden_size) numpy array
        n_samples: Number of pairs to sample

    Returns:
        weight_pairs: (n_samples, 2) array of [encoder, decoder] pairs

    Performance:
        - O(n_samples) time and memory vs O(hidden_size * d) for full extraction
        - For d=1024, hidden=512: sample 5K from 524K pairs (1% overhead)
    """
    hidden_size, d = encoder_weights.shape
    total_pairs = hidden_size * d

    # If requesting more samples than exist, return all pairs
    if n_samples >= total_pairs:
        return get_all_weight_pairs(encoder_weights, decoder_weights)

    # Random sampling without replacement
    indices = np.random.choice(total_pairs, n_samples, replace=False)

    # Convert flat indices to (i, j) coordinates
    i_indices = indices // d
    j_indices = indices % d

    # Extract pairs: encoder[i,j] and decoder[j,i]
    encoder_samples = encoder_weights[i_indices, j_indices]
    decoder_samples = decoder_weights[j_indices, i_indices]

    return np.column_stack([encoder_samples, decoder_samples])


def get_all_weight_pairs(
    encoder_weights: np.ndarray,
    decoder_weights: np.ndarray
) -> np.ndarray:
    """
    Extract all encoder-decoder weight pairs.

    Same logic as measure_decoupling.get_weight_pairs() but optimized.

    Args:
        encoder_weights: (hidden_size, d)
        decoder_weights: (d, hidden_size)

    Returns:
        weight_pairs: (hidden_size * d, 2)
    """
    hidden_size, d = encoder_weights.shape

    # Vectorized extraction (faster than nested loops)
    i_grid, j_grid = np.mgrid[0:hidden_size, 0:d]
    encoder_flat = encoder_weights[i_grid.ravel(), j_grid.ravel()]
    decoder_flat = decoder_weights[j_grid.ravel(), i_grid.ravel()]

    return np.column_stack([encoder_flat, decoder_flat])


def compute_lightweight_clustering(
    weight_pairs: np.ndarray,
    eps: float = 0.1,
    min_samples: int = 50
) -> Dict[str, Any]:
    """
    Fast DBSCAN clustering without GMM fitting (save GMM for post-training).

    Args:
        weight_pairs: (n_pairs, 2) array
        eps: DBSCAN epsilon parameter (radius)
        min_samples: DBSCAN minimum samples per cluster

    Returns:
        Dictionary with:
        - n_clusters_dbscan: Number of clusters found
        - silhouette_score: Clustering quality (None if <2 clusters)
        - cluster_centers: Top 3 cluster centroids [[enc, dec], ...]
        - n_noise_points: Number of noise points (-1 label)

    Performance:
        ~50ms for 5K samples (vs 300-500ms with GMM)
    """
    if not SKLEARN_AVAILABLE:
        return {
            "n_clusters_dbscan": 1,
            "silhouette_score": None,
            "cluster_centers": [[0.0, 0.0]],
            "n_noise_points": 0,
            "error": "scikit-learn not installed"
        }

    # DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(weight_pairs)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()

    # Compute silhouette score if we have enough structure
    silhouette = None
    if n_clusters >= 2 and n_noise < len(labels) * 0.5:
        non_noise_mask = labels != -1
        if non_noise_mask.sum() > 0:
            try:
                silhouette = silhouette_score(
                    weight_pairs[non_noise_mask],
                    labels[non_noise_mask]
                )
            except:
                silhouette = None

    # Extract cluster centers (top 3 by size)
    cluster_centers = []
    if n_clusters > 0:
        unique_labels = [l for l in set(labels) if l != -1]
        cluster_sizes = [(l, (labels == l).sum()) for l in unique_labels]
        cluster_sizes.sort(key=lambda x: x[1], reverse=True)

        for label, _ in cluster_sizes[:3]:  # Top 3 clusters
            cluster_mask = labels == label
            center = weight_pairs[cluster_mask].mean(axis=0)
            cluster_centers.append([float(center[0]), float(center[1])])

    # Fallback if no clusters found
    if not cluster_centers:
        cluster_centers = [[float(weight_pairs[:, 0].mean()),
                           float(weight_pairs[:, 1].mean())]]

    return {
        "n_clusters_dbscan": int(n_clusters),
        "silhouette_score": float(silhouette) if silhouette is not None else None,
        "cluster_centers": cluster_centers,
        "n_noise_points": int(n_noise)
    }


def compute_weight_statistics(weight_pairs: np.ndarray) -> Dict[str, float]:
    """
    Compute basic statistical metrics.

    Args:
        weight_pairs: (n_pairs, 2) array

    Returns:
        Dictionary with encoder/decoder statistics and correlation
    """
    encoder = weight_pairs[:, 0]
    decoder = weight_pairs[:, 1]

    correlation = np.corrcoef(encoder, decoder)[0, 1]

    return {
        "weight_correlation": float(correlation),
        "encoder_std": float(encoder.std()),
        "decoder_std": float(decoder.std()),
        "encoder_mean": float(encoder.mean()),
        "decoder_mean": float(decoder.mean())
    }


def compute_lightweight_blob_metrics(
    model: torch.nn.Module,
    n_samples: int = 5000,
    compute_full_stats: bool = False,
    eps: float = 0.1,
    min_samples: int = 50
) -> Dict[str, Any]:
    """
    Compute blob formation metrics on sampled weight pairs for real-time monitoring.

    This is the main function called by ExperimentTracker during training.

    Args:
        model: TwoLayerNet model with fc1 (encoder) and fc2 (decoder)
        n_samples: Number of weight pairs to sample (default: 5000)
                   Use smaller values for faster computation
                   Use larger values (or compute_full_stats=True) for more accuracy
        compute_full_stats: If True, use all pairs instead of sampling (slow)
        eps: DBSCAN epsilon parameter (cluster radius)
        min_samples: DBSCAN minimum samples per cluster

    Returns:
        Dictionary with metrics:
        {
            "weight_correlation": float,  # Pearson correlation
            "encoder_std": float,         # Weight magnitude spread
            "decoder_std": float,
            "encoder_mean": float,
            "decoder_mean": float,
            "n_clusters_dbscan": int,     # Number of blobs found
            "silhouette_score": float | None,  # Clustering quality
            "cluster_centers": List[List[float]],  # Top 3 cluster centers
            "n_noise_points": int,        # Noise points in DBSCAN
            "computation_time_ms": float, # Time taken
            "n_pairs_analyzed": int       # Actual number of pairs used
        }

    Performance:
        - With n_samples=5000: ~100ms per call (~0.5% of 20s epoch)
        - With compute_full_stats=True (d=1024, hidden=512): ~2-5s per call

    Example:
        >>> from mft_denoising.diagnostics import compute_lightweight_blob_metrics
        >>> metrics = compute_lightweight_blob_metrics(model, n_samples=5000)
        >>> print(f"Clusters: {metrics['n_clusters_dbscan']}")
        >>> print(f"Silhouette: {metrics['silhouette_score']}")
    """
    start_time = time.time()

    # Extract weights as numpy arrays
    encoder_weights = model.fc1.weight.data.cpu().numpy()  # (hidden_size, d)
    decoder_weights = model.fc2.weight.data.cpu().numpy()  # (d, hidden_size)

    # Sample or extract all pairs
    if compute_full_stats:
        weight_pairs = get_all_weight_pairs(encoder_weights, decoder_weights)
    else:
        weight_pairs = sample_weight_pairs(encoder_weights, decoder_weights, n_samples)

    # Compute statistical metrics
    stats = compute_weight_statistics(weight_pairs)

    # Compute clustering metrics
    clustering = compute_lightweight_clustering(weight_pairs, eps=eps, min_samples=min_samples)

    # Combine results
    metrics = {
        **stats,
        **clustering,
        "computation_time_ms": float((time.time() - start_time) * 1000),
        "n_pairs_analyzed": len(weight_pairs)
    }

    return metrics


def adaptive_sample_size(epoch: int, total_epochs: int, base_samples: int = 5000) -> int:
    """
    Optionally use adaptive sampling: more samples late in training when blobs stabilize.

    Args:
        epoch: Current epoch
        total_epochs: Total training epochs
        base_samples: Base number of samples

    Returns:
        Adjusted sample size

    Note:
        This is an optional enhancement. By default, use fixed base_samples.
    """
    progress = epoch / total_epochs

    if progress < 0.3:
        # Early training: minimal sampling
        return max(base_samples // 2, 2000)
    elif progress > 0.8:
        # Late training: more accurate sampling
        return base_samples * 2
    else:
        # Mid training: standard sampling
        return base_samples


# Convenience function for post-training full analysis
def full_blob_analysis(model: torch.nn.Module) -> Dict[str, Any]:
    """
    Comprehensive analysis using all weight pairs (no sampling).

    Use this post-training for final detailed metrics.
    Delegates to compute_lightweight_blob_metrics with compute_full_stats=True.

    Args:
        model: Trained model

    Returns:
        Full diagnostics dictionary
    """
    return compute_lightweight_blob_metrics(model, compute_full_stats=True)


def analyze_encoder_graph(
    encoder_weights: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Analyze encoder weights as a bipartite graph and compute graph-theoretic statistics.

    Converts encoder weights into a rounded bipartite graph representation where edges
    are created based on weight magnitude threshold. Computes degree distributions,
    bidegree heat maps, overlap histograms, and 4-cycle counts.

    Args:
        encoder_weights: numpy array of shape (hidden_size, d) - the encoder weight matrix E
        threshold: float (default 0.5) - threshold for rounding. Weights with |E[i,j]| < threshold
                   become 0, others become sign(E[i,j])

    Returns:
        Dictionary with:
        {
            "G_dir": np.ndarray,  # (d, hidden_size) directed graph
            "G_undir": np.ndarray,  # (d, hidden_size) undirected graph (absolute value)
            "left_degree_hist": Tuple[np.ndarray, np.ndarray],  # (counts, bins) for left nodes
            "right_degree_hist": Tuple[np.ndarray, np.ndarray],  # (counts, bins) for right nodes
            "left_bidegree_heatmap": np.ndarray,  # 2D array for (in_deg, out_deg) pairs
            "right_bidegree_heatmap": np.ndarray,  # 2D array for (in_deg, out_deg) pairs
            "left_overlap_hist": Tuple[np.ndarray, np.ndarray],  # (counts, bins) of GG^T off-diagonal
            "right_overlap_hist": Tuple[np.ndarray, np.ndarray],  # (counts, bins) of G^TG off-diagonal
            "four_cycle_count": float,  # Trace value Tr(G G^T G G^T)
        }

    Example:
        >>> from mft_denoising.diagnostics import analyze_encoder_graph
        >>> encoder = model.fc1.weight.data.cpu().numpy()
        >>> results = analyze_encoder_graph(encoder, threshold=0.5)
        >>> print(f"4-cycles: {results['four_cycle_count']}")
    """
    hidden_size, d = encoder_weights.shape

    # Step 1: Construct directed graph G_dir
    # E is (hidden_size, d), we want G_dir to be (d, hidden_size)
    # G_dir[j, i] = 0 if |E[i, j]| < threshold, else sign(E[i, j])
    E_abs = np.abs(encoder_weights)
    E_sign = np.sign(encoder_weights)
    G_dir = np.where(E_abs >= threshold, E_sign, 0.0)
    G_dir = G_dir.T  # Transpose to get (d, hidden_size) shape

    # Step 2: Construct undirected graph G_undir (absolute value)
    G_undir = np.abs(G_dir)

    # Step 3: Compute degree histograms
    # Left nodes (d): out-degree = number of non-zero entries per row
    # Right nodes (hidden_size): in-degree = number of non-zero entries per column
    left_out_degrees = (G_undir > 0).sum(axis=1).astype(int)  # (d,)
    right_in_degrees = (G_undir > 0).sum(axis=0).astype(int)  # (hidden_size,)

    # Create histograms with bins up to max degree (typically 20-30)
    max_left_degree = max(left_out_degrees.max(), 30) if len(left_out_degrees) > 0 else 30
    max_right_degree = max(right_in_degrees.max(), 30) if len(right_in_degrees) > 0 else 30
    
    left_counts, left_bins = np.histogram(left_out_degrees, bins=min(max_left_degree + 1, 31), range=(0, max_left_degree))
    right_counts, right_bins = np.histogram(right_in_degrees, bins=min(max_right_degree + 1, 31), range=(0, max_right_degree))

    # Step 4: Compute bidegree heat maps
    # For the directed graph G_dir, orientation is determined by sign:
    #   G_dir[j, i] = +1 means edge from left node j to right node i
    #   G_dir[j, i] = -1 means edge from right node i to left node j (reversed)
    #   G_dir[j, i] = 0 means no edge
    
    # Left nodes (d nodes):
    #   out-degree = count of +1 entries in row (edges going out to right nodes)
    #   in-degree = count of -1 entries in row (edges coming in from right nodes)
    left_out_degrees_dir = (G_dir > 0).sum(axis=1).astype(int)  # +1 entries per row
    left_in_degrees = (G_dir < 0).sum(axis=1).astype(int)  # -1 entries per row
    left_out_degrees_arr = left_out_degrees_dir
    
    # Right nodes (hidden_size nodes):
    #   in-degree = count of +1 entries in column (edges coming in from left nodes)
    #   out-degree = count of -1 entries in column (edges going out to left nodes)
    right_in_degrees_dir = (G_dir > 0).sum(axis=0).astype(int)  # +1 entries per column
    right_out_degrees = (G_dir < 0).sum(axis=0).astype(int)  # -1 entries per column
    right_in_degrees_arr = right_in_degrees_dir
    
    # Create 2D histograms for bidegree heat maps
    max_in_deg_left = int(max(left_in_degrees.max(), 30) if len(left_in_degrees) > 0 and left_in_degrees.max() > 0 else 30)
    max_out_deg_left = int(max(left_out_degrees_arr.max(), 30) if len(left_out_degrees_arr) > 0 and left_out_degrees_arr.max() > 0 else 30)
    max_in_deg_right = int(max(right_in_degrees_arr.max(), 30) if len(right_in_degrees_arr) > 0 and right_in_degrees_arr.max() > 0 else 30)
    max_out_deg_right = int(max(right_out_degrees.max(), 30) if len(right_out_degrees) > 0 and right_out_degrees.max() > 0 else 30)
    
    bins_left_in = int(min(max_in_deg_left + 1, 31))
    bins_left_out = int(min(max_out_deg_left + 1, 31))
    bins_right_in = int(min(max_in_deg_right + 1, 31))
    bins_right_out = int(min(max_out_deg_right + 1, 31))
    
    left_bidegree_heatmap, _, _ = np.histogram2d(
        left_in_degrees, left_out_degrees_arr,
        bins=(bins_left_in, bins_left_out),
        range=[[0, max_in_deg_left], [0, max_out_deg_left]]
    )
    
    right_bidegree_heatmap, _, _ = np.histogram2d(
        right_in_degrees_arr, right_out_degrees,
        bins=(bins_right_in, bins_right_out),
        range=[[0, max_in_deg_right], [0, max_out_deg_right]]
    )

    # Step 5: Compute overlap histograms
    # Left-left overlap: G_undir @ G_undir.T (shape (d, d))
    left_overlap = G_undir @ G_undir.T
    
    # Right-right overlap: G_undir.T @ G_undir (shape (hidden_size, hidden_size))
    right_overlap = G_undir.T @ G_undir
    
    # Extract off-diagonal entries (exclude self-overlaps)
    left_overlap_offdiag = left_overlap[np.triu_indices(d, k=1)]
    right_overlap_offdiag = right_overlap[np.triu_indices(hidden_size, k=1)]
    
    # Create histograms for overlap values
    if len(left_overlap_offdiag) > 0:
        left_overlap_max = max(left_overlap_offdiag.max(), 10) if len(left_overlap_offdiag) > 0 else 10
        left_overlap_counts, left_overlap_bins = np.histogram(
            left_overlap_offdiag,
            bins=min(int(left_overlap_max) + 1, 51),
            range=(0, left_overlap_max)
        )
    else:
        left_overlap_counts = np.array([0])
        left_overlap_bins = np.array([0, 1])
    
    if len(right_overlap_offdiag) > 0:
        right_overlap_max = max(right_overlap_offdiag.max(), 10) if len(right_overlap_offdiag) > 0 else 10
        right_overlap_counts, right_overlap_bins = np.histogram(
            right_overlap_offdiag,
            bins=min(int(right_overlap_max) + 1, 51),
            range=(0, right_overlap_max)
        )
    else:
        right_overlap_counts = np.array([0])
        right_overlap_bins = np.array([0, 1])

    # Step 6: Compute 4-cycle count
    # Tr(G_undir @ G_undir.T @ G_undir @ G_undir.T)
    GGT = G_undir @ G_undir.T  # (d, d)
    GGTG = GGT @ G_undir  # (d, hidden_size)
    GGTGGT = GGTG @ G_undir.T  # (d, d)
    four_cycle_count = float(np.trace(GGTGGT))

    return {
        "G_dir": G_dir,
        "G_undir": G_undir,
        "left_degree_hist": (left_counts, left_bins),
        "right_degree_hist": (right_counts, right_bins),
        "left_bidegree_heatmap": left_bidegree_heatmap,
        "right_bidegree_heatmap": right_bidegree_heatmap,
        "left_overlap_hist": (left_overlap_counts, left_overlap_bins),
        "right_overlap_hist": (right_overlap_counts, right_overlap_bins),
        "four_cycle_count": four_cycle_count,
    }


def analyze_encoder_graph_from_checkpoint(
    checkpoint_path: str | Path,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Load encoder weights from a checkpoint file and analyze as a bipartite graph.

    Convenience function that loads weights from a .pth checkpoint file and calls
    analyze_encoder_graph. Handles both direct paths and experiment directory paths.

    Args:
        checkpoint_path: Path to checkpoint file (.pth) or experiment directory containing model.pth
        threshold: float (default 0.5) - threshold for rounding weights

    Returns:
        Dictionary with graph analysis results (same as analyze_encoder_graph)

    Example:
        >>> from mft_denoising.diagnostics import analyze_encoder_graph_from_checkpoint
        >>> results = analyze_encoder_graph_from_checkpoint("experiments/my_experiment/model.pth")
        >>> print(f"4-cycles: {results['four_cycle_count']}")
        
        >>> # Or from experiment directory
        >>> results = analyze_encoder_graph_from_checkpoint("experiments/my_experiment/")
    """
    checkpoint_path = Path(checkpoint_path)
    
    # If it's a directory, look for model.pth inside
    if checkpoint_path.is_dir():
        model_path = checkpoint_path / "model.pth"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found in directory: {checkpoint_path}\n"
                f"Expected: {model_path}"
            )
    else:
        model_path = checkpoint_path
        if not model_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    # Load checkpoint
    state_dict = torch.load(model_path, map_location='cpu')
    
    # Handle compiled model checkpoints (strip _orig_mod. prefix if present)
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('_orig_mod.'):
                new_key = k[len('_orig_mod.'):]
                new_state_dict[new_key] = v
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict
    
    # Extract encoder weights
    if 'fc1.weight' not in state_dict:
        raise ValueError(
            f"Checkpoint does not contain 'fc1.weight' key.\n"
            f"Available keys: {list(state_dict.keys())}"
        )
    
    encoder_weights = state_dict['fc1.weight'].numpy()  # (hidden_size, d)
    
    # Run graph analysis
    return analyze_encoder_graph(encoder_weights, threshold=threshold)


def save_graph_histograms(
    results: Dict[str, Any],
    output_dir: Path,
    threshold: float = 0.5
) -> Path:
    """
    Generate and save histogram visualizations from graph analysis results.
    
    Creates a "graph_histograms" subfolder in the output directory and saves
    6 histogram plots as PNG files.
    
    Args:
        results: Dictionary from analyze_encoder_graph()
        output_dir: Directory where graph_histograms subfolder will be created
        threshold: Threshold used for analysis (for plot titles)
    
    Returns:
        Path to the graph_histograms directory
    """
    # Create graph_histograms subfolder
    histograms_dir = output_dir / "graph_histograms"
    histograms_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract dimensions from graph shapes
    d, hidden_size = results["G_dir"].shape  # G_dir is (d, hidden_size)
    
    # Extract data from results
    left_counts, left_bins = results["left_degree_hist"]  # d nodes (undirected)
    right_counts, right_bins = results["right_degree_hist"]  # hidden_size nodes (undirected)
    left_bidegree_heatmap = results["left_bidegree_heatmap"]  # d nodes (directed)
    right_bidegree_heatmap = results["right_bidegree_heatmap"]  # hidden_size nodes (directed)
    left_overlap_counts, left_overlap_bins = results["left_overlap_hist"]  # d nodes (undirected)
    right_overlap_counts, right_overlap_bins = results["right_overlap_hist"]  # hidden_size nodes (undirected)
    
    # 1. d node degree histogram (undirected graph)
    fig, ax = plt.subplots(figsize=(10, 6))
    bin_centers = (left_bins[:-1] + left_bins[1:]) / 2
    ax.bar(bin_centers, left_counts, width=left_bins[1] - left_bins[0], 
           alpha=0.7, color='steelblue', edgecolor='black')
    ax.set_xlabel('Out-Degree', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'd Node Out-Degree Distribution (undirected, d={d}, threshold={threshold})', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(histograms_dir / "d_degree_histogram.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. hidden_size node degree histogram (undirected graph)
    fig, ax = plt.subplots(figsize=(10, 6))
    bin_centers = (right_bins[:-1] + right_bins[1:]) / 2
    ax.bar(bin_centers, right_counts, width=right_bins[1] - right_bins[0],
           alpha=0.7, color='coral', edgecolor='black')
    ax.set_xlabel('In-Degree', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'hidden_size Node In-Degree Distribution (undirected, hidden_size={hidden_size}, threshold={threshold})',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(histograms_dir / "hidden_size_degree_histogram.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. d node bidegree heatmap (directed graph)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(left_bidegree_heatmap, cmap='viridis', aspect='auto', origin='lower')
    ax.set_xlabel('Out-Degree', fontsize=12)
    ax.set_ylabel('In-Degree', fontsize=12)
    ax.set_title(f'd Node Bidegree Heatmap (directed, d={d}, threshold={threshold})',
                 fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Count')
    plt.tight_layout()
    plt.savefig(histograms_dir / "d_bidegree_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. hidden_size node bidegree heatmap (directed graph)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(right_bidegree_heatmap, cmap='plasma', aspect='auto', origin='lower')
    ax.set_xlabel('Out-Degree', fontsize=12)
    ax.set_ylabel('In-Degree', fontsize=12)
    ax.set_title(f'hidden_size Node Bidegree Heatmap (directed, hidden_size={hidden_size}, threshold={threshold})',
                 fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Count')
    plt.tight_layout()
    plt.savefig(histograms_dir / "hidden_size_bidegree_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 5. d node overlap histogram (undirected graph)
    fig, ax = plt.subplots(figsize=(10, 6))
    if len(left_overlap_bins) > 1:
        bin_centers = (left_overlap_bins[:-1] + left_overlap_bins[1:]) / 2
        width = left_overlap_bins[1] - left_overlap_bins[0] if len(left_overlap_bins) > 1 else 1.0
        ax.bar(bin_centers, left_overlap_counts, width=width,
               alpha=0.7, color='mediumseagreen', edgecolor='black')
    ax.set_xlabel('Overlap Value', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'd-d Overlap Distribution (undirected G_undir, GG^T off-diagonal, d={d}, threshold={threshold})',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(histograms_dir / "d_overlap_histogram.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 6. hidden_size node overlap histogram (undirected graph)
    fig, ax = plt.subplots(figsize=(10, 6))
    if len(right_overlap_bins) > 1:
        bin_centers = (right_overlap_bins[:-1] + right_overlap_bins[1:]) / 2
        width = right_overlap_bins[1] - right_overlap_bins[0] if len(right_overlap_bins) > 1 else 1.0
        ax.bar(bin_centers, right_overlap_counts, width=width,
               alpha=0.7, color='mediumpurple', edgecolor='black')
    ax.set_xlabel('Overlap Value', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'hidden_size-hidden_size Overlap Distribution (undirected G_undir, G^TG off-diagonal, hidden_size={hidden_size}, threshold={threshold})',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(histograms_dir / "hidden_size_overlap_histogram.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return histograms_dir


if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(
        description="Analyze encoder weights as a bipartite graph from a checkpoint file"
    )
    parser.add_argument(
        "checkpoint_path",
        type=str,
        help="Path to checkpoint file (.pth) or experiment directory containing model.pth"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for rounding weights (default: 0.5)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save results as JSON file"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print only summary statistics (4-cycle count, degree ranges, etc.)"
    )
    parser.add_argument(
        "--save-histograms",
        action="store_true",
        help="Generate and save histogram visualizations to graph_histograms subfolder"
    )
    
    args = parser.parse_args()
    
    try:
        # Determine output directory for histograms
        checkpoint_path_obj = Path(args.checkpoint_path)
        if checkpoint_path_obj.is_dir():
            # If it's a directory (experiment directory), use it directly
            output_dir = checkpoint_path_obj
        else:
            # If it's a file, use its parent directory
            output_dir = checkpoint_path_obj.parent
        
        # Run analysis
        results = analyze_encoder_graph_from_checkpoint(
            args.checkpoint_path,
            threshold=args.threshold
        )
        
        # Prepare output (convert numpy arrays to lists for JSON serialization)
        output_dict = {
            "checkpoint_path": str(args.checkpoint_path),
            "threshold": args.threshold,
            "four_cycle_count": results["four_cycle_count"],
            "left_degree_hist": {
                "counts": results["left_degree_hist"][0].tolist(),
                "bins": results["left_degree_hist"][1].tolist()
            },
            "right_degree_hist": {
                "counts": results["right_degree_hist"][0].tolist(),
                "bins": results["right_degree_hist"][1].tolist()
            },
            "left_bidegree_heatmap_shape": results["left_bidegree_heatmap"].shape,
            "right_bidegree_heatmap_shape": results["right_bidegree_heatmap"].shape,
            "left_overlap_hist": {
                "counts": results["left_overlap_hist"][0].tolist(),
                "bins": results["left_overlap_hist"][1].tolist()
            },
            "right_overlap_hist": {
                "counts": results["right_overlap_hist"][0].tolist(),
                "bins": results["right_overlap_hist"][1].tolist()
            },
            "G_dir_shape": results["G_dir"].shape,
            "G_undir_shape": results["G_undir"].shape,
        }
        
        if args.summary:
            # Print summary
            print(f"Graph Analysis Summary")
            print(f"======================")
            print(f"Checkpoint: {args.checkpoint_path}")
            print(f"Threshold: {args.threshold}")
            print(f"Graph shape: {results['G_dir'].shape}")
            print(f"\n4-cycle count: {results['four_cycle_count']}")
            
            # Degree statistics
            left_counts, left_bins = results["left_degree_hist"]
            right_counts, right_bins = results["right_degree_hist"]
            
            left_nonzero = left_counts > 0
            right_nonzero = right_counts > 0
            
            if left_nonzero.any():
                left_deg_range = (left_bins[:-1][left_nonzero].min(), left_bins[:-1][left_nonzero].max())
                print(f"Left node degree range: {left_deg_range[0]:.0f} - {left_deg_range[1]:.0f}")
            
            if right_nonzero.any():
                right_deg_range = (right_bins[:-1][right_nonzero].min(), right_bins[:-1][right_nonzero].max())
                print(f"Right node degree range: {right_deg_range[0]:.0f} - {right_deg_range[1]:.0f}")
            
            # Overlap statistics
            left_overlap_counts, left_overlap_bins = results["left_overlap_hist"]
            right_overlap_counts, right_overlap_bins = results["right_overlap_hist"]
            
            left_overlap_nonzero = left_overlap_counts > 0
            right_overlap_nonzero = right_overlap_counts > 0
            
            if left_overlap_nonzero.any():
                left_overlap_range = (left_overlap_bins[:-1][left_overlap_nonzero].min(), 
                                     left_overlap_bins[:-1][left_overlap_nonzero].max())
                print(f"Left overlap range: {left_overlap_range[0]:.2f} - {left_overlap_range[1]:.2f}")
            
            if right_overlap_nonzero.any():
                right_overlap_range = (right_overlap_bins[:-1][right_overlap_nonzero].min(),
                                      right_overlap_bins[:-1][right_overlap_nonzero].max())
                print(f"Right overlap range: {right_overlap_range[0]:.2f} - {right_overlap_range[1]:.2f}")
        else:
            # Print full results
            print(json.dumps(output_dict, indent=2))
        
        # Save to file if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(output_dict, f, indent=2)
            print(f"\nResults saved to: {args.output}")
        
        # Generate histograms if requested
        if args.save_histograms:
            histograms_dir = save_graph_histograms(results, output_dir, threshold=args.threshold)
            print(f"\nHistograms saved to: {histograms_dir}")
            print(f"  - d_degree_histogram.png (undirected graph)")
            print(f"  - hidden_size_degree_histogram.png (undirected graph)")
            print(f"  - d_bidegree_heatmap.png (directed graph)")
            print(f"  - hidden_size_bidegree_heatmap.png (directed graph)")
            print(f"  - d_overlap_histogram.png (undirected graph)")
            print(f"  - hidden_size_overlap_histogram.png (undirected graph)")
    
    except Exception as e:
        print(f"Error: {e}", file=__import__('sys').stderr)
        exit(1)
