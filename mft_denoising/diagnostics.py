"""
Lightweight real-time diagnostics for monitoring blob formation during training.

Provides efficient per-epoch metrics with <5% overhead by sampling weight pairs.
Reuses core logic from experiments_claude/analysis/measure_decoupling.py.
"""

import time
import numpy as np
import torch
from typing import Dict, Any, Optional, Tuple, List

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
    eps: float = 0.05,
    min_samples: int = 25
) -> Dict[str, Any]:
    """
    Fast DBSCAN clustering without GMM fitting (save GMM for post-training).

    DBSCAN identifies "blobs" in weight pair space without assuming Gaussian
    distributions. This is faster than GMM fitting and robust to noise.

    Args:
        weight_pairs: (n_pairs, 2) array of [encoder, decoder] weight values
        eps: DBSCAN epsilon parameter (cluster radius)
             Points within eps distance are considered neighbors
             Typical value: 0.05 for normalized weights (updated from 0.1)
        min_samples: DBSCAN minimum samples per cluster
             Fewer points → noise (label = -1)
             Typical value: 25 for 5K samples (updated from 50)

    Returns:
        Dictionary with:
        - n_clusters_dbscan: Number of clusters found (excluding noise)
        - silhouette_score: Clustering quality in [-1, 1]
            > 0.7: Excellent separation
            0.5-0.7: Good separation
            < 0.5: Weak or overlapping clusters
            None: Single cluster or too much noise
        - cluster_centers: Top 3 cluster centroids [[enc, dec], ...]
        - n_noise_points: Number of noise points (-1 label)

    Performance:
        ~50ms for 5K samples (vs 300-500ms with GMM)

    DBSCAN Overview:
        1. Mark points with >= min_samples neighbors within eps as core points
        2. Connect core points within eps distance into clusters
        3. Label remaining points as noise (-1)
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
    # Returns labels: [0, 1, 2, ..., -1] where -1 indicates noise
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(weight_pairs)

    # Count clusters (excluding noise label -1)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()

    # Compute silhouette score if we have enough structure
    # Silhouette measures how similar points are to their own cluster
    # compared to other clusters
    silhouette = None
    if n_clusters >= 2 and n_noise < len(labels) * 0.5:  # Need >=2 clusters, not too much noise
        non_noise_mask = labels != -1
        if non_noise_mask.sum() > 0:
            try:
                silhouette = silhouette_score(
                    weight_pairs[non_noise_mask],
                    labels[non_noise_mask]
                )
            except:
                # Can fail if all points in same cluster or distance metric issues
                silhouette = None

    # Extract cluster centers (top 3 by size)
    # Useful for visualizing blob positions
    cluster_centers = []
    if n_clusters > 0:
        # Get unique cluster labels (excluding -1 noise)
        unique_labels = [l for l in set(labels) if l != -1]

        # Sort clusters by size (largest first)
        cluster_sizes = [(l, (labels == l).sum()) for l in unique_labels]
        cluster_sizes.sort(key=lambda x: x[1], reverse=True)

        # Compute centroids for top 3 clusters
        for label, _ in cluster_sizes[:3]:  # Top 3 clusters
            cluster_mask = labels == label
            center = weight_pairs[cluster_mask].mean(axis=0)  # Average of all points in cluster
            cluster_centers.append([float(center[0]), float(center[1])])

    # Fallback if no clusters found (all noise or single cluster)
    # Use overall mean as "center"
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
    eps: float = 0.05,
    min_samples: int = 25
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
