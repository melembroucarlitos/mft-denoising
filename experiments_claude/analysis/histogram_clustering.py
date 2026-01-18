"""
Memory-efficient histogram-based clustering for full weight pair analysis.

This module provides post-training blob analysis that avoids OOM by clustering
histogram bins instead of individual weight pairs. Uses adaptive radial density
thresholding validated on both 2-cluster and 3+ cluster cases.
"""

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from scipy.ndimage import maximum_filter
from typing import Dict, Any, Optional, List, Tuple
import time


def analyze_weight_pairs_histogram(
    encoder_weights: np.ndarray,
    decoder_weights: np.ndarray,
    n_bins: int = 100,
    n_radial_bins: int = 50,
    dbscan_eps: float = 0.1,
    dbscan_min_samples: int = 5,
    radial_separation_factor: float = 0.3,
    peak_threshold_factor: float = 0.2,
    quality_gate_silhouette: float = 0.3,
    quality_gate_noise_frac: float = 0.5
) -> Dict[str, Any]:
    """
    Comprehensive blob analysis using histogram-based clustering.

    This method bins weight pairs into a 2D histogram, then clusters the bin centers
    instead of individual points. This avoids O(n²) memory issues with DBSCAN on
    524K points and provides accurate cluster counts even for rare blobs (~1% of points).

    Algorithm:
    1. Extract all encoder-decoder weight pairs
    2. Create 2D histogram (100×100 bins by default)
    3. Whiten bin centers using StandardScaler
    4. Compute radial density profile (distance vs density)
    5. Find local maxima in density with std-based minimum separation
    6. Threshold = 0.2 × second-largest peak density (adaptive)
    7. Run DBSCAN on selected high-density bins
    8. Compute cluster statistics and quality metrics

    Args:
        encoder_weights: (hidden_size, d) numpy array
        decoder_weights: (d, hidden_size) numpy array
        n_bins: Number of bins per dimension for 2D histogram
        n_radial_bins: Number of bins for radial density profile
        dbscan_eps: DBSCAN epsilon (neighborhood radius)
        dbscan_min_samples: DBSCAN minimum samples per cluster
        radial_separation_factor: Minimum separation = factor × radial_std
        peak_threshold_factor: Threshold = factor × second-largest peak
        quality_gate_silhouette: Minimum silhouette score to pass quality gate
        quality_gate_noise_frac: Maximum noise fraction to pass quality gate

    Returns:
        Dictionary with:
        - n_clusters: Number of clusters found
        - silhouette_score: Clustering quality
        - cluster_centers: List of cluster centroids in original space
        - cluster_sizes: Number of points per cluster
        - n_noise_bins: Number of noise bins
        - total_points: Total weight pairs analyzed
        - quality_gate_passed: Whether metrics pass quality thresholds
        - computation_time_ms: Time taken
        - debug_info: Additional debugging information

    Validation:
        ✓ reference_3blob_full (epoch 20): 3 clusters, silhouette=0.827
        ✓ diagnostic_demo (epoch 4): 2 clusters (correctly handles 2-cluster case)
        ✓ Detects rare blobs with only ~1% of points
    """
    start_time = time.time()

    # 1. Extract all weight pairs
    hidden_size, d = encoder_weights.shape
    i_indices = np.repeat(np.arange(hidden_size), d)
    j_indices = np.tile(np.arange(d), hidden_size)
    encoder_flat = encoder_weights[i_indices, j_indices]
    decoder_flat = decoder_weights[j_indices, i_indices]
    total_pairs = len(encoder_flat)

    # 2. Create 2D histogram
    hist, xedges, yedges = np.histogram2d(encoder_flat, decoder_flat, bins=n_bins)

    # 3. Get bin centers and densities
    bin_centers = []
    densities = []
    for i in range(n_bins):
        for j in range(n_bins):
            if hist[i, j] > 0:
                center_x = (xedges[i] + xedges[i+1]) / 2
                center_y = (yedges[j] + yedges[j+1]) / 2
                bin_centers.append([center_x, center_y])
                densities.append(hist[i, j])

    bin_centers = np.array(bin_centers)
    densities = np.array(densities)

    # 4. Whiten (standardize to unit variance for isotropic distances)
    scaler = StandardScaler()
    bin_centers_whitened = scaler.fit_transform(bin_centers)

    # 5. Compute radial distances from origin
    distances = np.sqrt(bin_centers_whitened[:, 0]**2 + bin_centers_whitened[:, 1]**2)

    # 6. Radial density profile
    radial_hist, radial_edges = np.histogram(distances, bins=n_radial_bins, weights=densities)
    radial_counts, _ = np.histogram(distances, bins=n_radial_bins)
    radial_density = np.zeros_like(radial_hist, dtype=float)
    nonzero_mask = radial_counts > 0
    radial_density[nonzero_mask] = radial_hist[nonzero_mask] / radial_counts[nonzero_mask]
    radial_centers = (radial_edges[:-1] + radial_edges[1:]) / 2

    # 7. Find local maxima with std-based separation
    local_max_mask = (radial_density == maximum_filter(radial_density, size=3)) & (radial_density > 0)
    local_maxima = radial_density[local_max_mask]
    local_maxima_radii = radial_centers[local_max_mask]

    radial_std = np.std(distances)
    min_separation = radial_separation_factor * radial_std

    # Filter peaks by separation
    sorted_indices = np.argsort(local_maxima)[::-1]
    peaks_filtered = []
    for idx in sorted_indices:
        r = local_maxima_radii[idx]
        p = local_maxima[idx]
        if all(abs(r - existing_r) > min_separation for existing_r, _ in peaks_filtered):
            peaks_filtered.append((r, p))

    # 8. Adaptive thresholding
    if len(peaks_filtered) >= 2:
        p_star = peaks_filtered[1][1]  # Second-largest peak
        threshold = peak_threshold_factor * p_star
        threshold_method = "second_peak"
    else:
        threshold = np.percentile(densities, 50)
        threshold_method = "median_fallback"

    # 9. Select bins above threshold
    selected_mask = densities >= threshold
    selected_bins = bin_centers_whitened[selected_mask]
    selected_densities = densities[selected_mask]

    # 10. DBSCAN clustering on selected bins
    dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
    labels = dbscan.fit_predict(selected_bins)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))

    # 11. Compute silhouette score
    silhouette = None
    if n_clusters >= 2 and n_noise < quality_gate_noise_frac * len(selected_bins):
        non_noise_mask = labels != -1
        if np.sum(non_noise_mask) > 0:
            try:
                silhouette = float(silhouette_score(
                    selected_bins[non_noise_mask],
                    labels[non_noise_mask]
                ))
            except Exception:
                silhouette = None

    # 12. Quality gate
    quality_gate_passed = (
        n_clusters >= 2 and
        silhouette is not None and
        silhouette > quality_gate_silhouette and
        n_noise < quality_gate_noise_frac * len(selected_bins)
    )

    # 13. Compute cluster statistics (map back to original space)
    cluster_centers_original = []
    cluster_sizes = []

    if n_clusters > 0:
        # Map labels back to all bins
        full_labels = np.full(len(bin_centers), -1, dtype=int)
        full_labels[selected_mask] = labels

        for cluster_id in range(n_clusters):
            cluster_mask = full_labels == cluster_id
            cluster_points = bin_centers[cluster_mask]
            cluster_weights = densities[cluster_mask]

            # Weighted centroid in original space
            center_encoder = float(np.average(cluster_points[:, 0], weights=cluster_weights))
            center_decoder = float(np.average(cluster_points[:, 1], weights=cluster_weights))

            # Total points in this cluster
            n_points = int(cluster_weights.sum())

            cluster_centers_original.append([center_encoder, center_decoder])
            cluster_sizes.append(n_points)

    computation_time_ms = (time.time() - start_time) * 1000

    return {
        "n_clusters": int(n_clusters),
        "silhouette_score": silhouette,
        "cluster_centers": cluster_centers_original,
        "cluster_sizes": cluster_sizes,
        "n_noise_bins": int(n_noise),
        "total_points": int(total_pairs),
        "quality_gate_passed": bool(quality_gate_passed),
        "computation_time_ms": float(computation_time_ms),
        "debug_info": {
            "n_bins_total": int(len(bin_centers)),
            "n_bins_selected": int(len(selected_bins)),
            "n_peaks_found": int(len(peaks_filtered)),
            "threshold_value": float(threshold),
            "threshold_method": threshold_method,
            "radial_std": float(radial_std),
            "min_separation": float(min_separation)
        }
    }


def analyze_model_histogram(
    model: torch.nn.Module,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to analyze a PyTorch model.

    Args:
        model: TwoLayerNet model with fc1 (encoder) and fc2 (decoder)
        **kwargs: Additional arguments passed to analyze_weight_pairs_histogram

    Returns:
        Histogram clustering results
    """
    encoder_weights = model.fc1.weight.detach().cpu().numpy()
    decoder_weights = model.fc2.weight.detach().cpu().numpy()

    return analyze_weight_pairs_histogram(
        encoder_weights,
        decoder_weights,
        **kwargs
    )


def assign_labels_from_centers(
    points: np.ndarray,
    cluster_centers: List[List[float]],
    noise_threshold: float = 3.0
) -> np.ndarray:
    """
    Assign cluster labels to all points based on nearest cluster center.
    
    Points that are too far from any center (beyond noise_threshold × typical_scale)
    are labeled as noise (-1).
    
    Args:
        points: (N, 2) array of weight pairs
        cluster_centers: List of [encoder, decoder] cluster centroids
        noise_threshold: Distance threshold (in units of typical scale)
    
    Returns:
        labels: (N,) array with cluster IDs (0, 1, ..., K-1) or -1 for noise
    
    Algorithm:
        1. Compute Euclidean distance from each point to all cluster centers
        2. Assign to nearest cluster
        3. Mark as noise if distance > noise_threshold × typical_scale
           where typical_scale = mean of minimum inter-center distances
    """
    N = len(points)
    K = len(cluster_centers)
    labels = np.full(N, -1, dtype=int)
    
    if K == 0:
        return labels
    
    centers_array = np.array(cluster_centers)  # (K, 2)
    
    # Compute distances to all centers: (N, K)
    # Broadcasting: points[:, np.newaxis, :] is (N, 1, 2)
    #               centers_array[np.newaxis, :, :] is (1, K, 2)
    dists = np.linalg.norm(
        points[:, np.newaxis, :] - centers_array[np.newaxis, :, :],
        axis=2
    )
    
    # Assign to nearest cluster
    nearest_cluster = np.argmin(dists, axis=1)
    nearest_dist = np.min(dists, axis=1)
    
    # Estimate typical scale from inter-center distances
    if K > 1:
        center_dists = np.linalg.norm(
            centers_array[:, np.newaxis, :] - centers_array[np.newaxis, :, :],
            axis=2
        )
        # Set diagonal to inf to exclude self-distances
        np.fill_diagonal(center_dists, np.inf)
        # Typical scale = mean of minimum distances to other centers
        typical_scale = np.min(center_dists, axis=1).mean()
    else:
        # Single cluster: use std of distances as typical scale
        typical_scale = np.std(nearest_dist)
    
    # Mark points as belonging to cluster if within threshold
    noise_mask = nearest_dist > noise_threshold * typical_scale
    labels[~noise_mask] = nearest_cluster[~noise_mask]
    
    return labels


def analyze_weight_pairs_histogram_with_labels(
    encoder_weights: np.ndarray,
    decoder_weights: np.ndarray,
    **kwargs
) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    Variant that returns both clustering results and per-point labels.
    
    This is a convenience wrapper around analyze_weight_pairs_histogram
    that also assigns labels to all individual weight pairs.
    
    Args:
        encoder_weights: (hidden_size, d) numpy array
        decoder_weights: (d, hidden_size) numpy array
        **kwargs: Passed to analyze_weight_pairs_histogram
    
    Returns:
        (results, labels) tuple:
        - results: Standard histogram clustering results dictionary
        - labels: (N,) array of cluster assignments for all N weight pairs
    """
    # Run histogram clustering
    results = analyze_weight_pairs_histogram(encoder_weights, decoder_weights, **kwargs)
    
    # Extract all weight pairs
    hidden_size, d = encoder_weights.shape
    i_indices = np.repeat(np.arange(hidden_size), d)
    j_indices = np.tile(np.arange(d), hidden_size)
    encoder_flat = encoder_weights[i_indices, j_indices]
    decoder_flat = decoder_weights[j_indices, i_indices]
    weight_pairs = np.column_stack([encoder_flat, decoder_flat])
    
    # Assign labels based on cluster centers
    if results["n_clusters"] > 0:
        labels = assign_labels_from_centers(
            weight_pairs,
            results["cluster_centers"],
            noise_threshold=3.0
        )
    else:
        # No clusters found, mark all as noise
        labels = np.full(len(weight_pairs), -1, dtype=int)
    
    return results, labels
