"""
Clustering and Gaussian Mixture Model functions for encoder weight analysis.

Used for two-stage training: cluster encoder weights into Gaussian components,
then sample frozen encoder from the mixture distribution.
"""

import numpy as np
import torch
from typing import Tuple, Dict, List, Optional
from sklearn.mixture import GaussianMixture


def cluster_encoder_weights(weights: np.ndarray, n_clusters: int = 3) -> Tuple[List[Dict[str, float]], np.ndarray]:
    """
    Cluster encoder weights into Gaussian components.
    
    Args:
        weights: Flattened encoder weights as numpy array
        n_clusters: Number of clusters/Gaussian components
    
    Returns:
        Tuple of:
        - List of dicts with 'mean' and 'variance' for each cluster
        - Mixture probabilities (proportion of weights in each cluster)
    """
    weights_flat = weights.flatten() if weights.ndim > 1 else weights
    weights_flat = weights_flat.reshape(-1, 1)  # Reshape for sklearn
    
    # Fit Gaussian Mixture Model
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    gmm.fit(weights_flat)
    
    # Extract parameters (convert numpy types to Python native types for JSON serialization)
    cluster_params = []
    for i in range(n_clusters):
        mean = float(gmm.means_[i][0])  # Convert numpy float to Python float
        # Get variance from covariance (which is a scalar for 1D)
        variance = float(gmm.covariances_[i][0, 0])  # Convert numpy float to Python float
        cluster_params.append({
            'mean': mean,
            'variance': variance
        })
    
    # Mixture probabilities (weights/proportions)
    mixture_probs = gmm.weights_
    
    return cluster_params, mixture_probs


def sample_frozen_encoder(
    shape: Tuple[int, ...], 
    gmm_params: List[Dict[str, float]], 
    mixture_probs: np.ndarray,
    random_seed: Optional[int] = None
) -> torch.Tensor:
    """
    Sample a frozen encoder matrix from the Gaussian mixture model.
    
    Each entry is sampled independently:
    1. Sample cluster assignment from mixture probabilities
    2. Sample weight value from that cluster's Gaussian
    
    Args:
        shape: Shape of encoder weight matrix (e.g., (hidden_size, input_size))
        gmm_params: List of dicts with 'mean' and 'variance' for each cluster
        mixture_probs: Mixture probabilities (must sum to 1)
        random_seed: Optional random seed for reproducibility
    
    Returns:
        Frozen encoder weight tensor
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Ensure mixture probabilities sum to 1
    mixture_probs = mixture_probs / mixture_probs.sum()
    
    # Total number of entries
    num_entries = np.prod(shape)
    
    # Sample cluster assignments for each entry
    cluster_indices = np.random.choice(
        len(gmm_params), 
        size=num_entries, 
        p=mixture_probs
    )
    
    # Sample weights from corresponding Gaussians
    sampled_weights = np.zeros(num_entries)
    for i, cluster_idx in enumerate(cluster_indices):
        params = gmm_params[cluster_idx]
        sampled_weights[i] = np.random.normal(
            params['mean'],
            np.sqrt(params['variance'])
        )
    
    # Reshape to original encoder shape
    encoder_matrix = sampled_weights.reshape(shape)
    
    return torch.tensor(encoder_matrix, dtype=torch.float32)
