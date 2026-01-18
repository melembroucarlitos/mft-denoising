"""
Automated metrics for measuring encoder-decoder weight decoupling quality.

Provides quantitative measures of:
- Clustering quality (silhouette score)
- Number of distinct blobs
- Statistical independence of weight pairs
- Distribution structure (Gaussian mixture fitting)
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import json

try:
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import silhouette_score
    from sklearn.mixture import GaussianMixture
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Install with: pip install scikit-learn")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_model_weights(experiment_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load encoder and decoder weights from saved model.

    Args:
        experiment_name: Name of experiment

    Returns:
        (encoder_weights, decoder_weights) as numpy arrays
        encoder_weights: shape (hidden_size, d)
        decoder_weights: shape (d, hidden_size)
    """
    repo_root = Path(__file__).parent.parent.parent
    experiments_dir = repo_root / "experiments"

    matching_dirs = list(experiments_dir.glob(f"{experiment_name}*"))

    if not matching_dirs:
        raise FileNotFoundError(f"No experiment found matching: {experiment_name}")

    model_path = matching_dirs[-1] / "model.pth"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not saved for experiment: {experiment_name}")

    # Load model state
    state_dict = torch.load(model_path, map_location='cpu')

    encoder_weights = state_dict['fc1.weight'].numpy()  # (hidden_size, d)
    decoder_weights = state_dict['fc2.weight'].numpy()  # (d, hidden_size)

    return encoder_weights, decoder_weights


def get_weight_pairs(encoder_weights: np.ndarray, decoder_weights: np.ndarray) -> np.ndarray:
    """
    Extract encoder-decoder weight pairs for each connection.

    Pairs: encoder[i,j] with decoder[j,i] (path: input j -> hidden i -> output j)

    Args:
        encoder_weights: (hidden_size, d)
        decoder_weights: (d, hidden_size)

    Returns:
        weight_pairs: (n_pairs, 2) where n_pairs = hidden_size * d
                     column 0: encoder weights
                     column 1: decoder weights
    """
    hidden_size, d = encoder_weights.shape

    encoder_flat = []
    decoder_flat = []

    for i in range(hidden_size):
        for j in range(d):
            encoder_flat.append(encoder_weights[i, j])
            decoder_flat.append(decoder_weights[j, i])

    weight_pairs = np.column_stack([encoder_flat, decoder_flat])
    return weight_pairs


def compute_clustering_metrics(weight_pairs: np.ndarray) -> Dict[str, Any]:
    """
    Compute clustering quality metrics for weight pairs.

    Args:
        weight_pairs: (n_pairs, 2) array of (encoder, decoder) pairs

    Returns:
        Dictionary with metrics:
        - silhouette_score: Clustering quality (-1 to 1, higher is better)
        - n_clusters: Number of distinct clusters found
        - cluster_labels: Cluster assignment for each point
    """
    if not SKLEARN_AVAILABLE:
        return {"error": "scikit-learn not installed"}

    # Use DBSCAN to find clusters automatically
    # eps and min_samples tuned for weight pair data
    dbscan = DBSCAN(eps=0.1, min_samples=50)
    labels = dbscan.fit_predict(weight_pairs)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # Exclude noise (-1 label)
    n_noise = list(labels).count(-1)

    metrics = {
        "n_clusters": n_clusters,
        "n_noise_points": n_noise,
        "cluster_labels": labels.tolist()
    }

    # Compute silhouette score (only if we have 2+ clusters and not too much noise)
    if n_clusters >= 2 and n_noise < len(labels) * 0.5:
        # Exclude noise points for silhouette calculation
        non_noise_mask = labels != -1
        if non_noise_mask.sum() > 0:
            silhouette = silhouette_score(
                weight_pairs[non_noise_mask],
                labels[non_noise_mask]
            )
            metrics["silhouette_score"] = silhouette
        else:
            metrics["silhouette_score"] = None
    else:
        metrics["silhouette_score"] = None
        metrics["note"] = "Too few clusters or too much noise for silhouette score"

    return metrics


def fit_gaussian_mixture(weight_pairs: np.ndarray, max_components: int = 5) -> Dict[str, Any]:
    """
    Fit Gaussian Mixture Model to weight pairs and select best using BIC.

    Args:
        weight_pairs: (n_pairs, 2) array
        max_components: Maximum number of Gaussian components to try

    Returns:
        Dictionary with:
        - best_n_components: Optimal number of components (by BIC)
        - bic_scores: BIC for each n_components tested
        - aic_scores: AIC for each n_components tested
        - mixture_params: Parameters of best GMM (means, covariances, weights)
    """
    if not SKLEARN_AVAILABLE:
        return {"error": "scikit-learn not installed"}

    bic_scores = []
    aic_scores = []

    for n_components in range(1, max_components + 1):
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        gmm.fit(weight_pairs)
        bic_scores.append(gmm.bic(weight_pairs))
        aic_scores.append(gmm.aic(weight_pairs))

    # Best model: lowest BIC
    best_n_components = np.argmin(bic_scores) + 1

    # Refit best model
    best_gmm = GaussianMixture(n_components=best_n_components, random_state=42)
    best_gmm.fit(weight_pairs)

    return {
        "best_n_components": best_n_components,
        "bic_scores": bic_scores,
        "aic_scores": aic_scores,
        "mixture_means": best_gmm.means_.tolist(),
        "mixture_weights": best_gmm.weights_.tolist()
    }


def compute_statistical_metrics(weight_pairs: np.ndarray) -> Dict[str, float]:
    """
    Compute statistical metrics about weight pair distribution.

    Args:
        weight_pairs: (n_pairs, 2) array

    Returns:
        Dictionary with:
        - encoder_mean, encoder_std: Mean and std of encoder weights
        - decoder_mean, decoder_std: Mean and std of decoder weights
        - correlation: Pearson correlation between encoder and decoder
        - encoder_decoder_product_mean: Mean of encoder * decoder products
    """
    encoder = weight_pairs[:, 0]
    decoder = weight_pairs[:, 1]

    correlation = np.corrcoef(encoder, decoder)[0, 1]
    product_mean = np.mean(encoder * decoder)

    return {
        "encoder_mean": float(encoder.mean()),
        "encoder_std": float(encoder.std()),
        "decoder_mean": float(decoder.mean()),
        "decoder_std": float(decoder.std()),
        "correlation": float(correlation),
        "encoder_decoder_product_mean": float(product_mean)
    }


def analyze_experiment(experiment_name: str, save_metrics: bool = True) -> Dict[str, Any]:
    """
    Complete analysis of an experiment's weight decoupling.

    Args:
        experiment_name: Name of experiment to analyze
        save_metrics: If True, save metrics JSON to experiment directory

    Returns:
        Dictionary with all computed metrics

    Example:
        >>> metrics = analyze_experiment('reference_3blob_20260117_123456')
        >>> print(f"Silhouette score: {metrics['clustering']['silhouette_score']}")
        >>> print(f"Number of blobs: {metrics['clustering']['n_clusters']}")
    """
    print(f"Analyzing experiment: {experiment_name}")

    # Load weights
    encoder_weights, decoder_weights = load_model_weights(experiment_name)
    weight_pairs = get_weight_pairs(encoder_weights, decoder_weights)

    print(f"  Weight pairs shape: {weight_pairs.shape}")
    print(f"  Encoder range: [{weight_pairs[:, 0].min():.4f}, {weight_pairs[:, 0].max():.4f}]")
    print(f"  Decoder range: [{weight_pairs[:, 1].min():.4f}, {weight_pairs[:, 1].max():.4f}]")

    # Compute metrics
    metrics = {
        "experiment_name": experiment_name,
        "weight_pair_shape": weight_pairs.shape,
        "statistical": compute_statistical_metrics(weight_pairs),
        "clustering": compute_clustering_metrics(weight_pairs),
        "gaussian_mixture": fit_gaussian_mixture(weight_pairs)
    }

    # Summary
    print(f"\n  Statistical metrics:")
    print(f"    Encoder-Decoder correlation: {metrics['statistical']['correlation']:.4f}")
    print(f"    Encoder mean: {metrics['statistical']['encoder_mean']:.4f}")
    print(f"    Decoder mean: {metrics['statistical']['decoder_mean']:.4f}")

    if SKLEARN_AVAILABLE:
        print(f"\n  Clustering metrics:")
        print(f"    Number of clusters: {metrics['clustering']['n_clusters']}")
        if metrics['clustering']['silhouette_score'] is not None:
            print(f"    Silhouette score: {metrics['clustering']['silhouette_score']:.4f}")

        print(f"\n  Gaussian Mixture:")
        print(f"    Best number of components: {metrics['gaussian_mixture']['best_n_components']}")

    # Save metrics
    if save_metrics:
        repo_root = Path(__file__).parent.parent.parent
        experiments_dir = repo_root / "experiments"
        matching_dirs = list(experiments_dir.glob(f"{experiment_name}*"))

        if matching_dirs:
            metrics_path = matching_dirs[-1] / "decoupling_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"\n  Saved metrics to: {metrics_path}")

    return metrics


def compare_experiments_metrics(experiment_names: list) -> None:
    """
    Compare decoupling metrics across multiple experiments.

    Args:
        experiment_names: List of experiment names to compare

    Example:
        >>> compare_experiments_metrics(['reference_3blob', 'test_init_005', 'test_init_010'])
    """
    print("=" * 100)
    print("DECOUPLING METRICS COMPARISON")
    print("=" * 100)

    all_metrics = []

    for name in experiment_names:
        try:
            metrics = analyze_experiment(name, save_metrics=False)
            all_metrics.append(metrics)
        except Exception as e:
            print(f"\nError analyzing {name}: {e}")
            continue

    # Summary table
    print("\n" + "=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)
    print(f"{'Experiment':<40} {'Correlation':>12} {'Clusters':>10} {'Silhouette':>12} {'GMM Comp':>10}")
    print("-" * 100)

    for metrics in all_metrics:
        exp_name = metrics['experiment_name'][:38]
        corr = metrics['statistical']['correlation']
        n_clusters = metrics['clustering'].get('n_clusters', 'N/A')
        silhouette = metrics['clustering'].get('silhouette_score', 'N/A')
        if silhouette != 'N/A' and silhouette is not None:
            silhouette = f"{silhouette:.4f}"
        gmm_comp = metrics['gaussian_mixture'].get('best_n_components', 'N/A')

        print(f"{exp_name:<40} {corr:>12.4f} {str(n_clusters):>10} {str(silhouette):>12} {str(gmm_comp):>10}")

    print("=" * 100)


if __name__ == "__main__":
    print("Decoupling Metrics Analyzer")
    print("=" * 80)

    if not SKLEARN_AVAILABLE:
        print("ERROR: scikit-learn not installed")
        print("Install with: pip install scikit-learn")
    else:
        print("Ready to analyze experiments")
        print("\nExample usage:")
        print("""
        from measure_decoupling import analyze_experiment, compare_experiments_metrics

        # Analyze single experiment
        metrics = analyze_experiment('reference_3blob_20260117_123456')

        # Compare multiple experiments
        compare_experiments_metrics(['reference_3blob', 'test_init_005'])
        """)
