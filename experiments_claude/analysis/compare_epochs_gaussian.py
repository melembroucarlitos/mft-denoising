"""
Per-epoch Gaussian health analysis for comparing blob quality across training.

Loads per-epoch checkpoints and computes Gaussian-ness diagnostics for each epoch.
Enables comparison of blob formation dynamics during training.
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mft_denoising.nn import TwoLayerNet
from experiments_claude.analysis.histogram_clustering import analyze_weight_pairs_histogram_with_labels
from experiments_claude.analysis.gaussian_diagnostics import gaussian_health_checks, assess_gaussian_quality


def load_checkpoint_weights(
    experiment_dir: Path,
    epoch: int,
    checkpoint_pattern: str = "checkpoint_epoch_{}.pth"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load encoder and decoder weights from a checkpoint file.

    Args:
        experiment_dir: Path to experiment directory
        epoch: Epoch number
        checkpoint_pattern: Pattern for checkpoint filename

    Returns:
        (encoder_weights, decoder_weights) as numpy arrays
    """
    checkpoint_path = experiment_dir / checkpoint_pattern.format(epoch)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint
    state_dict = torch.load(checkpoint_path, map_location='cpu')

    encoder_weights = state_dict['fc1.weight'].numpy()  # (hidden_size, d)
    decoder_weights = state_dict['fc2.weight'].numpy()  # (d, hidden_size)

    return encoder_weights, decoder_weights


def analyze_single_epoch_gaussian(
    encoder_weights: np.ndarray,
    decoder_weights: np.ndarray,
    epoch: int,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Analyze Gaussian health for a single epoch's weight snapshot.

    Args:
        encoder_weights: (hidden_size, d) numpy array
        decoder_weights: (d, hidden_size) numpy array
        epoch: Epoch number (for logging)
        verbose: Print detailed output

    Returns:
        Dictionary with:
        - clustering: histogram clustering results
        - gaussian_health: per-cluster Gaussian diagnostics
        - weight_pairs_shape: shape of weight pairs array
    """
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"Analyzing Epoch {epoch}")
        print(f"{'=' * 70}")

    # Run histogram clustering with label assignment
    clustering_results, labels = analyze_weight_pairs_histogram_with_labels(
        encoder_weights,
        decoder_weights
    )

    if verbose:
        print(f"  Clusters found: {clustering_results['n_clusters']}")
        if clustering_results['silhouette_score'] is not None:
            print(f"  Silhouette score: {clustering_results['silhouette_score']:.4f}")

    # Extract all weight pairs for Gaussian health checks
    hidden_size, d = encoder_weights.shape
    i_indices = np.repeat(np.arange(hidden_size), d)
    j_indices = np.tile(np.arange(d), hidden_size)
    encoder_flat = encoder_weights[i_indices, j_indices]
    decoder_flat = decoder_weights[j_indices, i_indices]
    weight_pairs = np.column_stack([encoder_flat, decoder_flat])

    # Compute Gaussian health if clusters found
    gaussian_health = {}
    if clustering_results['n_clusters'] > 0:
        if verbose:
            print(f"  Computing Gaussian health checks...")

        gaussian_health = gaussian_health_checks(
            points=weight_pairs,
            labels=labels,
            n_clusters=clustering_results['n_clusters'],
            sample_size=50000,
            n_projections=20,
            eps=1e-6
        )

        if verbose:
            print(f"\n  Gaussian Health Results:")
            for cluster_id in sorted(gaussian_health.keys()):
                health = gaussian_health[cluster_id]

                if "note" in health:
                    print(f"    Cluster {cluster_id} ({health['n']} points): {health['note']}")
                    continue

                assessment = assess_gaussian_quality(health)
                print(f"    Cluster {cluster_id} ({health['n']:,} points):")
                print(f"      Radial KS: {health['radial_KS']:.3f}")
                tail_str = ', '.join([f"{t}: {r:.2f}" for t, r in health['tail_ratios'].items()])
                print(f"      Tail ratios: {{{tail_str}}}")
                print(f"      Angle uniformity (R): {health['angle_R']:.4f}")
                print(f"      Projection AD (median): {health['proj_AD_median']:.2f}")
                print(f"      Mardia kurtosis dev: {health['mardia_kurtosis_dev']:.2f}")
                print(f"      → Assessment: {assessment}")

    return {
        "epoch": epoch,
        "weight_pairs_shape": weight_pairs.shape,
        "clustering": clustering_results,
        "gaussian_health": gaussian_health
    }


def analyze_epochs_gaussian(
    experiment_name: str,
    epochs: Optional[List[int]] = None,
    checkpoint_pattern: str = "checkpoint_epoch_{}.pth",
    save_results: bool = True,
    verbose: bool = True
) -> Dict[int, Dict[str, Any]]:
    """
    Analyze Gaussian health across multiple epoch checkpoints.

    Args:
        experiment_name: Name of experiment (with or without timestamp suffix)
        epochs: List of epoch numbers to analyze. If None, auto-detect from checkpoints.
        checkpoint_pattern: Pattern for checkpoint filenames
        save_results: If True, save results JSON to experiment directory
        verbose: Print detailed output

    Returns:
        Dict mapping epoch -> analysis results

    Example:
        >>> results = analyze_epochs_gaussian('diagnostic_demo_20260118_010203')
        >>>
        >>> # Compare tail ratios across epochs
        >>> for epoch, data in results.items():
        ...     for cluster_id, health in data['gaussian_health'].items():
        ...         print(f"Epoch {epoch} Cluster {cluster_id}: tail_14 = {health['tail_ratios'][14]:.2f}")
    """
    # Find experiment directory
    repo_root = Path(__file__).parent.parent.parent
    experiments_dir = repo_root / "experiments"

    matching_dirs = list(experiments_dir.glob(f"{experiment_name}*"))

    if not matching_dirs:
        raise FileNotFoundError(f"No experiment found matching: {experiment_name}")

    experiment_dir = matching_dirs[-1]

    if verbose:
        print(f"Analyzing experiment: {experiment_dir.name}")
        print(f"Directory: {experiment_dir}")

    # Auto-detect epochs if not provided
    if epochs is None:
        checkpoint_files = list(experiment_dir.glob(checkpoint_pattern.replace("{}", "*")))
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoints found in {experiment_dir}")

        # Extract epoch numbers from filenames
        epochs = []
        for cp in checkpoint_files:
            try:
                # Parse epoch number from filename like "checkpoint_epoch_3.pth"
                epoch_str = cp.stem.split('_')[-1]
                epochs.append(int(epoch_str))
            except (ValueError, IndexError):
                continue

        epochs = sorted(epochs)

        if verbose:
            print(f"Auto-detected epochs: {epochs}")

    # Analyze each epoch
    all_results = {}

    for epoch in epochs:
        try:
            # Load checkpoint weights
            encoder_weights, decoder_weights = load_checkpoint_weights(
                experiment_dir,
                epoch,
                checkpoint_pattern
            )

            # Analyze Gaussian health
            epoch_results = analyze_single_epoch_gaussian(
                encoder_weights,
                decoder_weights,
                epoch,
                verbose=verbose
            )

            all_results[epoch] = epoch_results

        except Exception as e:
            print(f"  Error analyzing epoch {epoch}: {e}")
            continue

    # Save results if requested
    if save_results and all_results:
        results_path = experiment_dir / "gaussian_health_per_epoch.json"
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)

        if verbose:
            print(f"\nResults saved to: {results_path}")

    return all_results


def compare_with_reference(
    diagnostic_results: Dict[int, Dict[str, Any]],
    reference_experiment: str = 'reference_3blob_full_20260118_011103',
    reference_epoch: int = 20
) -> None:
    """
    Compare diagnostic_demo epochs with reference_3blob_full epoch 20.

    Prints side-by-side comparison of Gaussian health metrics.

    Args:
        diagnostic_results: Results from analyze_epochs_gaussian()
        reference_experiment: Reference experiment name
        reference_epoch: Reference epoch number
    """
    print(f"\n{'=' * 80}")
    print(f"COMPARISON WITH REFERENCE")
    print(f"{'=' * 80}")

    # Load reference analysis
    from experiments_claude.analysis.measure_decoupling import analyze_experiment

    print(f"\nAnalyzing reference: {reference_experiment} epoch {reference_epoch}...")
    ref_metrics = analyze_experiment(
        reference_experiment,
        save_metrics=False,
        compute_gaussian_health=True
    )

    ref_health = ref_metrics.get('gaussian_health', {})

    # Print comparison table
    print(f"\n{'=' * 80}")
    print("GAUSSIAN HEALTH COMPARISON")
    print(f"{'=' * 80}")
    print(f"{'Experiment':<30} {'Cluster':<10} {'Radial KS':<12} {'Angle R':<12} {'Proj AD':<12} {'Tail 14':<10}")
    print("-" * 80)

    # Reference rows
    for cluster_id in sorted(ref_health.keys()):
        health = ref_health[cluster_id]
        if "note" in health:
            continue

        print(f"{f'Reference (ep {reference_epoch})':<30} {cluster_id:<10} "
              f"{health['radial_KS']:<12.3f} {health['angle_R']:<12.4f} "
              f"{health['proj_AD_median']:<12.2f} {health['tail_ratios'].get(14, float('nan')):<10.2f}")

    print("-" * 80)

    # Diagnostic demo rows
    for epoch in sorted(diagnostic_results.keys()):
        epoch_data = diagnostic_results[epoch]
        gaussian_health = epoch_data.get('gaussian_health', {})

        for cluster_id in sorted(gaussian_health.keys()):
            health = gaussian_health[cluster_id]
            if "note" in health:
                continue

            print(f"{f'Diagnostic (ep {epoch})':<30} {cluster_id:<10} "
                  f"{health['radial_KS']:<12.3f} {health['angle_R']:<12.4f} "
                  f"{health['proj_AD_median']:<12.2f} {health['tail_ratios'].get(14, float('nan')):<10.2f}")

    print("=" * 80)


if __name__ == "__main__":
    print("Per-Epoch Gaussian Health Analyzer")
    print("=" * 70)

    print("\nExample usage:")
    print("""
    from compare_epochs_gaussian import analyze_epochs_gaussian, compare_with_reference

    # Analyze all epochs in experiment
    results = analyze_epochs_gaussian('diagnostic_demo_20260118_012345')

    # Compare with reference
    compare_with_reference(results)

    # Or analyze specific epochs
    results = analyze_epochs_gaussian('diagnostic_demo_20260118_012345', epochs=[1, 2, 3, 4])
    """)
