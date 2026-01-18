"""
Plot Gaussian-ness trends over training epochs.

Visualizes tail ratios, radial KS, angle uniformity, and projection AD
across epochs to show how blob quality evolves during training.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Optional
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def plot_gaussianity_trends(
    epoch_results: Dict[int, Dict[str, Any]],
    experiment_name: str,
    output_path: Optional[Path] = None,
    log_scale_tail: bool = False
) -> None:
    """
    Plot Gaussian-ness metrics trends across epochs.

    Args:
        epoch_results: Results from analyze_epochs_gaussian()
        experiment_name: Name for plot titles
        output_path: Path to save figure (if None, just display)
        log_scale_tail: Use log scale for tail ratio plots
    """
    epochs = sorted(epoch_results.keys())

    # Extract metrics per cluster per epoch
    # Structure: {cluster_id: {metric_name: [values per epoch]}}
    cluster_metrics = {}

    for epoch in epochs:
        gaussian_health = epoch_results[epoch].get('gaussian_health', {})

        for cluster_id, health in gaussian_health.items():
            if 'note' in health:
                continue

            if cluster_id not in cluster_metrics:
                cluster_metrics[cluster_id] = {
                    'epochs': [],
                    'radial_KS': [],
                    'angle_R': [],
                    'proj_AD_median': [],
                    'tail_6': [],
                    'tail_10': [],
                    'tail_14': [],
                    'mardia_kurtosis_dev': [],
                    'n_points': []
                }

            cluster_metrics[cluster_id]['epochs'].append(epoch)
            cluster_metrics[cluster_id]['radial_KS'].append(health['radial_KS'])
            cluster_metrics[cluster_id]['angle_R'].append(health['angle_R'])
            cluster_metrics[cluster_id]['proj_AD_median'].append(health['proj_AD_median'])
            # Handle both string and integer keys
            tail_ratios = health['tail_ratios']
            cluster_metrics[cluster_id]['tail_6'].append(tail_ratios.get(6, tail_ratios.get('6', float('nan'))))
            cluster_metrics[cluster_id]['tail_10'].append(tail_ratios.get(10, tail_ratios.get('10', float('nan'))))
            cluster_metrics[cluster_id]['tail_14'].append(tail_ratios.get(14, tail_ratios.get('14', float('nan'))))
            cluster_metrics[cluster_id]['mardia_kurtosis_dev'].append(health['mardia_kurtosis_dev'])
            cluster_metrics[cluster_id]['n_points'].append(health['n'])

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Gaussian-ness Trends: {experiment_name}', fontsize=16, fontweight='bold')

    # Plot 1: Radial KS distance
    ax = axes[0, 0]
    for cluster_id, metrics in sorted(cluster_metrics.items()):
        ax.plot(metrics['epochs'], metrics['radial_KS'], marker='o', label=f'Cluster {cluster_id}')
    ax.axhline(y=0.05, color='green', linestyle='--', alpha=0.5, label='Good threshold')
    ax.axhline(y=0.10, color='orange', linestyle='--', alpha=0.5, label='Moderate threshold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Radial KS Distance')
    ax.set_title('Radial Distribution (χ²₂ fit)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Angular uniformity (Rayleigh R)
    ax = axes[0, 1]
    for cluster_id, metrics in sorted(cluster_metrics.items()):
        ax.plot(metrics['epochs'], metrics['angle_R'], marker='s', label=f'Cluster {cluster_id}')

    # Expected uniform threshold for m=50k samples
    m = 50000
    uniform_threshold = 3 / np.sqrt(m)
    ax.axhline(y=uniform_threshold, color='green', linestyle='--', alpha=0.5,
               label=f'Uniform threshold (m={m//1000}k)')
    ax.axhline(y=0.05, color='orange', linestyle='--', alpha=0.5, label='Weak bias')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Rayleigh R Statistic')
    ax.set_title('Angular Uniformity')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Tail ratios (log scale for percentiles)
    ax = axes[1, 0]
    for cluster_id, metrics in sorted(cluster_metrics.items()):
        epochs_arr = np.array(metrics['epochs'])
        ax.plot(epochs_arr, metrics['tail_6'], marker='o', linestyle='-', alpha=0.7,
                label=f'Cluster {cluster_id} (90%)')
        ax.plot(epochs_arr, metrics['tail_10'], marker='s', linestyle='--', alpha=0.7,
                label=f'Cluster {cluster_id} (99%)')
        ax.plot(epochs_arr, metrics['tail_14'], marker='^', linestyle=':', alpha=0.7,
                label=f'Cluster {cluster_id} (99.9%)')

    ax.axhline(y=1.0, color='green', linestyle='-', linewidth=2, alpha=0.5, label='Expected (Gaussian)')

    if log_scale_tail:
        ax.set_yscale('log')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Tail Ratio (empirical / expected)')
    ax.set_title('Tail Behavior at χ²₂ Quantiles')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3, which='both')

    # Plot 4: Projection AD median
    ax = axes[1, 1]
    for cluster_id, metrics in sorted(cluster_metrics.items()):
        ax.plot(metrics['epochs'], metrics['proj_AD_median'], marker='d', label=f'Cluster {cluster_id}')
    ax.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='Excellent')
    ax.axhline(y=1.0, color='orange', linestyle='--', alpha=0.5, label='Good')
    ax.axhline(y=2.0, color='red', linestyle='--', alpha=0.5, label='Moderate')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Median Anderson-Darling')
    ax.set_title('1D Projection Normality (20 random directions)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Set y-scale for Projection AD based on range
    max_ad = max([max(m['proj_AD_median']) for m in cluster_metrics.values()])
    if max_ad > 100:
        ax.set_yscale('log')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {output_path}")
    else:
        plt.show()


def plot_log_percentiles(
    epoch_results: Dict[int, Dict[str, Any]],
    experiment_name: str,
    cluster_id: int = 0,
    output_path: Optional[Path] = None
) -> None:
    """
    Plot log-scale tail percentiles for a specific cluster across epochs.

    Focuses on tail behavior requested by user: "plot the log percentiles
    for the different cutoffs".

    Args:
        epoch_results: Results from analyze_epochs_gaussian()
        experiment_name: Name for plot titles
        cluster_id: Which cluster to analyze
        output_path: Path to save figure
    """
    epochs = []
    tail_6_ratios = []
    tail_10_ratios = []
    tail_14_ratios = []

    for epoch in sorted(epoch_results.keys()):
        gaussian_health = epoch_results[epoch].get('gaussian_health', {})

        if cluster_id in gaussian_health:
            health = gaussian_health[cluster_id]
            if 'note' not in health:
                epochs.append(epoch)
                # Handle both string and integer keys
                tail_ratios = health['tail_ratios']
                tail_6_ratios.append(tail_ratios.get(6, tail_ratios.get('6', float('nan'))))
                tail_10_ratios.append(tail_ratios.get(10, tail_ratios.get('10', float('nan'))))
                tail_14_ratios.append(tail_ratios.get(14, tail_ratios.get('14', float('nan'))))

    if not epochs:
        print(f"No data found for cluster {cluster_id}")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.semilogy(epochs, tail_6_ratios, marker='o', linewidth=2,
                label='90th percentile (χ²₂ = 6)', color='blue')
    ax.semilogy(epochs, tail_10_ratios, marker='s', linewidth=2,
                label='99th percentile (χ²₂ = 10)', color='orange')
    ax.semilogy(epochs, tail_14_ratios, marker='^', linewidth=2,
                label='99.9th percentile (χ²₂ = 14)', color='red')

    ax.axhline(y=1.0, color='green', linestyle='--', linewidth=2,
               label='Expected (Gaussian)', alpha=0.7)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Tail Ratio (empirical / expected)', fontsize=12)
    ax.set_title(f'Log-Scale Tail Percentiles: {experiment_name}\nCluster {cluster_id}',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {output_path}")
    else:
        plt.show()


def plot_cluster_size_evolution(
    epoch_results: Dict[int, Dict[str, Any]],
    experiment_name: str,
    output_path: Optional[Path] = None
) -> None:
    """
    Plot how cluster sizes evolve over epochs.

    Args:
        epoch_results: Results from analyze_epochs_gaussian()
        experiment_name: Name for plot titles
        output_path: Path to save figure
    """
    epochs = sorted(epoch_results.keys())

    # Extract cluster sizes
    cluster_sizes = {}

    for epoch in epochs:
        gaussian_health = epoch_results[epoch].get('gaussian_health', {})

        for cluster_id, health in gaussian_health.items():
            if 'note' in health:
                continue

            if cluster_id not in cluster_sizes:
                cluster_sizes[cluster_id] = {'epochs': [], 'sizes': [], 'percentages': []}

            total_size = sum([h['n'] for h in gaussian_health.values() if 'note' not in h])

            cluster_sizes[cluster_id]['epochs'].append(epoch)
            cluster_sizes[cluster_id]['sizes'].append(health['n'])
            cluster_sizes[cluster_id]['percentages'].append(100 * health['n'] / total_size)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Cluster Evolution: {experiment_name}', fontsize=14, fontweight='bold')

    # Plot 1: Absolute sizes
    for cluster_id, data in sorted(cluster_sizes.items()):
        ax1.plot(data['epochs'], data['sizes'], marker='o', label=f'Cluster {cluster_id}')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Number of Points')
    ax1.set_title('Cluster Sizes (absolute)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Percentages
    for cluster_id, data in sorted(cluster_sizes.items()):
        ax2.plot(data['epochs'], data['percentages'], marker='s', label=f'Cluster {cluster_id}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Percentage of Total (%)')
    ax2.set_title('Cluster Sizes (percentage)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    print("Gaussian-ness Trend Visualization")
    print("=" * 70)

    print("\nExample usage:")
    print("""
    from compare_epochs_gaussian import analyze_epochs_gaussian
    from plot_gaussianity_trends import plot_gaussianity_trends, plot_log_percentiles

    # Analyze epochs
    results = analyze_epochs_gaussian('reference_3blob_20260118_123456')

    # Plot all Gaussian-ness metrics
    plot_gaussianity_trends(results, 'Reference 3-Blob',
                           output_path='gaussianity_trends.png')

    # Focus on tail behavior for outer cluster (log scale)
    plot_log_percentiles(results, 'Reference 3-Blob', cluster_id=0,
                        output_path='tail_percentiles_cluster0.png')

    # Plot cluster size evolution
    plot_cluster_size_evolution(results, 'Reference 3-Blob',
                                output_path='cluster_evolution.png')
    """)
