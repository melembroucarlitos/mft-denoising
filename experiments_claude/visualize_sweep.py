"""
Quick visualization of hyperparameter sweep results.

Generates sanity-check plots:
- Training time vs final loss
- Number of clusters across experiments
- Silhouette score (blob quality)
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Any


def load_sweep_results(results_path: str = 'experiments_claude/sweep_results_final.json') -> List[Dict[str, Any]]:
    """Load sweep results from JSON file."""
    with open(results_path, 'r') as f:
        return json.load(f)


def plot_sweep_summary(results: List[Dict[str, Any]], output_dir: str = 'experiments_claude/figures'):
    """
    Generate quick sanity-check plots from sweep results.

    Creates 3-panel figure:
    1. Training time vs loss (colored by n_clusters)
    2. Cluster count per experiment (bar chart)
    3. Blob quality (silhouette score) per experiment
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Filter successful experiments
    successful = [r for r in results if r.get('success', False)]

    if not successful:
        print("No successful experiments found!")
        return

    print(f"Plotting {len(successful)} successful experiments")

    # Extract metrics
    names = [r['name'] for r in successful]
    times = [r['training_time_seconds'] for r in successful]
    losses = [r['final_test_loss'] for r in successful]
    clusters = [r['final_n_clusters'] for r in successful]
    silhouettes = [r.get('final_silhouette') for r in successful]

    # Create figure
    fig = plt.figure(figsize=(18, 5))

    # Plot 1: Time vs Loss (colored by clusters)
    ax1 = plt.subplot(1, 3, 1)
    scatter = ax1.scatter(times, losses, c=clusters, cmap='viridis', s=100, alpha=0.7, edgecolors='black')
    ax1.set_xlabel('Training Time (seconds)', fontsize=11)
    ax1.set_ylabel('Final Test Loss', fontsize=11)
    ax1.set_title('Training Efficiency', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Number of Clusters', fontsize=10)

    # Add reference line if available
    ref_idx = [i for i, name in enumerate(names) if 'reference' in name.lower()]
    if ref_idx:
        ax1.scatter([times[ref_idx[0]]], [losses[ref_idx[0]]],
                   marker='*', s=400, c='red', edgecolors='black', linewidths=2,
                   label='Reference', zorder=10)
        ax1.legend(fontsize=10)

    # Plot 2: Cluster counts (bar chart)
    ax2 = plt.subplot(1, 3, 2)
    colors = plt.cm.viridis(np.array(clusters) / max(clusters))
    bars = ax2.bar(range(len(names)), clusters, color=colors, edgecolor='black')
    ax2.set_xlabel('Experiment', fontsize=11)
    ax2.set_ylabel('Number of Clusters', fontsize=11)
    ax2.set_title('Cluster Formation', fontsize=12, fontweight='bold')
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels([f"{i+1}" for i in range(len(names))], fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=3, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Target: 3 clusters')
    ax2.legend(fontsize=9)

    # Plot 3: Blob quality (silhouette score)
    ax3 = plt.subplot(1, 3, 3)
    valid_sils = [(i, s) for i, s in enumerate(silhouettes) if s is not None]
    if valid_sils:
        indices, sil_values = zip(*valid_sils)
        sil_colors = [colors[i] for i in indices]
        bars = ax3.bar(indices, sil_values, color=sil_colors, edgecolor='black')
        ax3.set_xlabel('Experiment', fontsize=11)
        ax3.set_ylabel('Silhouette Score', fontsize=11)
        ax3.set_title('Blob Quality (multi-cluster only)', fontsize=12, fontweight='bold')
        ax3.set_xticks(range(len(names)))
        ax3.set_xticklabels([f"{i+1}" for i in range(len(names))], fontsize=8)
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.set_ylim([0, 1])
        ax3.axhline(y=0.5, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='Good threshold')
        ax3.axhline(y=0.7, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='Excellent threshold')
        ax3.legend(fontsize=9)
    else:
        ax3.text(0.5, 0.5, 'No multi-cluster experiments',
                ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Blob Quality', fontsize=12, fontweight='bold')

    plt.tight_layout()

    # Save figure
    output_path = output_dir / 'sweep_summary.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Also create a legend mapping experiment numbers to names
    legend_path = output_dir / 'sweep_legend.txt'
    with open(legend_path, 'w') as f:
        f.write("Experiment Index to Name Mapping\n")
        f.write("=" * 60 + "\n\n")
        for i, name in enumerate(names):
            f.write(f"{i+1:2d}. {name}\n")
    print(f"Saved experiment legend: {legend_path}")

    return output_path


def print_sweep_summary(results: List[Dict[str, Any]]):
    """Print text summary of sweep results."""
    successful = [r for r in results if r.get('success', False)]
    failed = [r for r in results if not r.get('success', False)]

    print("\n" + "="*80)
    print("SWEEP SUMMARY")
    print("="*80)
    print(f"\nTotal experiments: {len(results)}")
    print(f"  Successful: {len(successful)}")
    print(f"  Failed: {len(failed)}")

    if not successful:
        print("\nNo successful experiments to analyze!")
        return

    # Find reference experiment
    ref = [r for r in successful if 'reference' in r['name'].lower()]
    if ref:
        ref = ref[0]
        print(f"\n📌 Reference baseline:")
        print(f"   Time: {ref['training_time_seconds']:.1f}s, Loss: {ref['final_test_loss']:.3f}, Clusters: {ref['final_n_clusters']}")

    # Find 3-cluster experiments
    three_cluster = [r for r in successful if r['final_n_clusters'] == 3]
    print(f"\n✓ Experiments with 3-cluster structure: {len(three_cluster)}/{len(successful)}")

    if three_cluster:
        # Best blob quality
        with_sil = [r for r in three_cluster if r.get('final_silhouette') is not None]
        if with_sil:
            best_quality = max(with_sil, key=lambda x: x['final_silhouette'])
            print(f"\n🏆 Best blob quality:")
            print(f"   {best_quality['name']}")
            print(f"   Silhouette: {best_quality['final_silhouette']:.3f}, Time: {best_quality['training_time_seconds']:.1f}s")

        # Fastest 3-cluster
        fastest = min(three_cluster, key=lambda x: x['training_time_seconds'])
        print(f"\n⚡ Fastest 3-cluster training:")
        print(f"   {fastest['name']}")
        print(f"   Time: {fastest['training_time_seconds']:.1f}s, Loss: {fastest['final_test_loss']:.3f}")

        # Lowest loss
        lowest_loss = min(three_cluster, key=lambda x: x['final_test_loss'])
        print(f"\n🎯 Lowest loss (3-cluster):")
        print(f"   {lowest_loss['name']}")
        print(f"   Loss: {lowest_loss['final_test_loss']:.3f}, Time: {lowest_loss['training_time_seconds']:.1f}s")

    # Overall fastest
    fastest_overall = min(successful, key=lambda x: x['training_time_seconds'])
    print(f"\n⏱️  Fastest overall:")
    print(f"   {fastest_overall['name']}")
    print(f"   Time: {fastest_overall['training_time_seconds']:.1f}s, Clusters: {fastest_overall['final_n_clusters']}")

    print("\n" + "="*80)


if __name__ == "__main__":
    import sys

    # Default to final results, but allow intermediate monitoring
    results_file = sys.argv[1] if len(sys.argv) > 1 else 'experiments_claude/sweep_results_final.json'

    print(f"Loading sweep results from: {results_file}")

    try:
        results = load_sweep_results(results_file)
    except FileNotFoundError:
        print(f"Error: {results_file} not found!")
        print("Run the sweep first or check intermediate results at:")
        print("  experiments_claude/sweep_results_intermediate.json")
        sys.exit(1)

    # Print text summary
    print_sweep_summary(results)

    # Generate plots
    print("\nGenerating plots...")
    plot_path = plot_sweep_summary(results)

    print(f"\n✓ Visualization complete!")
    print(f"  Main plot: {plot_path}")
    print(f"  Legend: experiments_claude/figures/sweep_legend.txt")
