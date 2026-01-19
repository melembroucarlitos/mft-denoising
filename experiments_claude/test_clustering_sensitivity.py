"""
Test DBSCAN clustering sensitivity on trained model.

Check if blob structure exists but DBSCAN parameters are wrong.
"""

import sys
from pathlib import Path
import torch
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from mft_denoising.nn import TwoLayerNet
from mft_denoising.diagnostics import get_all_weight_pairs

# Load trained model
model_path = Path('experiments/sweep_extended_baseline_20260118_104129/model.pth')
print(f"Loading model from: {model_path}")

# Create model with correct architecture
model = TwoLayerNet(input_size=1024, hidden_size=512)
state_dict = torch.load(model_path, map_location='cpu')
model.load_state_dict(state_dict)

# Extract sampled weight pairs (to avoid memory issues)
from mft_denoising.diagnostics import sample_weight_pairs

print("\nExtracting weight pairs...")
encoder_weights = model.fc1.weight.data.cpu().numpy()
decoder_weights = model.fc2.weight.data.cpu().numpy()
weight_pairs = sample_weight_pairs(encoder_weights, decoder_weights, n_samples=20000)
print(f"Sampled pairs: {len(weight_pairs)} (out of {512 * 1024} total)")
print(f"Encoder range: [{encoder_weights.min():.3f}, {encoder_weights.max():.3f}]")
print(f"Decoder range: [{decoder_weights.min():.3f}, {decoder_weights.max():.3f}]")

# Test different DBSCAN parameters
eps_values = [0.02, 0.05, 0.1, 0.15, 0.2]
min_samples_values = [10, 25, 50, 100]

print("\n" + "=" * 80)
print("DBSCAN PARAMETER SWEEP")
print("=" * 80)
print(f"{'eps':>6s} {'min_samp':>9s} {'clusters':>9s} {'noise%':>8s} {'silhouette':>11s}")
print("-" * 80)

best_result = None
max_clusters = 0

for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(weight_pairs)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        noise_pct = 100 * n_noise / len(labels)

        # Compute silhouette if possible
        sil = None
        if n_clusters >= 2 and n_noise < len(labels) * 0.5:
            non_noise_mask = labels != -1
            if non_noise_mask.sum() > 0:
                try:
                    sil = silhouette_score(weight_pairs[non_noise_mask], labels[non_noise_mask])
                except:
                    sil = None

        sil_str = f"{sil:.3f}" if sil is not None else "N/A"
        print(f"{eps:6.2f} {min_samples:9d} {n_clusters:9d} {noise_pct:7.1f}% {sil_str:>11s}")

        # Track best (most clusters with good silhouette)
        if n_clusters > max_clusters or (n_clusters == max_clusters and sil and (not best_result or sil > best_result['silhouette'])):
            max_clusters = n_clusters
            best_result = {
                'eps': eps,
                'min_samples': min_samples,
                'n_clusters': n_clusters,
                'silhouette': sil,
                'labels': labels,
                'noise_pct': noise_pct
            }

print("=" * 80)

if best_result and best_result['n_clusters'] > 1:
    print(f"\nBest clustering found:")
    print(f"  eps={best_result['eps']:.2f}, min_samples={best_result['min_samples']}")
    print(f"  Clusters: {best_result['n_clusters']}")
    sil_str = f"{best_result['silhouette']:.3f}" if best_result['silhouette'] else 'N/A'
    print(f"  Silhouette: {sil_str}")
    print(f"  Noise: {best_result['noise_pct']:.1f}%")

    # Visualize best clustering
    labels = best_result['labels']
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot each cluster in different color
    unique_labels = set(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        if label == -1:
            # Noise points in black
            mask = labels == label
            ax.scatter(weight_pairs[mask, 0], weight_pairs[mask, 1],
                      c='black', marker='x', s=10, alpha=0.3, label='Noise')
        else:
            mask = labels == label
            ax.scatter(weight_pairs[mask, 0], weight_pairs[mask, 1],
                      c=[color], s=20, alpha=0.6, label=f'Cluster {label}')

    ax.set_xlabel('Encoder Weight')
    ax.set_ylabel('Decoder Weight')
    ax.set_title(f'Best DBSCAN Clustering (eps={best_result["eps"]:.2f}, min_samples={best_result["min_samples"]})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plot_path = Path('experiments_claude/figures/dbscan_sensitivity.png')
    plot_path.parent.mkdir(exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_path}")
else:
    print("\nNo multi-cluster structure found with any parameters tested.")
    print("This suggests the weights genuinely form a single cluster (not decoupled).")

# Show basic statistics
print("\n" + "=" * 80)
print("WEIGHT PAIR STATISTICS")
print("=" * 80)
correlation = np.corrcoef(weight_pairs[:, 0], weight_pairs[:, 1])[0, 1]
print(f"Correlation: {correlation:.3f}")
print(f"Encoder mean: {weight_pairs[:, 0].mean():.6f}, std: {weight_pairs[:, 0].std():.6f}")
print(f"Decoder mean: {weight_pairs[:, 1].mean():.6f}, std: {weight_pairs[:, 1].std():.6f}")
print(f"Encoder/Decoder std ratio: {weight_pairs[:, 0].std() / weight_pairs[:, 1].std():.1f}x")
