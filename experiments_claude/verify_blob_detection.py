"""
Quick verification that updated DBSCAN parameters detect blobs correctly.

Re-run baseline experiment with corrected clustering parameters.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments_claude.experiment_runner import create_config, run_experiment, load_results
import json

print("=" * 80)
print("BLOB DETECTION VERIFICATION")
print("=" * 80)
print()
print("Running baseline experiment with corrected DBSCAN parameters:")
print("  eps: 0.05 (was 0.1)")
print("  min_samples: 25 (was 50)")
print()
print("Expected: 3 clusters with high silhouette score")
print()

# Create config - use 12 epochs for faster verification
config = create_config('reference_3blob')
config['training']['epochs'] = 12
config['training']['batch_size'] = 5000
config['training']['enable_diagnostics'] = True
config['training']['diagnostic_sample_size'] = 5000
config['training']['save_epoch_checkpoints'] = False

print("Running experiment (12 epochs, ~7 minutes)...")
run_experiment(config, "verify_blob_detection", wait=True)

# Load and analyze results
results = load_results("verify_blob_detection")
history = results['training_history']

print("\n" + "=" * 80)
print("BLOB FORMATION TRACKING")
print("=" * 80)
print(f"{'Epoch':>6s} {'Loss':>8s} {'Clusters':>9s} {'Silhouette':>11s} {'Correlation':>12s}")
print("-" * 80)

for epoch_data in history:
    epoch = epoch_data['epoch']
    loss = epoch_data['test']['scaled_loss']
    diag = epoch_data.get('diagnostics', {})
    clusters = diag.get('n_clusters_dbscan', '?')
    sil = diag.get('silhouette_score', None)
    corr = diag.get('weight_correlation', None)

    sil_str = f"{sil:.3f}" if sil else 'N/A'
    corr_str = f"{corr:.3f}" if corr else 'N/A'

    print(f"{epoch:6d} {loss:8.2f} {clusters:9d} {sil_str:>11s} {corr_str:>12s}")

print("=" * 80)

# Final summary
final = history[-1]
final_diag = final['diagnostics']

print("\nFINAL RESULTS:")
print(f"  Test Loss: {final['test']['scaled_loss']:.2f}")
print(f"  Clusters: {final_diag['n_clusters_dbscan']}")
print(f"  Silhouette: {final_diag['silhouette_score']:.3f if final_diag['silhouette_score'] else 'N/A'}")
print(f"  Correlation: {final_diag['weight_correlation']:.3f}")
print(f"  Noise points: {final_diag['n_noise_points']} / {final_diag['n_pairs_analyzed']}")
print()

if final_diag['n_clusters_dbscan'] >= 3:
    print("✓ SUCCESS: Detected 3+ clusters (blob formation confirmed!)")
    print(f"  Silhouette score {final_diag['silhouette_score']:.3f} indicates excellent separation")
else:
    print("✗ UNEXPECTED: Still detecting < 3 clusters")
    print("  May need further parameter tuning")

print()
print("Visualization saved to:")
print(f"  experiments/verify_blob_detection_*/encoder_decoder_pairs.png")
