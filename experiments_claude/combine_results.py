"""
Combine results from all 6 quick diagnostic experiments.
"""

import json
from pathlib import Path

# Load first experiment (quick_ref)
first_result_path = Path('experiments_claude/quick_diagnostic_results.json')
with open(first_result_path) as f:
    first_results = json.load(f)

# Load manual results (has duplicates due to print crashes)
manual_results_path = Path('experiments_claude/quick_diagnostic_results_manual.json')
with open(manual_results_path) as f:
    manual_results = json.load(f)

# Filter to only successful entries (remove error duplicates)
successful_manual = [r for r in manual_results if r['success']]

# Combine all results
all_results = first_results + successful_manual

print(f"Combined {len(all_results)} successful experiments:")
print()
for r in all_results:
    print(f"{r['name']:20s} | Loss: {r['final_test_loss']:6.2f} | "
          f"Clusters: {r['final_n_clusters']} | "
          f"Silhouette: {r['final_silhouette']:.3f if r['final_silhouette'] else 'N/A':>5s} | "
          f"Time: {r['training_time_seconds']:.1f}s")

# Save combined results
combined_path = Path('experiments_claude/quick_diagnostic_results_combined.json')
with open(combined_path, 'w') as f:
    json.dump(all_results, f, indent=2)

print()
print(f"✓ Combined results saved to: {combined_path}")
print()
print("Next step: Visualize results with:")
print(f"  python experiments_claude/visualize_sweep.py {combined_path}")
