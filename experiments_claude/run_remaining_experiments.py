"""
Manual execution of remaining quick diagnostic sweep experiments.

This script runs the 5 remaining experiments individually to avoid module caching issues.
Each experiment runs in the same process but we force module reloading.
"""

import json
import time
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Force module reload to ensure we have the latest bug fixes
import importlib
for module_name in ['experiments_claude.quick_hyperparam_sweep',
                     'experiments_claude.experiment_runner',
                     'mft_denoising.experiment',
                     'mft_denoising.train']:
    if module_name in sys.modules:
        del sys.modules[module_name]

from experiments_claude.experiment_runner import create_config, run_experiment, load_results

# Define remaining 5 experiments (experiment 1 already completed)
experiments = [
    ('quick_low_lr', 0.01, 5000, 0.03, 0.03, 8),
    ('quick_high_lr', 0.04, 5000, 0.03, 0.03, 8),
    ('quick_small_batch', 0.02, 2500, 0.03, 0.03, 8),
    ('quick_large_batch', 0.02, 10000, 0.03, 0.03, 8),
    ('quick_large_init', 0.02, 5000, 0.05, 0.05, 8)
]

print("=" * 80)
print("MANUAL EXECUTION OF REMAINING EXPERIMENTS")
print("=" * 80)
print(f"\nRunning {len(experiments)} experiments")
print("Each experiment: ~4-5 minutes")
print(f"Total estimated time: ~{len(experiments) * 5} minutes\n")

results = []

for i, (name, lr, batch, enc_init, dec_init, epochs) in enumerate(experiments, 2):
    print(f"\n[{i}/6] Running: {name}")
    print(f"  LR: {lr:.4f}, Batch: {batch}, Init: {enc_init:.2f}/{dec_init:.2f}, Epochs: {epochs}")

    # Create config for this experiment
    config = create_config('reference_3blob')
    config['training']['learning_rate'] = lr
    config['training']['batch_size'] = batch
    config['training']['epochs'] = epochs
    config['training']['enable_diagnostics'] = True
    config['training']['diagnostic_sample_size'] = 3000
    config['training']['save_epoch_checkpoints'] = False
    config['model']['encoder_initialization_scale'] = enc_init
    config['model']['decoder_initialization_scale'] = dec_init

    # Run experiment
    start = time.time()
    try:
        run_experiment(config, f"sweep_{name}", wait=True)
        elapsed = time.time() - start

        # Load results
        result_data = load_results(f"sweep_{name}")
        final = result_data['training_history'][-1]

        # Extract metrics
        metrics = {
            'name': name,
            'success': True,
            'training_time_seconds': elapsed,
            'final_train_loss': final['train']['loss'],
            'final_test_loss': final['test']['scaled_loss'],
            'final_n_clusters': final.get('diagnostics', {}).get('n_clusters_dbscan', 1),
            'final_silhouette': final.get('diagnostics', {}).get('silhouette_score', None),
            'config_params': {
                'learning_rate': lr,
                'batch_size': batch,
                'encoder_init_scale': enc_init,
                'decoder_init_scale': dec_init,
                'epochs': epochs
            }
        }
        results.append(metrics)

        print(f"  ✓ Success: {elapsed:.1f}s, "
              f"Loss: {metrics['final_test_loss']:.2f}, "
              f"Clusters: {metrics['final_n_clusters']}, "
              f"Silhouette: {metrics['final_silhouette']:.3f if metrics['final_silhouette'] else 'N/A'}")

    except Exception as e:
        elapsed = time.time() - start
        print(f"  ✗ Failed: {e}")
        metrics = {
            'name': name,
            'success': False,
            'error': str(e),
            'training_time_seconds': elapsed,
            'config_params': {
                'learning_rate': lr,
                'batch_size': batch,
                'encoder_init_scale': enc_init,
                'decoder_init_scale': dec_init,
                'epochs': epochs
            }
        }
        results.append(metrics)

# Save results
results_path = Path('experiments_claude/quick_diagnostic_results_manual.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 80)
print("MANUAL EXECUTION COMPLETE")
print("=" * 80)
print(f"\nCompleted {len(results)}/{len(experiments)} experiments")
print(f"Results saved to: {results_path}")

# Print summary
print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)
for r in results:
    if r['success']:
        sil = f"{r['final_silhouette']:.3f}" if r['final_silhouette'] else 'N/A'
        print(f"{r['name']:20s} | Loss: {r['final_test_loss']:6.2f} | "
              f"Clusters: {r['final_n_clusters']} | "
              f"Silhouette: {sil:>5s}")
    else:
        print(f"{r['name']:20s} | FAILED: {r['error']}")

print("\n" + "=" * 80)
print("NEXT STEPS")
print("=" * 80)
print("\n1. Combine with first experiment result:")
print("   python -c \"import json; r1=json.load(open('experiments_claude/quick_diagnostic_results.json')); r2=json.load(open('experiments_claude/quick_diagnostic_results_manual.json')); json.dump(r1+r2, open('experiments_claude/quick_diagnostic_results_combined.json','w'), indent=2)\"")
print("\n2. Visualize results:")
print("   python experiments_claude/visualize_sweep.py experiments_claude/quick_diagnostic_results_combined.json")
print("\n3. Check individual experiment directories:")
print("   ls -lt experiments/sweep_quick_* | head -10")
