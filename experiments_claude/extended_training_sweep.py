"""
Extended training sweep with longer epochs to observe blob formation.

Based on quick diagnostic results, using best hyperparameters with extended training:
- Baseline config: LR=0.02, Batch=5000, Init=0.03
- Extended epochs: 20 (vs 8 in quick diagnostic)
- Goal: Observe if blob formation emerges with longer training
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments_claude.experiment_runner import create_config, run_experiment, load_results
import json
import time


def design_extended_sweep():
    """
    Design extended training experiments.

    Tests the top 3 configurations from quick diagnostic with 20 epochs:
    1. Baseline (quick_ref): LR=0.02, Batch=5000, Init=0.03
    2. Large init (quick_large_init): LR=0.02, Batch=5000, Init=0.05
    3. Small batch (quick_small_batch): LR=0.02, Batch=2500, Init=0.03

    Each experiment: ~10-12 minutes → total ~35 minutes
    """
    experiments = [
        {
            'name': 'extended_baseline',
            'learning_rate': 0.02,
            'batch_size': 5000,
            'encoder_init_scale': 0.03,
            'decoder_init_scale': 0.03,
            'epochs': 20
        },
        {
            'name': 'extended_large_init',
            'learning_rate': 0.02,
            'batch_size': 5000,
            'encoder_init_scale': 0.05,
            'decoder_init_scale': 0.05,
            'epochs': 20
        },
        {
            'name': 'extended_small_batch',
            'learning_rate': 0.02,
            'batch_size': 2500,
            'encoder_init_scale': 0.03,
            'decoder_init_scale': 0.03,
            'epochs': 20
        }
    ]

    return experiments


def run_extended_sweep():
    """Run extended training sweep."""
    experiments = design_extended_sweep()

    print("=" * 80)
    print("EXTENDED TRAINING SWEEP")
    print("=" * 80)
    print()
    print(f"Configuration:")
    print(f"  - {len(experiments)} experiments")
    print(f"  - Epochs: 20 (extended from 8)")
    print(f"  - Estimated time: ~35 minutes total")
    print()
    print("Goal: Observe blob formation with longer training")
    print()

    print("Experiments to run:")
    for i, exp in enumerate(experiments, 1):
        print(f"  {i}. {exp['name']}: "
              f"LR={exp['learning_rate']:.3f}, "
              f"Batch={exp['batch_size']}, "
              f"Init={exp['encoder_init_scale']:.2f}, "
              f"Epochs={exp['epochs']}")
    print()

    results = []

    for i, exp_params in enumerate(experiments, 1):
        name = exp_params['name']
        lr = exp_params['learning_rate']
        batch = exp_params['batch_size']
        enc_init = exp_params['encoder_init_scale']
        dec_init = exp_params['decoder_init_scale']
        epochs = exp_params['epochs']

        print(f"\n[{i}/{len(experiments)}] Running: {name}")
        print(f"  LR: {lr:.4f}, Batch: {batch}, Init: {enc_init:.2f}/{dec_init:.2f}, Epochs: {epochs}")

        # Create config
        config = create_config('reference_3blob')
        config['training']['learning_rate'] = lr
        config['training']['batch_size'] = batch
        config['training']['epochs'] = epochs
        config['training']['enable_diagnostics'] = True
        config['training']['diagnostic_sample_size'] = 5000  # Full sampling for extended training
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
                'final_correlation': final.get('diagnostics', {}).get('weight_correlation', None),
                'config_params': {
                    'learning_rate': lr,
                    'batch_size': batch,
                    'encoder_init_scale': enc_init,
                    'decoder_init_scale': dec_init,
                    'epochs': epochs
                }
            }
            results.append(metrics)

            sil = f"{metrics['final_silhouette']:.3f}" if metrics['final_silhouette'] else 'N/A'
            corr = f"{metrics['final_correlation']:.3f}" if metrics['final_correlation'] else 'N/A'
            print(f"  ✓ Success: {elapsed:.1f}s")
            print(f"    Loss: {metrics['final_test_loss']:.2f}")
            print(f"    Clusters: {metrics['final_n_clusters']}")
            print(f"    Silhouette: {sil}")
            print(f"    Correlation: {corr}")

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

        # Save intermediate results
        results_path = Path('experiments_claude/extended_training_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("EXTENDED TRAINING SWEEP COMPLETE!")
    print("=" * 80)

    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    for r in results:
        if r['success']:
            sil = f"{r['final_silhouette']:.3f}" if r['final_silhouette'] else 'N/A'
            corr = f"{r['final_correlation']:.3f}" if r['final_correlation'] else 'N/A'
            print(f"{r['name']:25s} | Loss: {r['final_test_loss']:6.2f} | "
                  f"Clusters: {r['final_n_clusters']} | "
                  f"Sil: {sil:>5s} | Corr: {corr:>5s}")
        else:
            print(f"{r['name']:25s} | FAILED: {r['error']}")

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print()
    print("1. Visualize results:")
    print("   python experiments_claude/visualize_sweep.py experiments_claude/extended_training_results.json")
    print()
    print("2. Check individual experiment plots:")
    print("   ls -lt experiments/sweep_extended_*/encoder_decoder_pairs.png")
    print()
    print("3. Compare with quick diagnostic:")
    print("   Compare cluster formation between epoch 8 and epoch 20")


if __name__ == "__main__":
    run_extended_sweep()
