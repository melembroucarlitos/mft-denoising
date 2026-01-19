"""
Quick diagnostic sweep - 5-6 experiments in ~30 minutes.

Tests key parameters with reduced batch size and epochs to quickly identify:
- Whether blob formation occurs
- Which learning rates are stable
- Promising hyperparameter regions for deeper exploration

Configuration:
- Batch size: 5000 (half of default)
- Epochs: 8 (instead of 12)
- Diagnostic samples: 3000 (instead of 5000)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments_claude.quick_hyperparam_sweep import run_sweep, analyze_sweep_results
from experiments_claude.experiment_runner import create_config


def design_quick_diagnostic_sweep():
    """
    Design 5-6 key experiments for quick diagnostics.

    Tests:
    1. Reference (lr=0.02) - baseline
    2. Low LR (lr=0.01) - more stable?
    3. High LR (lr=0.04) - faster convergence?
    4. Small batch (batch=2500) - better blob formation?
    5. Large batch (batch=10000) - faster training?
    6. Large init (init=0.05) - different starting point?

    Each experiment: ~5 minutes → total ~30 minutes
    """
    ref_batch = 5000
    ref_lr = 0.02
    ref_init = 0.03

    experiments = [
        {
            'name': 'quick_ref',
            'learning_rate': ref_lr,
            'batch_size': ref_batch,
            'encoder_init_scale': ref_init,
            'decoder_init_scale': ref_init,
            'epochs': 8
        },
        {
            'name': 'quick_low_lr',
            'learning_rate': 0.01,
            'batch_size': ref_batch,
            'encoder_init_scale': ref_init,
            'decoder_init_scale': ref_init,
            'epochs': 8
        },
        {
            'name': 'quick_high_lr',
            'learning_rate': 0.04,
            'batch_size': ref_batch,
            'encoder_init_scale': ref_init,
            'decoder_init_scale': ref_init,
            'epochs': 8
        },
        {
            'name': 'quick_small_batch',
            'learning_rate': ref_lr,
            'batch_size': 2500,
            'encoder_init_scale': ref_init,
            'decoder_init_scale': ref_init,
            'epochs': 8
        },
        {
            'name': 'quick_large_batch',
            'learning_rate': ref_lr,
            'batch_size': 10000,
            'encoder_init_scale': ref_init,
            'decoder_init_scale': ref_init,
            'epochs': 8
        },
        {
            'name': 'quick_large_init',
            'learning_rate': ref_lr,
            'batch_size': ref_batch,
            'encoder_init_scale': 0.05,
            'decoder_init_scale': 0.05,
            'epochs': 8
        }
    ]

    print(f"Designed {len(experiments)} quick diagnostic experiments")
    print(f"Estimated time: ~5 minutes per experiment = ~30 minutes total")
    print()

    return experiments


def create_quick_sweep_config(
    learning_rate: float,
    batch_size: int,
    encoder_init_scale: float,
    decoder_init_scale: float,
    epochs: int
):
    """Create config for quick diagnostic sweep."""
    config = create_config('reference_3blob')

    # Set parameters
    config['training']['learning_rate'] = learning_rate
    config['training']['batch_size'] = batch_size
    config['training']['epochs'] = epochs
    config['training']['enable_diagnostics'] = True
    config['training']['diagnostic_sample_size'] = 3000  # Reduced from 5000
    config['training']['save_epoch_checkpoints'] = False

    config['model']['encoder_initialization_scale'] = encoder_init_scale
    config['model']['decoder_initialization_scale'] = decoder_init_scale

    return config


if __name__ == "__main__":
    import json

    print("=" * 80)
    print("QUICK DIAGNOSTIC SWEEP")
    print("=" * 80)
    print()
    print("Configuration:")
    print("  - 6 experiments")
    print("  - Batch size: 5000 (half of default)")
    print("  - Epochs: 8 (reduced from 12)")
    print("  - Estimated time: ~30 minutes")
    print()
    print("Purpose:")
    print("  - Verify blob formation occurs")
    print("  - Identify stable learning rates")
    print("  - Find promising hyperparameter regions")
    print()

    # Design experiments
    experiments = design_quick_diagnostic_sweep()

    print("Experiments to run:")
    for i, exp in enumerate(experiments, 1):
        print(f"  {i}. {exp['name']}: "
              f"LR={exp['learning_rate']:.3f}, "
              f"Batch={exp['batch_size']}, "
              f"Init={exp['encoder_init_scale']:.2f}")
    print()

    response = input("Start quick diagnostic sweep? [y/N]: ")
    if response.lower() != 'y':
        print("Cancelled.")
        exit(0)

    print("\n" + "=" * 80)
    print("Starting sweep...")
    print("=" * 80)

    # Convert to format expected by run_sweep
    sweep_experiments = []
    for exp in experiments:
        sweep_exp = {
            'name': exp['name'],
            'learning_rate': exp['learning_rate'],
            'batch_size': exp['batch_size'],
            'encoder_init_scale': exp['encoder_init_scale'],
            'decoder_init_scale': exp['decoder_init_scale'],
            'epochs': exp['epochs']
        }
        sweep_experiments.append(sweep_exp)

    # Create custom sweep configs
    from experiments_claude.quick_hyperparam_sweep import run_sweep_experiment

    results = []
    for i, exp_params in enumerate(sweep_experiments, 1):
        name = exp_params.pop('name')
        epochs = exp_params.pop('epochs')

        print(f"\n[{i}/{len(sweep_experiments)}] Running: {name}")
        print(f"  LR: {exp_params['learning_rate']:.4f}, "
              f"Batch: {exp_params['batch_size']}, "
              f"Init: {exp_params['encoder_init_scale']:.2f}/{exp_params['decoder_init_scale']:.2f}, "
              f"Epochs: {epochs}")

        # Create config
        config = create_quick_sweep_config(**exp_params, epochs=epochs)

        # Run experiment
        result = run_sweep_experiment(config, name)
        result['name'] = name
        results.append(result)

        # Print result
        if result['success']:
            print(f"  ✓ Success: {result['training_time_seconds']:.1f}s, "
                  f"Loss: {result['final_test_loss']:.2f}, "
                  f"Clusters: {result['final_n_clusters']}, "
                  f"Silhouette: {result['final_silhouette']:.3f if result['final_silhouette'] else 'N/A'}")
        else:
            print(f"  ✗ Failed: {result['error']}")

        # Save intermediate results
        results_path = Path('experiments_claude/quick_diagnostic_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("QUICK DIAGNOSTIC SWEEP COMPLETE!")
    print("=" * 80)

    # Save final results
    results_path = Path('experiments_claude/quick_diagnostic_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Analyze
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    analyze_sweep_results(results)

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print()
    print("1. Visualize results:")
    print("   python experiments_claude/visualize_sweep.py experiments_claude/quick_diagnostic_results.json")
    print()
    print("2. Check individual experiment details:")
    print("   ls -lt experiments/sweep_quick_* | head -10")
    print()
    print("3. If results look promising, run full sweep:")
    print("   python -c \"from experiments_claude.quick_hyperparam_sweep import design_sweep, run_sweep; run_sweep(design_sweep())\"")
