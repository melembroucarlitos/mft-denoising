"""
Tiny test sweep with minimal experiments for quick verification.

Tests sweep infrastructure with:
- Only 2-3 experiments
- Small batch size (1000 vs default 10000)
- Few epochs (3 vs default 12)
- Reduced diagnostic samples (1000 vs default 5000)

Should complete in ~5 minutes total.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments_claude.quick_hyperparam_sweep import (
    create_sweep_config,
    run_sweep,
    analyze_sweep_results
)


def design_tiny_sweep():
    """
    Design minimal sweep for testing.

    Just 3 experiments:
    1. Reference config (lr=0.02)
    2. Lower LR (lr=0.01)
    3. Higher LR (lr=0.04)

    All with small batch size and few epochs.
    """
    experiments = [
        {
            'name': 'tiny_ref_lr02',
            'learning_rate': 0.02,
            'batch_size': 1000,  # 10x smaller
            'encoder_init_scale': 0.03,
            'decoder_init_scale': 0.03,
            'epochs': 3  # 4x fewer
        },
        {
            'name': 'tiny_low_lr01',
            'learning_rate': 0.01,
            'batch_size': 1000,
            'encoder_init_scale': 0.03,
            'decoder_init_scale': 0.03,
            'epochs': 3
        },
        {
            'name': 'tiny_high_lr04',
            'learning_rate': 0.04,
            'batch_size': 1000,
            'encoder_init_scale': 0.03,
            'decoder_init_scale': 0.03,
            'epochs': 3
        }
    ]

    print(f"Designed {len(experiments)} tiny experiments")
    print(f"Estimated time: ~2 minutes per experiment = ~6 minutes total")

    return experiments


def create_tiny_sweep_config(
    learning_rate: float,
    batch_size: int,
    encoder_init_scale: float,
    decoder_init_scale: float,
    epochs: int
):
    """Create config for tiny sweep with reduced diagnostic overhead."""
    from experiments_claude.experiment_runner import create_config

    config = create_config('reference_3blob')

    # Set parameters
    config['training']['learning_rate'] = learning_rate
    config['training']['batch_size'] = batch_size
    config['training']['epochs'] = epochs
    config['training']['enable_diagnostics'] = True
    config['training']['diagnostic_sample_size'] = 1000  # Reduced from 5000
    config['training']['save_epoch_checkpoints'] = False

    config['model']['encoder_initialization_scale'] = encoder_init_scale
    config['model']['decoder_initialization_scale'] = decoder_init_scale

    return config


def run_tiny_sweep():
    """Run the tiny test sweep."""
    from experiments_claude.quick_hyperparam_sweep import run_sweep_experiment
    import json

    experiments = design_tiny_sweep()
    results = []

    print("\nStarting TINY TEST SWEEP...")
    print("=" * 80)

    for i, exp_params in enumerate(experiments, 1):
        name = exp_params.pop('name')
        epochs = exp_params.pop('epochs')

        print(f"\n[{i}/{len(experiments)}] Running: {name}")
        print(f"  LR: {exp_params['learning_rate']:.4f}, "
              f"Batch: {exp_params['batch_size']}, "
              f"Epochs: {epochs}")

        # Create config
        config = create_tiny_sweep_config(**exp_params, epochs=epochs)

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
        results_path = Path('experiments_claude/tiny_sweep_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("TINY SWEEP COMPLETE!")
    print(f"Results saved to: experiments_claude/tiny_sweep_results.json")

    # Analyze
    analyze_sweep_results(results)

    return results


if __name__ == "__main__":
    print("=" * 80)
    print("TINY TEST SWEEP - Quick verification of sweep infrastructure")
    print("=" * 80)
    print()
    print("This will run 3 minimal experiments:")
    print("  - Small batch size (1000)")
    print("  - Few epochs (3)")
    print("  - Reduced diagnostics (1000 samples)")
    print()
    print("Estimated completion time: ~6 minutes")
    print()

    input("Press Enter to start the tiny sweep, or Ctrl+C to cancel...")

    results = run_tiny_sweep()

    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print()
    print("1. Check results:")
    print("   cat experiments_claude/tiny_sweep_results.json")
    print()
    print("2. Visualize results:")
    print("   python experiments_claude/visualize_sweep.py experiments_claude/tiny_sweep_results.json")
    print()
    print("3. If everything looks good, run the full sweep:")
    print("   python -c \"from experiments_claude.quick_hyperparam_sweep import design_sweep, run_sweep; run_sweep(design_sweep())\"")
