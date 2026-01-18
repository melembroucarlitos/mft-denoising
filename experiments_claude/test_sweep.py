"""
Test run of hyperparameter sweep infrastructure.

Validates sweep runner with minimal compute:
- 1 reference config (3 epochs, reduced data)
- 2 tiny test configs (2 epochs, tiny data)

Total time: ~3-5 minutes
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments_claude.quick_hyperparam_sweep import (
    create_sweep_config,
    run_sweep_experiment,
    analyze_sweep_results
)


def create_test_experiments():
    """
    Create minimal test experiments to validate infrastructure.

    Returns:
        List of experiment parameter dicts
    """
    experiments = []

    # 1. Reference config (reduced)
    experiments.append({
        'name': 'test_reference',
        'learning_rate': 0.02,
        'batch_size': 10000,
        'encoder_init_scale': 0.03,
        'decoder_init_scale': 0.03,
        'epochs': 3,
        'n_train': 10000  # Much smaller dataset
    })

    # 2. High LR test
    experiments.append({
        'name': 'test_high_lr',
        'learning_rate': 0.08,
        'batch_size': 5000,
        'encoder_init_scale': 0.03,
        'decoder_init_scale': 0.03,
        'epochs': 2,
        'n_train': 5000
    })

    # 3. Large init test
    experiments.append({
        'name': 'test_large_init',
        'learning_rate': 0.02,
        'batch_size': 5000,
        'encoder_init_scale': 0.1,
        'decoder_init_scale': 0.1,
        'epochs': 2,
        'n_train': 5000
    })

    return experiments


def create_test_config(
    learning_rate: float,
    batch_size: int,
    encoder_init_scale: float,
    decoder_init_scale: float,
    epochs: int,
    n_train: int
):
    """Create test config with reduced dataset."""
    config = create_sweep_config(
        learning_rate=learning_rate,
        batch_size=batch_size,
        encoder_init_scale=encoder_init_scale,
        decoder_init_scale=decoder_init_scale,
        epochs=epochs
    )

    # Override dataset size
    config['data']['n_train'] = n_train
    config['data']['n_val'] = min(1000, n_train // 10)

    return config


def run_test_sweep():
    """
    Run test sweep and validate infrastructure.

    Returns:
        True if all tests pass, False otherwise
    """
    print("=" * 80)
    print("TEST SWEEP - Infrastructure Validation")
    print("=" * 80)
    print()

    experiments = create_test_experiments()

    print(f"Running {len(experiments)} test experiments...")
    print(f"Estimated time: 3-5 minutes\n")

    results = []
    all_passed = True

    for i, exp_params in enumerate(experiments, 1):
        name = exp_params.pop('name')
        n_train = exp_params.pop('n_train')

        print(f"\n[{i}/{len(experiments)}] Testing: {name}")
        print(f"  LR: {exp_params['learning_rate']:.4f}, Batch: {exp_params['batch_size']}, "
              f"Init: {exp_params['encoder_init_scale']:.2f}, Epochs: {exp_params['epochs']}, "
              f"N_train: {n_train}")

        # Create config
        config = create_test_config(**exp_params, n_train=n_train)

        # Run experiment
        result = run_sweep_experiment(config, name)
        result['name'] = name
        results.append(result)

        # Validate result
        if result['success']:
            print(f"  ✓ SUCCESS")
            print(f"    Time: {result['training_time_seconds']:.1f}s")
            print(f"    Final test loss: {result['final_test_loss']:.3f}")
            print(f"    Clusters: {result['final_n_clusters']}")
            if result['final_silhouette'] is not None:
                print(f"    Silhouette: {result['final_silhouette']:.3f}")
        else:
            print(f"  ✗ FAILED: {result['error']}")
            all_passed = False

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]

    print(f"\nSuccess rate: {len(successful)}/{len(results)}")

    if failed:
        print("\nFailed experiments:")
        for r in failed:
            print(f"  - {r['name']}: {r['error']}")

    if all_passed:
        print("\n✓ ALL TESTS PASSED")
        print("\nInfrastructure validated. Ready for full sweep.")
    else:
        print("\n✗ SOME TESTS FAILED")
        print("\nFix issues before running full sweep.")

    return all_passed, results


if __name__ == "__main__":
    print("Hyperparameter Sweep - Test Run")
    print()
    print("This validates the sweep infrastructure with minimal compute.")
    print("Total runtime: ~3-5 minutes")
    print()

    passed, results = run_test_sweep()

    if passed:
        print("\n" + "=" * 80)
        print("READY FOR FULL SWEEP")
        print("=" * 80)
        print("\nTo run full sweep:")
        print("  from experiments_claude.quick_hyperparam_sweep import design_sweep, run_sweep")
        print("  experiments = design_sweep(budget_hours=1.5)")
        print("  results = run_sweep(experiments)")

    sys.exit(0 if passed else 1)
