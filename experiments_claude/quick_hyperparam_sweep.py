"""
Quick hyperparameter sweep for reference_3blob configuration.

Goal: Test stability and efficiency with 1-2 hour total budget.
Focus: Learning rate, batch size, initialization scale - the "fast knobs"

Strategy:
- Test ~20-25 configurations total
- Use reduced epochs (10-12 instead of 20) for speed
- Track: training time, final loss, blob quality (silhouette, n_clusters)
- Find: stable regions, potential speedups, failure modes
"""

import time
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, List
import itertools

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments_claude.experiment_runner import create_config, run_experiment
from mft_denoising.experiment import ExperimentConfig


def create_sweep_config(
    learning_rate: float,
    batch_size: int,
    encoder_init_scale: float,
    decoder_init_scale: float = None,
    epochs: int = 12  # Reduced from 20 for speed
) -> Dict[str, Any]:
    """
    Create a config variant for the sweep.

    Args:
        learning_rate: Adam learning rate
        batch_size: Training batch size
        encoder_init_scale: Encoder weight initialization scale
        decoder_init_scale: Decoder weight init scale (if None, use encoder_init_scale)
        epochs: Number of training epochs

    Returns:
        Config dictionary
    """
    if decoder_init_scale is None:
        decoder_init_scale = encoder_init_scale

    config = create_config('reference_3blob')

    # Modify parameters
    config['training']['learning_rate'] = learning_rate
    config['training']['batch_size'] = batch_size
    config['training']['epochs'] = epochs
    config['training']['enable_diagnostics'] = True  # Track blob formation
    config['training']['save_epoch_checkpoints'] = False  # Don't save checkpoints (save space/time)

    config['model']['encoder_initialization_scale'] = encoder_init_scale
    config['model']['decoder_initialization_scale'] = decoder_init_scale

    return config


def run_sweep_experiment(
    config: Dict[str, Any],
    name_suffix: str
) -> Dict[str, Any]:
    """
    Run a single sweep experiment and extract key metrics.

    Returns:
        Dictionary with:
        - training_time_seconds
        - final_train_loss
        - final_test_loss
        - final_n_clusters (from diagnostics)
        - final_silhouette (from diagnostics)
        - final_correlation
        - config_params (for reference)
    """
    start_time = time.time()

    experiment_name = f"sweep_{name_suffix}"

    try:
        # Run experiment using subprocess
        run_experiment(config, experiment_name, wait=True)

        # Load results from saved JSON
        from mft_denoising.experiment import ExperimentConfig
        from experiments_claude.experiment_runner import load_results

        result = load_results(experiment_name)
        training_time = time.time() - start_time

        # Extract final metrics
        final_epoch = result['training_history'][-1]

        metrics = {
            'success': True,
            'training_time_seconds': training_time,
            'final_train_loss': final_epoch['train']['loss'],
            'final_test_loss': final_epoch['test']['scaled_loss'],
            'final_n_clusters': final_epoch.get('diagnostics', {}).get('n_clusters_dbscan', 1),
            'final_silhouette': final_epoch.get('diagnostics', {}).get('silhouette_score', None),
            'final_correlation': final_epoch.get('diagnostics', {}).get('weight_correlation', None),
            'config_params': {
                'learning_rate': config['training']['learning_rate'],
                'batch_size': config['training']['batch_size'],
                'encoder_init_scale': config['model']['encoder_initialization_scale'],
                'decoder_init_scale': config['model']['decoder_initialization_scale'],
                'epochs': config['training']['epochs']
            }
        }

        return metrics

    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'training_time_seconds': time.time() - start_time,
            'config_params': {
                'learning_rate': config['training']['learning_rate'],
                'batch_size': config['training']['batch_size'],
                'encoder_init_scale': config['model']['encoder_initialization_scale'],
                'decoder_init_scale': config['model']['decoder_initialization_scale'],
                'epochs': config['training']['epochs']
            }
        }


def design_sweep(budget_hours: float = 1.5) -> List[Dict[str, Any]]:
    """
    Design sweep configurations with time budget.

    Strategy:
    1. LR sweep around reference (0.02): [0.005, 0.01, 0.02, 0.04, 0.08]
    2. Batch size sweep: [5120, 10240, 15360, 20480]
    3. Init scale sweep: [0.01, 0.02, 0.03, 0.05, 0.1]
    4. Combinations of promising values

    Estimated time per experiment: ~3-4 minutes (12 epochs vs 20)
    Budget: 1.5 hours = 90 minutes → ~22-30 experiments

    Returns:
        List of config parameter dicts
    """

    # Reference values
    ref_lr = 0.02
    ref_batch = 10000
    ref_init = 0.03

    experiments = []

    # 1. Learning rate sweep (5 experiments) - keep batch and init at reference
    for lr in [0.005, 0.01, 0.02, 0.04, 0.08]:
        experiments.append({
            'name': f'lr{lr:.3f}',
            'learning_rate': lr,
            'batch_size': ref_batch,
            'encoder_init_scale': ref_init,
            'decoder_init_scale': ref_init
        })

    # 2. Batch size sweep (4 experiments) - keep LR and init at reference
    for batch in [5120, 10240, 15360, 20480]:
        if batch != ref_batch:  # Skip reference (already in LR sweep)
            experiments.append({
                'name': f'batch{batch}',
                'learning_rate': ref_lr,
                'batch_size': batch,
                'encoder_init_scale': ref_init,
                'decoder_init_scale': ref_init
            })

    # 3. Init scale sweep (4 experiments) - keep LR and batch at reference
    for init in [0.01, 0.02, 0.05, 0.1]:
        if init != ref_init:  # Skip reference
            experiments.append({
                'name': f'init{init:.2f}',
                'learning_rate': ref_lr,
                'batch_size': ref_batch,
                'encoder_init_scale': init,
                'decoder_init_scale': init
            })

    # 4. Promising combinations (6 experiments)
    # High LR + large batch (faster training?)
    experiments.append({
        'name': 'fast_lr04_batch20k',
        'learning_rate': 0.04,
        'batch_size': 20480,
        'encoder_init_scale': ref_init,
        'decoder_init_scale': ref_init
    })

    # Low LR + small batch (more stable?)
    experiments.append({
        'name': 'stable_lr01_batch5k',
        'learning_rate': 0.01,
        'batch_size': 5120,
        'encoder_init_scale': ref_init,
        'decoder_init_scale': ref_init
    })

    # Large init + high LR (aggressive training)
    experiments.append({
        'name': 'aggressive_lr04_init01',
        'learning_rate': 0.04,
        'batch_size': ref_batch,
        'encoder_init_scale': 0.1,
        'decoder_init_scale': 0.1
    })

    # Small init + low LR (conservative)
    experiments.append({
        'name': 'conservative_lr01_init01',
        'learning_rate': 0.01,
        'batch_size': ref_batch,
        'encoder_init_scale': 0.01,
        'decoder_init_scale': 0.01
    })

    # Large batch + large init (efficiency test)
    experiments.append({
        'name': 'efficient_batch20k_init05',
        'learning_rate': ref_lr,
        'batch_size': 20480,
        'encoder_init_scale': 0.05,
        'decoder_init_scale': 0.05
    })

    # Asymmetric init (encoder larger than decoder)
    experiments.append({
        'name': 'asym_enc05_dec02',
        'learning_rate': ref_lr,
        'batch_size': ref_batch,
        'encoder_init_scale': 0.05,
        'decoder_init_scale': 0.02
    })

    # 5. Overtraining test (2 experiments) - run 20 epochs instead of 12
    experiments.append({
        'name': 'overtrain_lr02_ep20',
        'learning_rate': ref_lr,
        'batch_size': ref_batch,
        'encoder_init_scale': ref_init,
        'decoder_init_scale': ref_init,
        'epochs': 20
    })

    experiments.append({
        'name': 'overtrain_lr04_ep20',
        'learning_rate': 0.04,
        'batch_size': ref_batch,
        'encoder_init_scale': ref_init,
        'decoder_init_scale': ref_init,
        'epochs': 20
    })

    print(f"Designed {len(experiments)} experiments")
    print(f"Estimated time: {len(experiments) * 3.5:.1f} minutes ({len(experiments) * 3.5 / 60:.1f} hours)")

    return experiments


def run_sweep(experiments: List[Dict[str, Any]], save_results: bool = True) -> List[Dict[str, Any]]:
    """
    Run full hyperparameter sweep.

    Args:
        experiments: List of experiment parameter dicts
        save_results: If True, save results JSON after each experiment

    Returns:
        List of result dictionaries
    """
    results = []

    print(f"\nStarting sweep of {len(experiments)} experiments...")
    print("=" * 80)

    for i, exp_params in enumerate(experiments, 1):
        name = exp_params.pop('name')
        epochs = exp_params.pop('epochs', 12)

        print(f"\n[{i}/{len(experiments)}] Running: {name}")
        print(f"  LR: {exp_params['learning_rate']:.4f}, Batch: {exp_params['batch_size']}, "
              f"Init: {exp_params['encoder_init_scale']:.2f}/{exp_params['decoder_init_scale']:.2f}, "
              f"Epochs: {epochs}")

        # Create config
        config = create_sweep_config(**exp_params, epochs=epochs)

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
        if save_results:
            results_path = Path('experiments_claude/sweep_results_intermediate.json')
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("Sweep complete!")

    # Save final results
    if save_results:
        results_path = Path('experiments_claude/sweep_results_final.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {results_path}")

    return results


def analyze_sweep_results(results: List[Dict[str, Any]]) -> None:
    """
    Print analysis of sweep results.

    Shows:
    - Best performing configs (by loss, time, blob quality)
    - Failure modes
    - Stability regions
    - Speedup opportunities
    """
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]

    print("\n" + "=" * 80)
    print("SWEEP ANALYSIS")
    print("=" * 80)

    print(f"\nSuccess rate: {len(successful)}/{len(results)} ({100*len(successful)/len(results):.1f}%)")

    if failed:
        print(f"\nFailed experiments:")
        for r in failed:
            print(f"  - {r['name']}: {r['error']}")

    if not successful:
        print("\nNo successful experiments to analyze!")
        return

    # Sort by different metrics
    by_time = sorted(successful, key=lambda x: x['training_time_seconds'])
    by_loss = sorted(successful, key=lambda x: x['final_test_loss'])
    by_clusters = [r for r in successful if r['final_n_clusters'] >= 2]
    by_silhouette = sorted(
        [r for r in by_clusters if r['final_silhouette'] is not None],
        key=lambda x: x['final_silhouette'],
        reverse=True
    )

    print(f"\n🏆 FASTEST (training time):")
    for r in by_time[:3]:
        print(f"  {r['name']}: {r['training_time_seconds']:.1f}s "
              f"(LR={r['config_params']['learning_rate']:.3f}, Batch={r['config_params']['batch_size']})")

    print(f"\n📉 LOWEST LOSS:")
    for r in by_loss[:3]:
        print(f"  {r['name']}: Loss={r['final_test_loss']:.3f}, Time={r['training_time_seconds']:.1f}s "
              f"(LR={r['config_params']['learning_rate']:.3f})")

    print(f"\n🎯 BEST BLOB QUALITY (silhouette):")
    for r in by_silhouette[:3]:
        print(f"  {r['name']}: Sil={r['final_silhouette']:.3f}, Clusters={r['final_n_clusters']}, "
              f"Loss={r['final_test_loss']:.3f} "
              f"(LR={r['config_params']['learning_rate']:.3f}, Init={r['config_params']['encoder_init_scale']:.2f})")

    # Reference comparison
    ref_results = [r for r in successful if 'lr0.020' in r['name']]
    if ref_results:
        ref = ref_results[0]
        print(f"\n📊 REFERENCE (LR=0.02, Batch=10k, Init=0.03):")
        print(f"  Time: {ref['training_time_seconds']:.1f}s, Loss: {ref['final_test_loss']:.3f}, "
              f"Sil: {ref['final_silhouette']:.3f if ref['final_silhouette'] else 'N/A'}")

        # Find speedups
        speedups = [(r, ref['training_time_seconds'] / r['training_time_seconds'])
                    for r in successful
                    if r['training_time_seconds'] < ref['training_time_seconds']
                    and r['final_n_clusters'] >= 2]

        if speedups:
            speedups.sort(key=lambda x: x[1], reverse=True)
            print(f"\n⚡ SPEEDUPS vs REFERENCE:")
            for r, speedup in speedups[:3]:
                loss_ratio = r['final_test_loss'] / ref['final_test_loss']
                print(f"  {r['name']}: {speedup:.2f}x faster, Loss ratio: {loss_ratio:.2f}")


if __name__ == "__main__":
    print("Quick Hyperparameter Sweep for Reference 3-Blob")
    print("=" * 80)

    # Design sweep
    experiments = design_sweep(budget_hours=1.5)

    print("\nReady to run sweep.")
    print("This will take approximately 1-1.5 hours.")
    print("\nTo run:")
    print("  from experiments_claude.quick_hyperparam_sweep import run_sweep, analyze_sweep_results")
    print("  experiments = design_sweep()")
    print("  results = run_sweep(experiments)")
    print("  analyze_sweep_results(results)")
