"""
Flexible experiment runner for interactive hyperparameter exploration.

Provides utilities for:
- Quick config generation with parameter overrides
- Running experiments (full or quick test mode)
- Loading and analyzing results
- Comparing multiple experiments
"""

import json
import sys
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch

# Add parent directory to path to import mft_denoising modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from mft_denoising.config import ExperimentConfig


def create_config(
    base: str = "reference_3blob",
    **overrides
) -> Dict[str, Any]:
    """
    Create experiment config by loading base and applying overrides.

    Args:
        base: Name of base config file (without .json)
        **overrides: Parameter overrides using nested dict notation
                    Example: encoder_initialization_scale=0.05
                            learning_rate=0.01

    Returns:
        Config dictionary ready to save or run

    Example:
        >>> config = create_config('reference_3blob', encoder_initialization_scale=0.05)
        >>> config = create_config('reference_3blob', learning_rate=0.01, epochs=10)
    """
    base_path = Path(__file__).parent / "configs" / f"{base}.json"

    if not base_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_path}")

    with open(base_path) as f:
        config = json.load(f)

    # Apply overrides
    for key, value in overrides.items():
        # Handle nested keys like model.hidden_size
        if '.' in key:
            parts = key.split('.')
            target = config
            for part in parts[:-1]:
                target = target[part]
            target[parts[-1]] = value
        else:
            # Try to find the key in nested dicts
            found = False
            for section in ['model', 'training', 'loss', 'data']:
                if section in config and key in config[section]:
                    config[section][key] = value
                    found = True
                    break
            if not found:
                # Top-level key
                config[key] = value

    return config


def save_config(config: Dict[str, Any], name: str) -> Path:
    """Save config to experiments_claude/configs/ directory."""
    config_path = Path(__file__).parent / "configs" / f"{name}.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    return config_path


def run_experiment(
    config: Dict[str, Any],
    experiment_name: Optional[str] = None,
    wait: bool = True
) -> Optional[subprocess.Popen]:
    """
    Run an experiment with given config.

    Args:
        config: Configuration dictionary
        experiment_name: Name for this experiment (optional, will use config's name if not provided)
        wait: If True, wait for completion. If False, run in background and return process.

    Returns:
        None if wait=True, subprocess.Popen object if wait=False

    Example:
        >>> config = create_config('reference_3blob', encoder_initialization_scale=0.05)
        >>> run_experiment(config, 'test_init_005')  # Blocks until complete
        >>>
        >>> # Or run in background
        >>> process = run_experiment(config, 'test_init_005', wait=False)
        >>> # Do other work...
        >>> process.wait()  # Wait for completion later
    """
    if experiment_name:
        config['experiment_name'] = experiment_name

    # Save config temporarily
    temp_config_path = save_config(config, config['experiment_name'])

    # Run main.py with this config
    repo_root = Path(__file__).parent.parent
    cmd = [
        sys.executable,
        str(repo_root / "main.py"),
        str(temp_config_path)
    ]

    print(f"Launching experiment: {config['experiment_name']}")
    print(f"Config: {temp_config_path}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 80)

    if wait:
        subprocess.run(cmd, cwd=repo_root)
        return None
    else:
        process = subprocess.Popen(cmd, cwd=repo_root)
        print(f"Running in background (PID: {process.pid})")
        return process


def quick_run(
    config: Dict[str, Any],
    epochs: int = 5,
    n_train: int = 100000,
    experiment_name: Optional[str] = None
) -> None:
    """
    Run a quick test experiment with reduced epochs and training data.

    Useful for rapid iteration and testing parameter changes.

    Args:
        config: Base configuration
        epochs: Number of epochs (default: 5)
        n_train: Training samples (default: 100k, vs 300k in reference)
        experiment_name: Name for experiment

    Example:
        >>> config = create_config('reference_3blob', encoder_initialization_scale=0.1)
        >>> quick_run(config)  # Fast ~1-2 min test
    """
    # Override for quick testing
    config['training']['epochs'] = epochs
    config['data']['n_train'] = n_train

    if experiment_name:
        config['experiment_name'] = f"{experiment_name}_quick"
    elif '_quick' not in config['experiment_name']:
        config['experiment_name'] = f"{config['experiment_name']}_quick"

    run_experiment(config, wait=True)


def load_results(experiment_name: str) -> Dict[str, Any]:
    """
    Load results.json from an experiment.

    Args:
        experiment_name: Name of experiment (will search experiments/ directory)

    Returns:
        Dictionary containing results

    Example:
        >>> results = load_results('reference_3blob_20260117_123456')
        >>> print(f"Final test loss: {results['train_history'][-1]['test']['scaled_loss']}")
    """
    repo_root = Path(__file__).parent.parent
    experiments_dir = repo_root / "experiments"

    # Find matching experiment directory
    matching_dirs = list(experiments_dir.glob(f"{experiment_name}*"))

    if not matching_dirs:
        raise FileNotFoundError(f"No experiment found matching: {experiment_name}")

    # Sort by modification time (most recent first)
    matching_dirs = sorted(matching_dirs, key=lambda p: p.stat().st_mtime, reverse=True)

    if len(matching_dirs) > 1:
        print(f"Warning: Multiple experiments match '{experiment_name}':")
        for d in matching_dirs:
            print(f"  - {d.name}")
        print(f"Using most recent: {matching_dirs[0].name}")

    results_path = matching_dirs[0] / "results.json"

    with open(results_path) as f:
        return json.load(f)


def get_experiment_plot(experiment_name: str, plot_name: str = "encoder_decoder_pairs.png") -> Path:
    """
    Get path to a plot from an experiment.

    Args:
        experiment_name: Name of experiment
        plot_name: Name of plot file (default: final encoder-decoder pairs)

    Returns:
        Path to plot file

    Example:
        >>> plot_path = get_experiment_plot('reference_3blob_20260117_123456')
        >>> from IPython.display import Image; display(Image(plot_path))  # In notebook
    """
    repo_root = Path(__file__).parent.parent
    experiments_dir = repo_root / "experiments"

    matching_dirs = list(experiments_dir.glob(f"{experiment_name}*"))

    if not matching_dirs:
        raise FileNotFoundError(f"No experiment found matching: {experiment_name}")

    plot_path = matching_dirs[-1] / plot_name

    if not plot_path.exists():
        raise FileNotFoundError(f"Plot not found: {plot_path}")

    return plot_path


def compare_experiments(experiment_names: List[str]) -> None:
    """
    Compare multiple experiments side-by-side.

    Args:
        experiment_names: List of experiment names to compare

    Example:
        >>> compare_experiments(['reference_3blob', 'test_init_005', 'test_init_010'])
    """
    print("=" * 100)
    print("EXPERIMENT COMPARISON")
    print("=" * 100)

    for name in experiment_names:
        try:
            results = load_results(name)

            print(f"\n{name}:")
            print("-" * 80)

            # Extract final metrics
            if results['train_history']:
                final_epoch = results['train_history'][-1]
                print(f"  Epochs: {final_epoch['epoch']}")
                print(f"  Final train loss: {final_epoch['train']['scaled_loss']:.6f}")
                print(f"  Final test loss: {final_epoch['test']['scaled_loss']:.6f}")

            # Extract config
            config_path = Path(__file__).parent.parent / "experiments"
            matching_dirs = list(config_path.glob(f"{name}*"))
            if matching_dirs:
                config_file = matching_dirs[-1] / "config.json"
                with open(config_file) as f:
                    config = json.load(f)

                print(f"\n  Key parameters:")
                print(f"    encoder_init_scale: {config['model']['encoder_initialization_scale']}")
                print(f"    decoder_init_scale: {config['model']['decoder_initialization_scale']}")
                print(f"    decoder_reg: {config['training']['decoder_regularization']}")
                print(f"    lambda_on: {config['loss']['lambda_on']}")
                print(f"    learning_rate: {config['training']['learning_rate']}")

        except FileNotFoundError as e:
            print(f"\n{name}: NOT FOUND ({e})")

    print("\n" + "=" * 100)


def check_gpu() -> None:
    """Check GPU availability and memory."""
    if torch.cuda.is_available():
        print("GPU Status:")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"  Memory reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
        print(f"  Max memory allocated: {torch.cuda.max_memory_allocated(0) / 1e9:.2f} GB")
    else:
        print("WARNING: CUDA not available! Experiments will run on CPU (very slow)")


if __name__ == "__main__":
    # Quick demo
    print("Experiment Runner - Interactive Tools")
    print("=" * 80)

    check_gpu()

    print("\nExample usage:")
    print("""
    from experiment_runner import create_config, run_experiment, quick_run

    # Create config with modified parameter
    config = create_config('reference_3blob', encoder_initialization_scale=0.05)

    # Quick test (1-2 min)
    quick_run(config, experiment_name='test_init_005')

    # Full run (~10 min)
    run_experiment(config, 'test_init_005_full')

    # Load and analyze results
    results = load_results('test_init_005_full')
    print(results['train_history'][-1])

    # Compare experiments
    compare_experiments(['reference_3blob', 'test_init_005_full'])
    """)
