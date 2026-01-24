#!/usr/bin/env python3
"""
Temperature sweep script for SGLD-Adam two-stage training.

Sweeps over different sgld_temperature values while keeping all other parameters constant.
"""

import sys
import json
import subprocess
from pathlib import Path
from typing import List

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mft_denoising.config import ExperimentConfig

# Temperature values to sweep over (roughly dividing by 3 each time)
TEMPERATURE_VALUES = [
    0.0003,
    0.0001,
    0.000033,
    0.000011,
    0.0000036,
    0.0000012
]


def load_base_config(config_path: Path) -> ExperimentConfig:
    """Load the base configuration file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Base config file not found: {config_path}")
    
    return ExperimentConfig.load_json(config_path)


def create_temp_config(base_config: ExperimentConfig, temperature: float, output_dir: Path) -> Path:
    """
    Create a temporary config file with modified temperature.
    
    Args:
        base_config: Base experiment configuration
        temperature: Temperature value to set
        output_dir: Directory to save temporary config
    
    Returns:
        Path to temporary config file
    """
    # Create a copy of the config to avoid modifying the original
    import copy
    config = copy.deepcopy(base_config)
    
    # Modify the temperature
    config.training.sgld_temperature = temperature
    
    # Modify experiment name to include temperature for easy identification
    base_name = config.experiment_name
    # Format temperature as string without scientific notation for readability
    temp_str = f"{temperature:.9f}".rstrip('0').rstrip('.')
    config.experiment_name = f"{base_name}_T{temp_str}"
    
    # Create temp config file with temperature in filename
    temp_config_path = output_dir / f"config_temp_{temperature:.9f}.json"
    config.save_json(temp_config_path)
    
    return temp_config_path


def run_experiment(config_path: Path, script_path: Path) -> subprocess.CompletedProcess:
    """
    Run a single experiment with the given config.
    
    Args:
        config_path: Path to config file
        script_path: Path to training script
    
    Returns:
        CompletedProcess from subprocess.run
    """
    cmd = [sys.executable, str(script_path), str(config_path)]
    print(f"\n{'='*80}")
    print(f"Running experiment with config: {config_path.name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    result = subprocess.run(
        cmd,
        cwd=script_path.parent,
        capture_output=False,  # Show output in real-time
        text=True
    )
    
    return result


def main():
    """Main sweep function."""
    # Paths
    script_dir = Path(__file__).parent
    base_config_path = script_dir / "config_sgld_adam_t_sweep_base.json"
    train_script_path = script_dir / "train_sgld_adam.py"
    sweep_output_dir = script_dir / "sweep_configs"
    
    # Create output directory for temp configs
    sweep_output_dir.mkdir(exist_ok=True)
    
    # Check if base config exists
    if not base_config_path.exists():
        print(f"Error: Base config file not found: {base_config_path}")
        print(f"Please create {base_config_path} with your base configuration.")
        sys.exit(1)
    
    # Load base config
    print(f"Loading base config from: {base_config_path}")
    base_config = load_base_config(base_config_path)
    
    print(f"\nTemperature sweep values: {TEMPERATURE_VALUES}")
    print(f"Total experiments: {len(TEMPERATURE_VALUES)}\n")
    
    # Store results
    results = []
    
    # Run experiments for each temperature
    for idx, temperature in enumerate(TEMPERATURE_VALUES, 1):
        print(f"\n{'#'*80}")
        print(f"Experiment {idx}/{len(TEMPERATURE_VALUES)}: Temperature = {temperature}")
        print(f"{'#'*80}")
        
        try:
            # Create temp config with this temperature
            temp_config_path = create_temp_config(base_config, temperature, sweep_output_dir)
            
            # Run experiment
            result = run_experiment(temp_config_path, train_script_path)
            
            # Record result
            results.append({
                "temperature": temperature,
                "config_path": str(temp_config_path),
                "success": result.returncode == 0,
                "return_code": result.returncode
            })
            
            if result.returncode == 0:
                print(f"\n✓ Experiment completed successfully for temperature {temperature}")
            else:
                print(f"\n✗ Experiment failed for temperature {temperature} (return code: {result.returncode})")
        
        except Exception as e:
            print(f"\n✗ Error running experiment for temperature {temperature}: {e}")
            results.append({
                "temperature": temperature,
                "success": False,
                "error": str(e)
            })
    
    # Print summary
    print(f"\n{'='*80}")
    print("SWEEP SUMMARY")
    print(f"{'='*80}")
    
    successful = [r for r in results if r.get("success", False)]
    failed = [r for r in results if not r.get("success", False)]
    
    print(f"Total experiments: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        print(f"\nSuccessful temperatures:")
        for r in successful:
            print(f"  T = {r['temperature']:.9f}")
    
    if failed:
        print(f"\nFailed temperatures:")
        for r in failed:
            error_msg = r.get("error", f"Return code: {r.get('return_code', 'unknown')}")
            print(f"  T = {r['temperature']:.9f}: {error_msg}")
    
    # Save results to JSON
    results_file = sweep_output_dir / "sweep_results.json"
    with open(results_file, "w") as f:
        json.dump({
            "temperature_values": TEMPERATURE_VALUES,
            "results": results,
            "summary": {
                "total": len(results),
                "successful": len(successful),
                "failed": len(failed)
            }
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
