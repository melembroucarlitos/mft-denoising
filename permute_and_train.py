#!/usr/bin/env python3
"""
permute_and_train.py

Load a trained model, permute its encoder-decoder weight pairs, and train a new model
starting from the permuted weights.

Usage:
    python permute_and_train.py --source_checkpoint experiments/exp_*/model.pth --config configs/standard_gpu_big.json --seed 123
"""

import argparse
import sys
from pathlib import Path
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from mft_denoising.nn import TwoLayerNet
from mft_denoising.config import ExperimentConfig
from mft_denoising.permutation import load_and_permute_model
from mft_denoising.data import create_dataloaders
from mft_denoising.experiment import ExperimentTracker
from main import train


def main():
    parser = argparse.ArgumentParser(
        description='Permute encoder-decoder weight pairs and train a new model'
    )
    
    parser.add_argument(
        '--source_checkpoint',
        type=str,
        required=True,
        help='Path to source model checkpoint (.pth file)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to experiment config JSON file'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for permutation (default: random)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for new experiment (default: auto-generate)'
    )
    
    parser.add_argument(
        '--save_permuted_checkpoint',
        action='store_true',
        help='Save the permuted model checkpoint before training'
    )
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return 1
    
    config = ExperimentConfig.load_json(config_path)
    
    # Override output directory if specified
    if args.output_dir:
        config.output_dir = args.output_dir
    
    # Determine device
    device_str = config.data.device
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Using CPU.")
        device_str = "cpu"
        config.data.device = "cpu"
    device = torch.device(device_str)
    
    # Load source checkpoint
    checkpoint_path = Path(args.source_checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return 1
    
    print(f"Loading source model from: {checkpoint_path}")
    print(f"Config: d={config.data.d}, hidden_size={config.model.hidden_size}")
    print(f"Permutation seed: {args.seed if args.seed is not None else 'random'}")
    
    # Create permuted model
    permuted_model = load_and_permute_model(
        checkpoint_path=checkpoint_path,
        config=config,
        seed=args.seed,
        device=device
    )
    
    print("Permuted model created successfully")
    
    # Save permuted checkpoint if requested
    if args.save_permuted_checkpoint:
        # Create output directory if it doesn't exist
        if config.output_dir:
            output_dir = Path(config.output_dir)
        else:
            # Auto-generate output directory
            from mft_denoising.experiment import ExperimentTracker
            tracker = ExperimentTracker(config)
            output_dir = tracker.output_dir
        
        output_dir.mkdir(parents=True, exist_ok=True)
        permuted_checkpoint_path = output_dir / "permuted_initial_model.pth"
        
        # Save permuted model state
        torch.save(permuted_model.state_dict(), permuted_checkpoint_path)
        print(f"Permuted model saved to: {permuted_checkpoint_path}")
    
    # Create dataloaders
    train_loader, test_loader = create_dataloaders(
        config.data,
        batch_size=config.training.batch_size
    )
    
    # Create experiment tracker
    tracker = ExperimentTracker(config)
    
    print(f"\nStarting training with permuted weights...")
    print(f"Output directory: {tracker.output_dir}")
    
    # Train the permuted model
    trained_model = train(
        model=permuted_model,
        train_loader=train_loader,
        test_loader=test_loader,
        config=config,
        tracker=tracker,
    )
    
    print("\nTraining complete!")
    
    # Save final results
    model_state = trained_model.state_dict() if config.save_model else None
    tracker.save_results(final_metrics={}, model_state=model_state)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
