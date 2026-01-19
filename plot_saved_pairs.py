#!/usr/bin/env python3
"""
Plot encoder-decoder weight pairs from saved .npz files.

This script can plot multiple epochs in parallel using multiprocessing.
Usage:
    python plot_saved_pairs.py <experiment_dir> [--epochs EPOCH_LIST] [--all] [--n-jobs N]
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import List, Optional
import sys

# Add parent directory to path to import from main
sys.path.insert(0, str(Path(__file__).parent))
from main import plot_encoder_decoder_pairs


def plot_single_epoch(args):
    """Plot a single epoch's weight pairs. Used for multiprocessing."""
    data_path, output_path = args
    try:
        plot_encoder_decoder_pairs(data_path=data_path, save_path=output_path)
        return f"Successfully plotted {output_path}"
    except Exception as e:
        return f"Error plotting {output_path}: {e}"


def find_epoch_files(experiment_dir: Path) -> List[Path]:
    """Find all encoder_decoder_pairs_epoch_*.npz files in experiment directory."""
    pattern = "encoder_decoder_pairs_epoch_*.npz"
    files = sorted(experiment_dir.glob(pattern))
    return files


def extract_epoch_number(filepath: Path) -> int:
    """Extract epoch number from filename like encoder_decoder_pairs_epoch_0001.npz"""
    name = filepath.stem  # Remove .npz extension
    epoch_str = name.split('_')[-1]  # Get last part (e.g., '0001')
    return int(epoch_str)


def main():
    parser = argparse.ArgumentParser(description='Plot encoder-decoder weight pairs from saved data')
    parser.add_argument('experiment_dir', type=str, help='Path to experiment directory')
    parser.add_argument('--epochs', type=str, help='Comma-separated list of epoch numbers (e.g., "1,2,3")')
    parser.add_argument('--all', action='store_true', help='Plot all epochs found in directory')
    parser.add_argument('--n-jobs', type=int, default=None, 
                       help=f'Number of parallel jobs (default: {cpu_count()})')
    parser.add_argument('--final-only', action='store_true', 
                       help='Only plot the final epoch (encoder_decoder_pairs.png)')
    
    args = parser.parse_args()
    
    experiment_dir = Path(args.experiment_dir)
    if not experiment_dir.exists():
        print(f"Error: Experiment directory not found: {experiment_dir}")
        return 1
    
    # Find all epoch files
    epoch_files = find_epoch_files(experiment_dir)
    
    if not epoch_files:
        print(f"No encoder_decoder_pairs_epoch_*.npz files found in {experiment_dir}")
        return 1
    
    # Determine which epochs to plot
    if args.final_only:
        # Plot only the final summary plot (if it exists as .npz, convert to .png)
        final_file = experiment_dir / 'encoder_decoder_pairs.npz'
        if final_file.exists():
            files_to_plot = [final_file]
            output_files = [experiment_dir / 'encoder_decoder_pairs.png']
        else:
            print("No encoder_decoder_pairs.npz found")
            return 1
    elif args.epochs:
        # Plot specific epochs
        epoch_nums = [int(e.strip()) for e in args.epochs.split(',')]
        files_to_plot = []
        output_files = []
        for epoch_num in epoch_nums:
            # Find file for this epoch
            matching = [f for f in epoch_files if extract_epoch_number(f) == epoch_num]
            if matching:
                files_to_plot.append(matching[0])
                output_files.append(experiment_dir / f'encoder_decoder_pairs_epoch_{epoch_num:04d}.png')
            else:
                print(f"Warning: No data file found for epoch {epoch_num}")
    elif args.all:
        # Plot all epochs
        files_to_plot = epoch_files
        output_files = [f.with_suffix('.png') for f in epoch_files]
    else:
        # Default: plot all epochs
        files_to_plot = epoch_files
        output_files = [f.with_suffix('.png') for f in epoch_files]
    
    if not files_to_plot:
        print("No files to plot")
        return 1
    
    print(f"Found {len(files_to_plot)} file(s) to plot")
    
    # Determine number of parallel jobs
    n_jobs = args.n_jobs if args.n_jobs is not None else cpu_count()
    n_jobs = min(n_jobs, len(files_to_plot))  # Don't use more jobs than files
    
    # Prepare arguments for multiprocessing
    plot_args = list(zip(files_to_plot, output_files))
    
    # Plot in parallel
    if n_jobs > 1 and len(files_to_plot) > 1:
        print(f"Plotting {len(files_to_plot)} files using {n_jobs} parallel workers...")
        with Pool(n_jobs) as pool:
            results = pool.map(plot_single_epoch, plot_args)
        for result in results:
            print(result)
    else:
        # Plot sequentially
        print(f"Plotting {len(files_to_plot)} file(s)...")
        for data_path, output_path in plot_args:
            result = plot_single_epoch((data_path, output_path))
            print(result)
    
    print(f"\nPlotting complete! Generated {len(output_files)} plot(s)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
