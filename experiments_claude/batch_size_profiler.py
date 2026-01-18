"""
Batch Size Profiler for finding optimal batch size for d=1024 experiments.

Runs quick experiments with different batch sizes to measure:
- Time per epoch
- GPU memory usage
- Final loss convergence
- Throughput (samples/second)

Generates markdown report with recommendations.
"""

import time
import json
import sys
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import traceback

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mft_denoising.config import ExperimentConfig
from mft_denoising.data import create_dataloaders
from mft_denoising.nn import TwoLayerNet
from mft_denoising.experiment import ExperimentTracker
from main import train


@dataclass
class BatchSizeProfile:
    """Results from profiling a single batch size."""
    batch_size: int
    epochs_completed: int
    avg_time_per_epoch: float  # seconds
    peak_gpu_memory_gb: float
    final_train_loss: float
    final_test_loss: float
    samples_per_second: float
    oom_error: bool = False
    error_message: Optional[str] = None


def profile_single_batch_size(
    config: ExperimentConfig,
    batch_size: int,
    test_epochs: int = 5
) -> BatchSizeProfile:
    """
    Profile a single batch size by running a short experiment.

    Args:
        config: Base experiment configuration
        batch_size: Batch size to test
        test_epochs: Number of epochs to run (default: 5)

    Returns:
        BatchSizeProfile with timing and memory stats
    """
    print(f"\n{'='*80}")
    print(f"Testing batch size: {batch_size}")
    print(f"{'='*80}")

    # Modify config for this test
    config.training.batch_size = batch_size
    config.training.epochs = test_epochs
    config.training.enable_diagnostics = False  # Disable for clean timing
    config.experiment_name = f"batch_profile_{batch_size}"
    config.save_plots = False  # Skip plotting for speed
    config.save_model = False  # Skip model saving for speed

    try:
        # Reset GPU memory stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        # Create dataloaders with specified batch size
        train_loader, test_loader = create_dataloaders(
            config.data,
            batch_size=batch_size
        )

        # Create model
        model = TwoLayerNet(
            input_size=config.data.d,
            hidden_size=config.model.hidden_size,
            encoder_initialization_scale=config.model.encoder_initialization_scale,
            decoder_initialization_scale=config.model.decoder_initialization_scale
        )

        # Create tracker
        tracker = ExperimentTracker(config)
        tracker.start()

        # Train and measure time
        start_time = time.time()
        model = train(model, train_loader, test_loader, config, tracker)
        end_time = time.time()

        total_time = end_time - start_time
        avg_time_per_epoch = total_time / test_epochs

        # Get memory stats
        peak_memory = 0.0
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1e9  # Convert to GB

        # Get final losses
        final_train_loss = tracker.train_history[-1]['train']['scaled_loss']
        final_test_loss = tracker.train_history[-1]['test']['scaled_loss']

        # Calculate throughput
        n_train_samples = config.data.n_train
        samples_per_second = (n_train_samples * test_epochs) / total_time

        print(f"✓ Success:")
        print(f"  Avg time/epoch: {avg_time_per_epoch:.2f}s")
        print(f"  Peak GPU memory: {peak_memory:.2f} GB")
        print(f"  Throughput: {samples_per_second:.0f} samples/sec")
        print(f"  Final train loss: {final_train_loss:.4f}")
        print(f"  Final test loss: {final_test_loss:.4f}")

        return BatchSizeProfile(
            batch_size=batch_size,
            epochs_completed=test_epochs,
            avg_time_per_epoch=avg_time_per_epoch,
            peak_gpu_memory_gb=peak_memory,
            final_train_loss=final_train_loss,
            final_test_loss=final_test_loss,
            samples_per_second=samples_per_second,
            oom_error=False
        )

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"✗ OOM Error: Batch size {batch_size} exceeds GPU memory")
            return BatchSizeProfile(
                batch_size=batch_size,
                epochs_completed=0,
                avg_time_per_epoch=0.0,
                peak_gpu_memory_gb=0.0,
                final_train_loss=0.0,
                final_test_loss=0.0,
                samples_per_second=0.0,
                oom_error=True,
                error_message="Out of memory"
            )
        else:
            raise

    except Exception as e:
        print(f"✗ Error: {str(e)}")
        traceback.print_exc()
        return BatchSizeProfile(
            batch_size=batch_size,
            epochs_completed=0,
            avg_time_per_epoch=0.0,
            peak_gpu_memory_gb=0.0,
            final_train_loss=0.0,
            final_test_loss=0.0,
            samples_per_second=0.0,
            oom_error=False,
            error_message=str(e)
        )


def profile_batch_sizes(
    config: ExperimentConfig,
    batch_sizes: List[int],
    test_epochs: int = 5,
    output_path: Optional[str] = None
) -> List[BatchSizeProfile]:
    """
    Profile multiple batch sizes and generate report.

    Args:
        config: Base experiment configuration
        batch_sizes: List of batch sizes to test (e.g., [2048, 5120, 10240, 20480])
        test_epochs: Number of epochs per test (default: 5)
        output_path: Path to save markdown report (default: auto-generate)

    Returns:
        List of BatchSizeProfile results

    Example:
        >>> from experiments_claude.experiment_runner import create_config
        >>> from experiments_claude.batch_size_profiler import profile_batch_sizes
        >>>
        >>> config = create_config('reference_3blob')
        >>> results = profile_batch_sizes(
        ...     config,
        ...     batch_sizes=[2048, 5120, 10240, 20480],
        ...     test_epochs=5,
        ...     output_path='experiments_claude/batch_profile_d1024.md'
        ... )
    """
    print(f"\n{'='*80}")
    print(f"BATCH SIZE PROFILING")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  d={config.data.d}, hidden_size={config.model.hidden_size}")
    print(f"  n_train={config.data.n_train}, n_val={config.data.n_val}")
    print(f"  Test epochs per batch size: {test_epochs}")
    print(f"  Batch sizes to test: {batch_sizes}")
    print(f"{'='*80}")

    results = []
    for batch_size in batch_sizes:
        profile = profile_single_batch_size(config, batch_size, test_epochs)
        results.append(profile)

        # Clean up between runs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Generate report
    if output_path is None:
        output_path = f"experiments_claude/batch_profile_d{config.data.d}_h{config.model.hidden_size}.md"

    generate_markdown_report(results, config, output_path)

    return results


def generate_markdown_report(
    results: List[BatchSizeProfile],
    config: ExperimentConfig,
    output_path: str
):
    """
    Generate markdown report with batch size profiling results.

    Args:
        results: List of BatchSizeProfile results
        config: Experiment configuration
        output_path: Path to save markdown file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write(f"# Batch Size Profile: d={config.data.d}, hidden={config.model.hidden_size}\n\n")

        f.write(f"**Configuration:**\n")
        f.write(f"- Input dimension (d): {config.data.d}\n")
        f.write(f"- Hidden size: {config.model.hidden_size}\n")
        f.write(f"- Training samples: {config.data.n_train}\n")
        f.write(f"- Validation samples: {config.data.n_val}\n")
        f.write(f"- Optimizer: {config.training.optimizer_type}\n")
        f.write(f"- Learning rate: {config.training.learning_rate}\n\n")

        f.write(f"**Test Parameters:**\n")
        f.write(f"- Epochs per test: {results[0].epochs_completed if results[0].epochs_completed > 0 else 'N/A'}\n")
        f.write(f"- GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n\n")

        f.write("## Results\n\n")

        # Table header
        f.write("| Batch Size | Time/Epoch | GPU Memory | Samples/Sec | Final Train Loss | Final Test Loss | Status |\n")
        f.write("|------------|------------|------------|-------------|------------------|-----------------|--------|\n")

        # Table rows
        for profile in results:
            if profile.oom_error:
                f.write(f"| {profile.batch_size:,} | OOM | - | - | - | - | ERROR |\n")
            elif profile.error_message:
                f.write(f"| {profile.batch_size:,} | ERROR | - | - | - | - | {profile.error_message[:20]} |\n")
            else:
                f.write(f"| {profile.batch_size:,} | "
                       f"{profile.avg_time_per_epoch:.2f}s | "
                       f"{profile.peak_gpu_memory_gb:.2f} GB | "
                       f"{profile.samples_per_second:.0f} | "
                       f"{profile.final_train_loss:.4f} | "
                       f"{profile.final_test_loss:.4f} | "
                       f"OK |\n")

        # Recommendation
        f.write("\n## Recommendation\n\n")

        # Find best batch size (highest throughput that didn't OOM)
        valid_results = [r for r in results if not r.oom_error and not r.error_message]

        if valid_results:
            best = max(valid_results, key=lambda x: x.samples_per_second)
            f.write(f"**Optimal batch size: {best.batch_size:,}**\n\n")
            f.write(f"- Best throughput: {best.samples_per_second:.0f} samples/sec\n")
            f.write(f"- Time per epoch: {best.avg_time_per_epoch:.2f}s\n")
            f.write(f"- GPU memory usage: {best.peak_gpu_memory_gb:.2f} GB\n")
            f.write(f"- Final test loss: {best.final_test_loss:.4f}\n\n")

            # Memory efficiency analysis
            f.write("### Memory Efficiency\n\n")
            for profile in valid_results:
                samples_per_gb = profile.samples_per_second / profile.peak_gpu_memory_gb if profile.peak_gpu_memory_gb > 0 else 0
                f.write(f"- Batch {profile.batch_size:,}: {samples_per_gb:.1f} samples/sec/GB\n")

        else:
            f.write("**No valid batch sizes found!**\n\n")
            f.write("All tested batch sizes either ran out of memory or encountered errors.\n")
            f.write("Try smaller batch sizes or reduce model complexity.\n")

        f.write("\n---\n\n")
        f.write(f"*Generated by batch_size_profiler.py*\n")

    print(f"\n{'='*80}")
    print(f"Report saved to: {output_path}")
    print(f"{'='*80}")


if __name__ == "__main__":
    """
    Command-line usage:
        python experiments_claude/batch_size_profiler.py <config_path>

    Example:
        python experiments_claude/batch_size_profiler.py experiments_claude/configs/reference_3blob.json
    """
    import argparse

    parser = argparse.ArgumentParser(description="Profile batch sizes for optimal GPU utilization")
    parser.add_argument("config", type=str, help="Path to base config JSON file")
    parser.add_argument("--batch-sizes", type=int, nargs="+",
                       default=[2048, 5120, 10240, 20480],
                       help="Batch sizes to test (default: 2048 5120 10240 20480)")
    parser.add_argument("--epochs", type=int, default=5,
                       help="Number of epochs per test (default: 5)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output markdown path (default: auto-generate)")

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    config = ExperimentConfig.load_json(config_path)

    # Run profiling
    results = profile_batch_sizes(
        config,
        batch_sizes=args.batch_sizes,
        test_epochs=args.epochs,
        output_path=args.output
    )

    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for profile in results:
        if profile.oom_error:
            print(f"Batch {profile.batch_size:,}: OOM")
        elif profile.error_message:
            print(f"Batch {profile.batch_size:,}: ERROR - {profile.error_message[:40]}")
        else:
            print(f"Batch {profile.batch_size:,}: "
                  f"{profile.avg_time_per_epoch:.2f}s/epoch, "
                  f"{profile.samples_per_second:.0f} samples/sec")
