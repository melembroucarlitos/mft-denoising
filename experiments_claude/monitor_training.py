"""
Real-time training monitor for visualizing blob formation metrics during training.

Watches results.json and updates plots as training progresses.
"""

import json
import time
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TrainingMonitor:
    """Monitor training progress by reading results.json in real-time."""

    def __init__(self, experiment_path: str):
        """
        Initialize training monitor.

        Args:
            experiment_path: Path to experiment directory or pattern (e.g., "experiments/test_*")
        """
        # Resolve experiment path
        experiment_path = Path(experiment_path)

        if not experiment_path.exists():
            # Try glob pattern
            matches = list(Path("experiments").glob(experiment_path.name))
            if matches:
                experiment_path = sorted(matches)[-1]  # Most recent
            else:
                raise FileNotFoundError(f"Experiment not found: {experiment_path}")

        self.experiment_dir = experiment_path
        self.results_path = self.experiment_dir / "results.json"
        self.last_mtime = 0
        self.last_epoch = 0

        print(f"Monitoring: {self.experiment_dir}")
        print(f"Results file: {self.results_path}")

    def read_results(self) -> Optional[Dict[str, Any]]:
        """
        Read results.json, handling partial writes.

        Returns:
            Results dictionary or None if file cannot be read
        """
        try:
            if not self.results_path.exists():
                return None

            with open(self.results_path, 'r') as f:
                data = json.load(f)
            return data
        except json.JSONDecodeError:
            # File being written, retry later
            return None
        except Exception as e:
            print(f"Error reading results: {e}")
            return None

    def extract_metrics(self, results: Dict[str, Any]) -> Dict[str, List]:
        """
        Extract time series metrics from results.

        Args:
            results: Results dictionary

        Returns:
            Dictionary of metric lists:
            {
                "epochs": [1, 2, 3, ...],
                "train_loss": [...],
                "test_loss": [...],
                "silhouette": [...],  # May contain None
                "n_clusters": [...],
                "correlation": [...],
            }
        """
        metrics = {
            "epochs": [],
            "train_loss": [],
            "test_loss": [],
            "silhouette": [],
            "n_clusters": [],
            "correlation": [],
        }

        for epoch_data in results.get("training_history", []):
            metrics["epochs"].append(epoch_data["epoch"])
            metrics["train_loss"].append(epoch_data["train"]["scaled_loss"])
            metrics["test_loss"].append(epoch_data["test"]["scaled_loss"])

            # Extract diagnostics if available
            if "diagnostics" in epoch_data:
                diag = epoch_data["diagnostics"]
                metrics["silhouette"].append(diag.get("silhouette_score"))
                metrics["n_clusters"].append(diag.get("n_clusters_dbscan", 1))
                metrics["correlation"].append(diag.get("weight_correlation", 0.0))
            else:
                # No diagnostics
                metrics["silhouette"].append(None)
                metrics["n_clusters"].append(None)
                metrics["correlation"].append(None)

        return metrics

    def plot_metrics(self, metrics: Dict[str, List], fig=None, axes=None) -> tuple:
        """
        Create or update 4-panel plot of training metrics.

        Args:
            metrics: Extracted metrics dictionary
            fig: Optional existing figure to update
            axes: Optional existing axes to update

        Returns:
            (fig, axes) tuple
        """
        if fig is None or axes is None:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f"Training Monitor: {self.experiment_dir.name}", fontsize=14)

        # Clear axes
        for ax in axes.flat:
            ax.clear()

        epochs = metrics["epochs"]

        # Panel 1: Silhouette Score
        ax = axes[0, 0]
        silhouettes = [s if s is not None else 0 for s in metrics["silhouette"]]
        has_silhouette = any(s is not None for s in metrics["silhouette"])
        if has_silhouette:
            ax.plot(epochs, silhouettes, marker='o', linestyle='-', linewidth=2, markersize=4, color='blue')
            ax.set_ylabel('Silhouette Score', fontsize=11)
            ax.set_title('Blob Quality (Silhouette Score)', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        else:
            ax.text(0.5, 0.5, 'No diagnostics available\n(enable_diagnostics=False)',
                   ha='center', va='center', transform=ax.transAxes, fontsize=11)
            ax.set_title('Blob Quality (Silhouette Score)', fontsize=12, fontweight='bold')

        # Panel 2: Number of Clusters
        ax = axes[0, 1]
        has_clusters = any(c is not None for c in metrics["n_clusters"])
        if has_clusters:
            clusters = [c if c is not None else 1 for c in metrics["n_clusters"]]
            ax.plot(epochs, clusters, marker='s', linestyle='steps-post', linewidth=2, markersize=4, color='green')
            ax.set_ylabel('Number of Clusters', fontsize=11)
            ax.set_title('Cluster Count (DBSCAN)', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=0)
        else:
            ax.text(0.5, 0.5, 'No diagnostics available',
                   ha='center', va='center', transform=ax.transAxes, fontsize=11)
            ax.set_title('Cluster Count (DBSCAN)', fontsize=12, fontweight='bold')

        # Panel 3: Weight Correlation
        ax = axes[1, 0]
        has_correlation = any(c is not None for c in metrics["correlation"])
        if has_correlation:
            correlations = [c if c is not None else 0 for c in metrics["correlation"]]
            ax.plot(epochs, correlations, marker='D', linestyle='-', linewidth=2, markersize=4, color='purple')
            ax.set_xlabel('Epoch', fontsize=11)
            ax.set_ylabel('Correlation', fontsize=11)
            ax.set_title('Encoder-Decoder Weight Correlation', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax.set_ylim(-1, 1)
        else:
            ax.text(0.5, 0.5, 'No diagnostics available',
                   ha='center', va='center', transform=ax.transAxes, fontsize=11)
            ax.set_xlabel('Epoch', fontsize=11)
            ax.set_title('Encoder-Decoder Weight Correlation', fontsize=12, fontweight='bold')

        # Panel 4: Train/Test Loss
        ax = axes[1, 1]
        ax.plot(epochs, metrics["train_loss"], label='Train Loss', marker='o', linestyle='-', linewidth=2, markersize=3, color='red')
        ax.plot(epochs, metrics["test_loss"], label='Test Loss', marker='s', linestyle='-', linewidth=2, markersize=3, color='orange')
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Scaled Loss', fontsize=11)
        ax.set_title('Training Progress', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

        plt.tight_layout()
        return fig, axes

    def start_live_plot(self, refresh_interval: int = 5, max_wait: int = 300):
        """
        Start live monitoring with automatic plot updates.

        Args:
            refresh_interval: Seconds between plot updates (default: 5)
            max_wait: Maximum seconds to wait for results.json before timing out (default: 300)

        """
        print(f"\nStarting live monitor...")
        print(f"Refresh interval: {refresh_interval}s")
        print(f"Press Ctrl+C to stop\n")

        fig, axes = None, None
        start_time = time.time()
        plt.ion()  # Interactive mode

        try:
            while True:
                # Check if results file exists
                if not self.results_path.exists():
                    elapsed = time.time() - start_time
                    if elapsed > max_wait:
                        print(f"Timeout: No results file after {max_wait}s")
                        break
                    print(f"Waiting for experiment to start... ({elapsed:.0f}s)")
                    time.sleep(refresh_interval)
                    continue

                # Read results
                results = self.read_results()
                if results is None:
                    time.sleep(1)
                    continue

                # Check if new epochs appeared
                current_epoch = len(results.get("training_history", []))
                if current_epoch > self.last_epoch:
                    self.last_epoch = current_epoch
                    print(f"Epoch {current_epoch} completed, updating plot...")

                    # Extract and plot metrics
                    metrics = self.extract_metrics(results)
                    fig, axes = self.plot_metrics(metrics, fig, axes)

                    plt.pause(0.1)  # Update display

                # Check if training completed
                if results.get("final_metrics") is not None:
                    print("\nTraining completed!")
                    break

                time.sleep(refresh_interval)

        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")

        finally:
            plt.ioff()
            if fig is not None:
                plt.show()  # Keep final plot open

    def generate_static_plot(self, output_path: Optional[str] = None):
        """
        Generate a static plot from completed training.

        Args:
            output_path: Path to save plot (default: experiment_dir/training_metrics.png)
        """
        results = self.read_results()
        if results is None:
            print("Error: Cannot read results.json")
            return

        metrics = self.extract_metrics(results)
        fig, axes = self.plot_metrics(metrics)

        if output_path is None:
            output_path = self.experiment_dir / "training_metrics.png"
        else:
            output_path = Path(output_path)

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
        plt.close(fig)


def main():
    """Command-line interface for training monitor."""
    parser = argparse.ArgumentParser(description="Monitor training progress in real-time")
    parser.add_argument("experiment", type=str,
                       help="Experiment directory path or pattern (e.g., 'experiments/test_*')")
    parser.add_argument("--interval", type=int, default=5,
                       help="Refresh interval in seconds (default: 5)")
    parser.add_argument("--static", action="store_true",
                       help="Generate static plot instead of live monitoring")
    parser.add_argument("--output", type=str, default=None,
                       help="Output path for static plot (only with --static)")

    args = parser.parse_args()

    try:
        monitor = TrainingMonitor(args.experiment)

        if args.static:
            monitor.generate_static_plot(args.output)
        else:
            monitor.start_live_plot(refresh_interval=args.interval)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
