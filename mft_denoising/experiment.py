"""
Experiment tracking and saving.

Handles saving experiment configurations, training metrics, and results to JSON files.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from mft_denoising.config import ExperimentConfig


class ExperimentTracker:
    """Tracks and saves experiment configuration and results."""
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize experiment tracker.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.output_dir = self._setup_output_dir()
        self.train_history: List[Dict[str, float]] = []
        self.start_time: Optional[float] = None
        
    def _setup_output_dir(self) -> Path:
        """Setup output directory for experiment."""
        if self.config.output_dir is not None:
            output_dir = Path(self.config.output_dir)
        else:
            # Auto-generate directory with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path("experiments") / f"{self.config.experiment_name}_{timestamp}"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    def start(self):
        """Mark the start of training."""
        self.start_time = time.time()
        
        # Save initial config
        config_path = self.output_dir / "config.json"
        self.config.save_json(config_path)
        print(f"Experiment configuration saved to: {config_path}")
        print(f"Output directory: {self.output_dir}")
    
    def log_epoch(self, epoch: int, train_metrics: Dict[str, float], test_metrics: Dict[str, float], model: Optional[Any] = None):
        """
        Log metrics for a single epoch.

        Args:
            epoch: Epoch number (1-indexed)
            train_metrics: Dictionary of training metrics (e.g., {"loss": 0.5, "scaled_loss": 0.4})
            test_metrics: Dictionary of test metrics (e.g., {"scaled_loss": 0.3})
            model: Optional model for computing diagnostics (if enable_diagnostics=True)
        """
        epoch_data = {
            "epoch": epoch,
            "train": train_metrics,
            "test": test_metrics,
        }

        # Compute real-time diagnostics if enabled
        if self.config.training.enable_diagnostics and model is not None:
            from mft_denoising.diagnostics import compute_lightweight_blob_metrics

            diagnostics = compute_lightweight_blob_metrics(
                model,
                n_samples=self.config.training.diagnostic_sample_size
            )
            epoch_data["diagnostics"] = diagnostics

            # Print summary for user feedback
            if diagnostics["silhouette_score"] is not None:
                print(f'  Diagnostics: clusters={diagnostics["n_clusters_dbscan"]}, '
                      f'silhouette={diagnostics["silhouette_score"]:.3f}, '
                      f'correlation={diagnostics["weight_correlation"]:.3f}')
            else:
                print(f'  Diagnostics: clusters={diagnostics["n_clusters_dbscan"]}, '
                      f'correlation={diagnostics["weight_correlation"]:.3f}')

        self.train_history.append(epoch_data)
    
    def save_results(self, final_metrics: Optional[Dict[str, Any]] = None, model_state: Optional[Dict] = None):
        """
        Save final results and training history.
        
        Args:
            final_metrics: Additional final metrics to save
            model_state: Model state dict to save (if save_model=True)
        """
        # Calculate training duration
        duration = None
        if self.start_time is not None:
            duration = time.time() - self.start_time
        
        # Compile results
        results = {
            "experiment_name": self.config.experiment_name,
            "output_dir": str(self.output_dir),
            "training_duration_seconds": duration,
            "training_history": self.train_history,
            "final_metrics": final_metrics or {},
        }
        
        # Save results JSON
        results_path = self.output_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {results_path}")
        
        # Save model if requested
        if self.config.save_model and model_state is not None:
            model_path = self.output_dir / "model.pth"
            import torch
            torch.save(model_state, model_path)
            print(f"Model saved to: {model_path}")
    
    def get_plot_path(self, plot_name: str) -> Path:
        """
        Get path for saving a plot.
        
        Args:
            plot_name: Name of the plot (e.g., "encoder_weights_histogram.png")
        
        Returns:
            Full path for the plot
        """
        return self.output_dir / plot_name
