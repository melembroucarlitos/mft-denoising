"""
Experiment tracking and saving.

Handles saving experiment configurations, training metrics, and results to JSON files.
"""

import json
import time
import numpy as np
import random
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
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
        self.colored_pairs: Optional[List[Tuple[int, int]]] = None  # Will be initialized in start() or initialize_colored_pairs()
        
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
    
    def initialize_colored_pairs(self, hidden_size: int, input_size: int, seed: Optional[int] = None):
        """
        Initialize colored pairs for visualization.
        
        Args:
            hidden_size: Hidden layer size (number of encoder neurons)
            input_size: Input dimension (d)
            seed: Random seed for pair selection (uses config seed if None)
        """
        if seed is None:
            seed = self.config.data.seed
        
        random.seed(seed)
        np.random.seed(seed)
        
        C = self.config.training.colored_points_count
        total_pairs = hidden_size * input_size
        
        # Randomly select C pairs (i, j) without replacement
        if C >= total_pairs:
            # If C >= total pairs, select all pairs
            pairs = [(i, j) for i in range(hidden_size) for j in range(input_size)]
        else:
            # Random selection without replacement
            indices = np.random.choice(total_pairs, C, replace=False)
            pairs = []
            for idx in indices:
                # Convert numpy int64 to Python int for JSON serializability
                i = int(idx // input_size)
                j = int(idx % input_size)
                pairs.append((i, j))
        
        self.colored_pairs = pairs
        print(f"Initialized {len(self.colored_pairs)} colored pairs for visualization")
    
    def start(self, model: Optional[Any] = None):
        """
        Mark the start of training.
        
        Args:
            model: Optional model to initialize colored pairs from (if None, will be initialized later)
        """
        self.start_time = time.time()
        
        # Initialize colored pairs if model is provided
        if model is not None:
            hidden_size = model.fc1.weight.shape[0]
            input_size = model.fc1.weight.shape[1]
            self.initialize_colored_pairs(hidden_size, input_size)
        
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
        # Initialize colored pairs if not already done and model is provided
        if self.colored_pairs is None and model is not None:
            hidden_size = model.fc1.weight.shape[0]
            input_size = model.fc1.weight.shape[1]
            self.initialize_colored_pairs(hidden_size, input_size)
        
        # Compute encoder_above_0_5_count if model is provided
        encoder_above_0_5_count = None
        if model is not None:
            encoder_weights = model.fc1.weight.data.cpu().numpy()
            encoder_pairs = encoder_weights.flatten()
            # Convert to Python int to ensure JSON serializability
            count_result = np.sum(encoder_pairs > 0.5)
            encoder_above_0_5_count = int(count_result.item() if hasattr(count_result, 'item') else count_result)
            # Add to both train and test metrics
            train_metrics = train_metrics.copy()
            test_metrics = test_metrics.copy()
            train_metrics["encoder_above_0_5_count"] = encoder_above_0_5_count
            test_metrics["encoder_above_0_5_count"] = encoder_above_0_5_count
        
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

        # Save per-epoch checkpoint if enabled
        if self.config.training.save_epoch_checkpoints and model is not None:
            checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch}.pth"
            import torch
            torch.save(model.state_dict(), checkpoint_path)
            print(f'  Checkpoint saved: {checkpoint_path.name}')

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
        
        # Helper function to convert numpy types to Python native types for JSON serialization
        def convert_to_native(obj):
            """Recursively convert numpy types to Python native types."""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_native(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_to_native(item) for item in obj)
            else:
                return obj
        
        # Compile results
        results = {
            "experiment_name": self.config.experiment_name,
            "output_dir": str(self.output_dir),
            "training_duration_seconds": duration,
            "training_history": convert_to_native(self.train_history),
            "final_metrics": convert_to_native(final_metrics or {}),
        }
        
        # Add colored pairs to results if initialized
        if self.colored_pairs is not None:
            results["colored_pairs"] = convert_to_native(self.colored_pairs)
        
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
