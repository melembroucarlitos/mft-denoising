"""
Configuration dataclasses for all hyperparameters.

Supports both programmatic configuration and loading from JSON files.
"""

from dataclasses import dataclass, asdict
from typing import Optional, Literal
import json
from pathlib import Path

from mft_denoising.data import DataConfig


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    hidden_size: int = 128
    encoder_initialization_scale: float = 1.0
    decoder_initialization_scale: float = 1.0


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    optimizer_type: Literal["sgld", "adam"] = "sgld"
    learning_rate: float = 1e-4
    temperature: float = 0.0  # Only used for SGLD
    epochs: int = 1
    batch_size: int = 128
    encoder_regularization: float = 0.0
    decoder_regularization: float = 0.0
    two_stage_training: bool = False  # Enable two-stage training with frozen GMM encoder
    stage2_epochs: Optional[int] = None  # Epochs for stage 2 (defaults to epochs if None)
    num_clusters: int = 3  # Number of Gaussian components for clustering
    num_traces: int = 1  # Number of traces (frozen encoder samples) to train in stage 2
    enable_diagnostics: bool = False  # Enable real-time blob formation diagnostics
    diagnostic_sample_size: int = 5000  # Number of weight pairs to sample per epoch for diagnostics
    save_epoch_checkpoints: bool = False  # Save model checkpoint after each epoch


@dataclass
class LossConfig:
    """Loss function configuration."""
    loss_type: Literal["scaled_mse", "logistic"] = "scaled_mse"
    lambda_on: float = 10.0


@dataclass
class ExperimentConfig:
    """Complete experiment configuration combining all configs."""
    model: ModelConfig
    training: TrainingConfig
    loss: LossConfig
    data: DataConfig
    experiment_name: str = "experiment"
    output_dir: Optional[str] = None  # None = auto-generate with timestamp
    save_model: bool = True
    save_plots: bool = True

    def to_dict(self) -> dict:
        """Convert config to dictionary for JSON serialization."""
        return {
            "model": asdict(self.model),
            "training": asdict(self.training),
            "loss": asdict(self.loss),
            "data": asdict(self.data),
            "experiment_name": self.experiment_name,
            "output_dir": self.output_dir,
            "save_model": self.save_model,
            "save_plots": self.save_plots,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        """Create config from dictionary (e.g., loaded from JSON)."""
        return cls(
            model=ModelConfig(**d["model"]),
            training=TrainingConfig(**d["training"]),
            loss=LossConfig(**d["loss"]),
            data=DataConfig(**d["data"]),
            experiment_name=d.get("experiment_name", "experiment"),
            output_dir=d.get("output_dir", None),
            save_model=d.get("save_model", True),
            save_plots=d.get("save_plots", True),
        )

    def save_json(self, path: Path):
        """Save config to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: Path) -> "ExperimentConfig":
        """Load config from JSON file."""
        path = Path(path)
        with open(path, "r") as f:
            d = json.load(f)
        return cls.from_dict(d)
