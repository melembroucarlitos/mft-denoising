"""
Optimizer implementations and factory.

Supports SGLD (Stochastic Gradient Langevin Dynamics) and ADAM optimizers.
Clean separation: choose either SGLD OR ADAM, no mixing.
"""

import torch
import numpy as np
from typing import Union

from mft_denoising.config import TrainingConfig


class SGLD:
    """Stochastic Gradient Langevin Dynamics optimizer"""
    
    def __init__(self, params, lr=1e-2, temperature=1.0):
        """
        Args:
            params: Model parameters to optimize
            lr: Learning rate
            temperature: Temperature parameter (noise_variance = 2 * lr * temperature)
        """
        self.params = list(params)
        self.lr = lr
        self.temperature = temperature
    
    def step(self):
        """Perform a single SGLD update step"""
        for p in self.params:
            if p.grad is None:
                continue
            
            # Standard gradient update
            d_p = p.grad.data
            
            # Add Langevin noise: noise ~ N(0, 2*lr*temperature)
            noise_std = np.sqrt(2 * self.lr * self.temperature)
            noise = torch.randn_like(p.data) * noise_std
            
            # SGLD update: θ_{t+1} = θ_t - lr * ∇L + noise
            p.data.add_(-self.lr * d_p + noise)
    
    def zero_grad(self):
        """Zero out gradients"""
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()


def create_optimizer(model, cfg: TrainingConfig, use_cuda: bool = False) -> Union[SGLD, torch.optim.Optimizer]:
    """
    Factory function to create optimizer based on TrainingConfig.
    
    Args:
        model: PyTorch model
        cfg: Training configuration
        use_cuda: Whether CUDA is available and being used
    
    Returns:
        Optimizer instance (SGLD, torch.optim.Adam, or torch.optim.AdamW)
    
    Note:
        Clean separation: returns either SGLD or ADAM/AdamW based on optimizer_type.
        Temperature parameter is only used for SGLD.
        For CUDA, uses AdamW with fused=True for better performance.
    """
    # Handle None learning_rate by using default
    learning_rate = cfg.learning_rate if cfg.learning_rate is not None else 1e-4
    temperature = cfg.temperature if cfg.temperature is not None else 0.0
    
    if cfg.optimizer_type == "sgld":
        return SGLD(model.parameters(), lr=learning_rate, temperature=temperature)
    elif cfg.optimizer_type == "adam":
        # Temperature is ignored for ADAM
        # Use AdamW with fused=True on CUDA for better performance
        if use_cuda:
            # Extract weight_decay from config if available, default to 0
            weight_decay = getattr(cfg, 'weight_decay', 0.0)
            return torch.optim.AdamW(
                model.parameters(), 
                lr=learning_rate,
                betas=(0.9, 0.999),
                weight_decay=weight_decay,
                fused=True
            )
        else:
            return torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer type: {cfg.optimizer_type}")
