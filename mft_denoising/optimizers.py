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


def create_optimizer(model, cfg: TrainingConfig) -> Union[SGLD, torch.optim.Optimizer]:
    """
    Factory function to create optimizer based on TrainingConfig.
    
    Args:
        model: PyTorch model
        cfg: Training configuration
    
    Returns:
        Optimizer instance (SGLD or torch.optim.Adam)
    
    Note:
        Clean separation: returns either SGLD or ADAM based on optimizer_type.
        Temperature parameter is only used for SGLD.
    """
    if cfg.optimizer_type == "sgld":
        return SGLD(model.parameters(), lr=cfg.learning_rate, temperature=cfg.temperature)
    elif cfg.optimizer_type == "adam":
        # Temperature is ignored for ADAM
        return torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    else:
        raise ValueError(f"Unknown optimizer type: {cfg.optimizer_type}")
