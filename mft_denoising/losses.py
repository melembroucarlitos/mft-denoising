"""
Loss function implementations and factory.

Supports multiple loss types including scaled MSE and logistic loss.
"""

import torch
from typing import Optional, Tuple, Callable

from mft_denoising.config import LossConfig


def scaled_mse_loss(
    output: torch.Tensor,
    label: torch.Tensor,
    lambda_on: float,
    mask_on: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Custom loss function that separately weights on/off positions using MSE.
    
    Args:
        output: Model predictions (B, d)
        label: Clean sparse signal (B, d)
        lambda_on: Weight for active position loss
        mask_on: Binary mask of active positions (B, d). If None, uses label as mask.
    
    Returns:
        total_loss, loss_on_w, loss_off
    """
    if mask_on is None:
        mask_on = label
    
    err_on = (output - label) * mask_on
    err_off = output * (1.0 - mask_on)
    loss_on_w = lambda_on * err_on.pow(2).sum(dim=1).mean()
    loss_off = err_off.pow(2).sum(dim=1).mean()
    return loss_on_w + loss_off, loss_on_w, loss_off


def logistic_loss(
    output: torch.Tensor,
    label: torch.Tensor,
    lambda_on: float,
    mask_on: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Logistic loss function (placeholder for future implementation).
    
    Args:
        output: Model predictions (B, d)
        label: Clean sparse signal (B, d)
        lambda_on: Weight for active position loss
        mask_on: Binary mask of active positions (B, d). If None, uses label as mask.
    
    Returns:
        total_loss, loss_on_w, loss_off
    """
    # Placeholder implementation - to be implemented
    raise NotImplementedError("Logistic loss not yet implemented")


def create_loss_function(cfg: LossConfig) -> Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Factory function to create loss function based on LossConfig.
    
    Args:
        cfg: Loss configuration
    
    Returns:
        Loss function that takes (output, label, mask_on) and returns (total_loss, loss_on, loss_off)
    """
    if cfg.loss_type == "scaled_mse":
        def loss_fn(output, label, mask_on=None):
            return scaled_mse_loss(output, label, cfg.lambda_on, mask_on)
        return loss_fn
    elif cfg.loss_type == "logistic":
        def loss_fn(output, label, mask_on=None):
            return logistic_loss(output, label, cfg.lambda_on, mask_on)
        return loss_fn
    else:
        raise ValueError(f"Unknown loss type: {cfg.loss_type}")
