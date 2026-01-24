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


def scaled_lp_loss(
    output: torch.Tensor,
    label: torch.Tensor,
    lambda_on: float,
    mask_on: Optional[torch.Tensor] = None,
    p: float = 2.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Custom loss function that separately weights on/off positions using Lp norm.
    
    Generalizes scaled_mse_loss to use |error|^p instead of |error|^2.
    For p=2, this is equivalent to scaled_mse_loss.
    
    Args:
        output: Model predictions (B, d)
        label: Clean sparse signal (B, d)
        lambda_on: Weight for active position loss
        mask_on: Binary mask of active positions (B, d). If None, uses label as mask.
        p: Power parameter for Lp norm (default: 2.0, which gives MSE)
    
    Returns:
        total_loss, loss_on_w, loss_off
    """
    if mask_on is None:
        mask_on = label
    
    err_on = (output - label) * mask_on
    err_off = output * (1.0 - mask_on)
    # Use abs() for numerical stability, though for even p it's not strictly necessary
    loss_on_w = lambda_on * err_on.abs().pow(p).sum(dim=1).mean()
    loss_off = err_off.abs().pow(p).sum(dim=1).mean()
    return loss_on_w + loss_off, loss_on_w, loss_off


def logistic_loss(
    output: torch.Tensor,
    label: torch.Tensor,
    lambda_on: float,
    mask_on: Optional[torch.Tensor] = None,
    logsumexp_scale: Optional[float] = None,
    lambda_off: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Log-sum-exp loss function with masked on/off coordinates.
    
    Computes: log(sum_i [exp(lambda_on * (f(x)_i - x*_i)^2) * mask_on[i] + 
                            exp(lambda_off * f(x)_i^2) * (1 - mask_on[i])])
    
    This loss uses different temperature parameters for signal (on) and off coordinates:
    - lambda_on: Inverse temperature for on-coordinates (where target is non-zero)
    - lambda_off: Inverse temperature for off-coordinates (where target is zero)
    
    The loss is scaled by logsumexp_scale (defaults to dimension d if None) to
    compensate for the fact that log-sum-exp doesn't scale with dimension.
    
    Args:
        output: Model predictions (B, d)
        label: Clean sparse signal (B, d)
        lambda_on: Inverse temperature parameter for on-coordinates
        mask_on: Binary mask of active positions (B, d). If None, uses label as mask.
        logsumexp_scale: Scaling factor. If None, defaults to dimension d (output.shape[-1])
        lambda_off: Inverse temperature parameter for off-coordinates (default: 1.0)
    
    Returns:
        (total_loss, loss_on, loss_off)
        - total_loss: Scaled combined log-sum-exp loss averaged over batch
        - loss_on: Scaled log-sum-exp over on-coordinates (diagnostic)
        - loss_off: Scaled log-sum-exp over off-coordinates (diagnostic)
    """
    # Handle mask: use label if mask_on is None
    if mask_on is None:
        mask_on = label
    
    # Compute squared errors
    # For on coordinates: (output - label)^2
    # For off coordinates: output^2 (since target is 0)
    squared_errors = (output - label) ** 2
    squared_errors_off = output ** 2
    
    # Create temperature-scaled terms per coordinate
    scaled_on = lambda_on * squared_errors * mask_on
    scaled_off = lambda_off * squared_errors_off * (1.0 - mask_on)
    combined_scaled = scaled_on + scaled_off
    
    # Compute combined log-sum-exp per sample
    log_sum_exp_combined = torch.logsumexp(combined_scaled, dim=1)
    
    # Compute separate diagnostic terms for on and off coordinates
    # Use large negative value for masked-out coordinates to prevent contribution to log-sum-exp
    large_neg = torch.tensor(-1e10, device=output.device, dtype=output.dtype)
    
    # For loss_on: log-sum-exp only over on coordinates
    scaled_on_masked = torch.where(mask_on > 0, scaled_on, large_neg)
    loss_on_per_sample = torch.logsumexp(scaled_on_masked, dim=1)
    
    # For loss_off: log-sum-exp only over off coordinates
    scaled_off_masked = torch.where(mask_on < 1, scaled_off, large_neg)
    loss_off_per_sample = torch.logsumexp(scaled_off_masked, dim=1)
    
    # Determine scaling factor: use provided value or default to dimension d
    if logsumexp_scale is None:
        scale = float(output.shape[-1])  # Dimension d
    else:
        scale = logsumexp_scale
    
    # Average over batch and scale
    total_loss = scale * log_sum_exp_combined.mean()
    loss_on = scale * loss_on_per_sample.mean()
    loss_off = scale * loss_off_per_sample.mean()
    
    return total_loss, loss_on, loss_off


def dual_lambda_scaled_mse_loss(
    output: torch.Tensor,  # (B, d_1 + d_2)
    label: torch.Tensor,   # (B, d_1 + d_2)
    lambda_1: float,
    lambda_2: float,
    d_1: int,
    mask_on: Optional[torch.Tensor] = None,
    lambda_off_1: float = 1.0,
    lambda_off_2: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Dual lambda scaled MSE loss for two input types.
    
    Applies lambda_1 to coordinates 0..d_1-1 (type 1) and lambda_2 to coordinates d_1..d_1+d_2-1 (type 2).
    Also applies lambda_off_1 and lambda_off_2 to scale the off-coordinate losses for each type.
    
    Args:
        output: Model predictions (B, d_1 + d_2)
        label: Clean sparse signal (B, d_1 + d_2)
        lambda_1: Weight for active position loss in type 1 (coordinates 0..d_1-1)
        lambda_2: Weight for active position loss in type 2 (coordinates d_1..d_1+d_2-1)
        d_1: Dimension of input type 1
        mask_on: Binary mask of active positions (B, d_1 + d_2). If None, uses label as mask.
        lambda_off_1: Weight for off-coordinate loss in type 1 (default: 1.0)
        lambda_off_2: Weight for off-coordinate loss in type 2 (default: 1.0)
    
    Returns:
        (total_loss, loss_on_1, loss_on_2, loss_off)
        - total_loss: Combined loss
        - loss_on_1: Loss on type 1 active coordinates
        - loss_on_2: Loss on type 2 active coordinates
        - loss_off: Loss on off coordinates (both types, scaled)
    """
    if mask_on is None:
        mask_on = label
    
    # Split into type 1 and type 2
    output_1 = output[:, :d_1]
    output_2 = output[:, d_1:]
    label_1 = label[:, :d_1]
    label_2 = label[:, d_1:]
    mask_1 = mask_on[:, :d_1]
    mask_2 = mask_on[:, d_1:]
    
    # Compute losses separately for each type
    err_on_1 = (output_1 - label_1) * mask_1
    err_on_2 = (output_2 - label_2) * mask_2
    err_off_1 = output_1 * (1.0 - mask_1)
    err_off_2 = output_2 * (1.0 - mask_2)
    
    loss_on_1 = lambda_1 * err_on_1.pow(2).sum(dim=1).mean()
    loss_on_2 = lambda_2 * err_on_2.pow(2).sum(dim=1).mean()
    loss_off_1 = lambda_off_1 * err_off_1.pow(2).sum(dim=1).mean()
    loss_off_2 = lambda_off_2 * err_off_2.pow(2).sum(dim=1).mean()
    
    total_loss = loss_on_1 + loss_on_2 + loss_off_1 + loss_off_2
    return total_loss, loss_on_1, loss_on_2, loss_off_1 + loss_off_2


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
    elif cfg.loss_type == "scaled_lp":
        p = cfg.lp_power if cfg.lp_power is not None else 2.0
        def loss_fn(output, label, mask_on=None):
            return scaled_lp_loss(output, label, cfg.lambda_on, mask_on, p)
        return loss_fn
    elif cfg.loss_type == "logistic":
        def loss_fn(output, label, mask_on=None):
            return logistic_loss(output, label, cfg.lambda_on, mask_on, cfg.logsumexp_scale, cfg.lambda_off)
        return loss_fn
    elif cfg.loss_type == "dual_lambda_scaled_mse":
        if cfg.lambda_1 is None or cfg.lambda_2 is None:
            raise ValueError("dual_lambda_scaled_mse requires both lambda_1 and lambda_2 to be set")
        # Note: This loss function requires d_1 parameter, which should be passed separately
        # The factory will return a function that needs d_1
        lambda_off_1 = cfg.lambda_off_1 if cfg.lambda_off_1 is not None else 1.0
        lambda_off_2 = cfg.lambda_off_2 if cfg.lambda_off_2 is not None else 1.0
        def loss_fn(output, label, mask_on=None, d_1=None):
            if d_1 is None:
                raise ValueError("dual_lambda_scaled_mse requires d_1 parameter")
            return dual_lambda_scaled_mse_loss(
                output, label, cfg.lambda_1, cfg.lambda_2, d_1, mask_on,
                lambda_off_1=lambda_off_1, lambda_off_2=lambda_off_2
            )
        return loss_fn
    else:
        raise ValueError(f"Unknown loss type: {cfg.loss_type}")
